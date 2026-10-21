"""
a memory-lean boundary-point SDP solver (JAX) for SDPs with symmetric primal blocks

the boundary-point method of Povh, Rendl, and Wiegele, as used for v2RDM theory by Mazziotti,
with conjugate-gradient solves for the dual update. for memory:

  - A is stored folded: each off-diagonal pair (i,j), (j,i) of a block appears once, as the
    upper-triangle element with the full coefficient (v2rdm_sdp(..., fold_constraint_matrices
    = True), or fold() below). for symmetric x, A_folded x = A x, and A^T y = sym(A_folded^T y),
    so the symmetric A A^T used by CG is A_folded D A_folded^T, with D = 1/2 off the diagonal.

  - A is applied one of two ways (the matvec option):
      'scatter': three flat arrays, row (int32), column (int32), and value (float32 when
                 that is exact, as it is for pdaggerq coefficients), i.e. 12 bytes per
                 nonzero, applied in fixed-size chunks by gather / scatter-add, so no
                 temporary scales with the number of nonzeros or the primal dimension.
      'bcoo':    one jax BCOO matrix with int32 indices and float64 values (16 bytes per
                 nonzero), whose products lower to cuSPARSE on GPU. A^T y contracts A's row
                 index, so A^T is never formed.

  - blocks are diagonalized one at a time and written back into x and z in place, and the
    dual error comes from x without forming A^T y again: the projection gives
    sym(U) = mu x_new - z_new, so A^T y - c + z_new = mu (x_new - x_old).

the arrays that scale with the primal dimension are x, z, c, and U.
"""
import sys
import time

import numpy as np
import scipy.linalg
import scipy.sparse as sp

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse
from typing import NamedTuple


def fold(A, block_dimensions):
    """
    fold a symmetrized constraint matrix (or objective vector, as a 1 x n matrix): move each
    element of a block's lower triangle onto the upper triangle, summing coefficients

    :param A: scipy sparse matrix (n_rows, n_primal)
    :param block_dimensions: size of each block
    """
    A = sp.coo_matrix(A)
    dims = np.asarray(block_dimensions, dtype=np.int64)
    offs = np.cumsum(np.concatenate([[0], dims * dims]))
    blk = np.searchsorted(offs, A.col, side='right') - 1
    loc = A.col - offs[blk]
    i, j = loc // dims[blk], loc % dims[blk]
    col = offs[blk] + np.minimum(i, j) * dims[blk] + np.maximum(i, j)
    return sp.csr_matrix((A.data, (A.row, col)), shape=A.shape)


def symmetrize(c, block_dimensions):
    """
    0.5 (C + C^T) for every block of a flat vector (e.g. a folded objective vector)
    """
    c = np.array(c, dtype=float)
    off = 0
    for n in block_dimensions:
        C = c[off:off + n * n].reshape(n, n)
        c[off:off + n * n] = (0.5 * (C + C.T)).ravel()
        off += n * n
    return c


def _host_eigh(U):
    """
    host eigendecomposition with LAPACK's dsyevr, which, unlike dsyevd, handles the nearly
    degenerate spectra that make the device eigensolver return NaN near convergence
    """
    U = 0.5 * (np.asarray(U) + np.asarray(U).T)
    try:
        return scipy.linalg.eigh(U, driver='evr')
    except Exception:
        scale = max(float(np.abs(U).max()), 1.0)
        return scipy.linalg.eigh(U + np.diag(np.arange(len(U)) * 1e-13 * scale), driver='evr')


def _robust_eigh(U):
    """
    eigh on the device, repaired on the host if the device eigensolver fails (jnp.linalg.eigh
    returns NaN rather than raising)
    """
    w, v = jnp.linalg.eigh(U)
    ok = jnp.all(jnp.isfinite(w)) & jnp.all(jnp.isfinite(v))
    shapes = (jax.ShapeDtypeStruct(w.shape, w.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype))
    return lax.cond(ok, lambda _: (w, v), lambda M: jax.pure_callback(_host_eigh, shapes, M), U)


class LoopState(NamedTuple):
    bpsdp_iter: jnp.ndarray
    x: jnp.ndarray
    y: jnp.ndarray
    z: jnp.ndarray
    mu: jnp.ndarray
    primal_error: jnp.ndarray
    dual_error: jnp.ndarray
    gap: jnp.ndarray


def bpsdp(x, b, c, block_dimensions, A, options, maxiter=50000, matvec='scatter', chunk=2**24):
    """
    solve min c.x s.t. A x = b, x >= 0 (blockwise) by the boundary-point algorithm.

    :param x: initial guess for the primal solution
    :param b: the constraint vector
    :param c: the objective vector (folded or symmetric)
    :param block_dimensions: size of each block
    :param A: the constraint matrix (scipy sparse), folded (v2rdm_sdp(...,
              fold_constraint_matrices = True)) or symmetrized, which is folded here
    :param options: any object with the attributes sdp_error_convergence,
                    sdp_objective_convergence, cg_convergence, dynamic_cg_convergence,
                    cg_maxiter, and mu_update_frequency (e.g., a types.SimpleNamespace)
    :param maxiter: maximum number of bpsdp iterations
    :param matvec: how A and A^T are applied:
                   'scatter': chunked gather / scatter-add over (row, column, value) arrays,
                              12 bytes per nonzero and no temporary of primal size (leanest)
                   'bcoo':    one jax BCOO matrix (int32 indices, float64 values), 16 bytes per
                              nonzero; on GPU, A x and A^T y lower to cuSPARSE (faster), and the
                              CG right-hand side needs one temporary of primal size
    :param chunk: number of nonzeros of A applied at once ('scatter' only)
    :return x, y, z: primal solution, dual solution, and dual slack
    """
    assert matvec in ('scatter', 'bcoo'), "matvec must be 'scatter' or 'bcoo'"
    print('')
    print('    ==> BPSDP: Boundary-point SDP (JAX, lean) <==')
    print('')

    setup_start_time = time.perf_counter()

    n_dual, n_primal = A.shape
    dims = [int(n) for n in block_dimensions]
    offs = np.cumsum([0] + [n * n for n in dims])
    assert n_primal < 2**31 and n_dual < 2**31, 'int32 indices'

    # a symmetrized A has entries in the lower triangle of some block; fold it
    A = sp.csr_matrix(A)
    blk = np.searchsorted(offs, A.indices, side='right') - 1
    loc = A.indices - offs[blk]
    if np.any(loc // np.asarray(dims)[blk] > loc % np.asarray(dims)[blk]):
        A = fold(A, dims)
    del blk, loc

    # A as (row, column, value) triplets in row order
    A.sum_duplicates()
    A.sort_indices()
    rows = np.repeat(np.arange(n_dual, dtype=np.int32), np.diff(A.indptr))
    cols = A.indices.astype(np.int32)
    nnz = A.nnz
    if matvec == 'scatter':
        # float32 values when that is exact, padded to whole chunks
        vals = A.data.astype(np.float32)
        if not np.array_equal(vals.astype(np.float64), A.data):
            vals = A.data
        chunk = min(chunk, nnz)
        pad = -nnz % chunk
        rows = np.concatenate([rows, np.full(pad, n_dual - 1, dtype=np.int32)]).reshape(-1, chunk)
        cols = np.concatenate([cols, np.zeros(pad, dtype=np.int32)]).reshape(-1, chunk)
        vals = np.concatenate([vals, np.zeros(pad, dtype=vals.dtype)]).reshape(-1, chunk)
        matrix = (rows, cols, vals)
        nbytes = rows.nbytes + cols.nbytes + vals.nbytes
    else:
        indices = np.stack([rows, cols], axis=1)
        matrix = sparse.BCOO((jnp.asarray(A.data), jnp.asarray(indices)), shape=(n_dual, n_primal),
                             indices_sorted=True, unique_indices=True)
        nbytes = A.data.nbytes + indices.nbytes
        del indices
    del A, rows, cols

    # flat positions of the diagonal elements of every block
    diagonal = np.concatenate([off + np.arange(n) * (n + 1) for off, n in zip(offs[:-1], dims)]).astype(np.int32)

    c = symmetrize(c, dims)
    x = np.asarray(x, dtype=float)

    print('    n_primal = %10i' % (n_primal))
    print('    n_dual   = %10i' % (n_dual))
    print('    nnz(A)   = %10i  (%.2f GB, %s)' % (nnz, nbytes / 2**30, matvec))
    print('')
    print('    problem setup time: %10.3f s' % (time.perf_counter() - setup_start_time))
    print('')
    print('      oiter     c.x         b.y     |c.x-b.y|      mu    ||Ax-b|| ||ATy-c+z||')

    def run(matrix, diagonal, b, c, x):

        if matvec == 'scatter':
            rows, cols, vals = matrix

            def Ax(*terms):
                """A v for v = sum of (scale, vector) terms, gathered per nonzero so that v is never formed"""
                def body(k, out):
                    t = vals[k] * sum(scale * v[cols[k]] for scale, v in terms)
                    return out.at[rows[k]].add(t, indices_are_sorted=True)
                return lax.fori_loop(0, len(rows), body, jnp.zeros(n_dual))

            def ATy(v):
                def body(k, out):
                    return out.at[cols[k]].add(vals[k] * v[rows[k]])
                return lax.fori_loop(0, len(rows), body, jnp.zeros(n_primal))
        else:
            def Ax(*terms):
                """A v for v = sum of (scale, vector) terms"""
                return matrix @ sum(scale * v for scale, v in terms)

            def ATy(v):
                # contract A's row index with v: A^T v, without forming A^T
                return sparse.bcoo_dot_general(matrix, v, dimension_numbers=(([0], [0]), ([], [])))

        # the symmetric A A^T = A_folded D A_folded^T, with D = 1 on block diagonals, 1/2 elsewhere
        half_off_diagonal = lambda w: 0.5 * w.at[diagonal].multiply(2.0)
        AAT = lambda v: Ax((1.0, half_off_diagonal(ATy(v))))

        def cond_fun(s):
            converged = ((s.primal_error < options.sdp_error_convergence) &
                         (s.dual_error < options.sdp_error_convergence) &
                         (s.gap < options.sdp_objective_convergence))
            return (s.bpsdp_iter < maxiter) & ~converged

        def body_fun(s):
            tau = 1.0

            # step 1: solve A A^T y = A (c - z) + tau mu (b - A x) by conjugate gradient,
            # with the right-hand side as one pass over A: A (c - z - tau mu x) + tau mu b
            cg_rhs = Ax((1.0, c), (-1.0, s.z), (-tau * s.mu, s.x)) + tau * s.mu * b
            if options.dynamic_cg_convergence:
                cg_convergence = jnp.where(s.bpsdp_iter == 0, 0.01,
                                           0.01 * jnp.minimum(s.primal_error, s.dual_error))
            else:
                cg_convergence = options.cg_convergence
            y, _ = jax.scipy.sparse.linalg.cg(AAT, cg_rhs, x0=s.y, tol=0.0, atol=cg_convergence,
                                              maxiter=options.cg_maxiter)

            # step 2: U = mu x + A^T y - c, split blockwise into x = U+ / mu, z = -U-
            U = s.mu * s.x + ATy(y) - c
            x, z = s.x, s.z
            dx2 = 0.0
            for off, n in zip(offs[:-1], dims):
                Ub = U[off:off + n * n].reshape(n, n)
                w, v = _robust_eigh(0.5 * (Ub + Ub.T))
                xb = ((v * jnp.maximum(w, 0.0)) @ v.T / s.mu).ravel()
                zb = -((v * jnp.minimum(w, 0.0)) @ v.T).ravel()
                dx2 = dx2 + jnp.sum((xb - x[off:off + n * n]) ** 2)
                x = x.at[off:off + n * n].set(xb)
                z = z.at[off:off + n * n].set(zb)

            # step 3: errors and mu
            dual_error = s.mu * jnp.sqrt(dx2)
            primal_error = jnp.linalg.norm(Ax((1.0, x)) - b)
            obj_p = jnp.dot(c, x)
            obj_d = jnp.dot(b, y)
            gap = jnp.abs(obj_p - obj_d)
            mu = jnp.where((s.bpsdp_iter % options.mu_update_frequency == 0) & (s.bpsdp_iter > 0),
                           s.mu * primal_error / dual_error, s.mu)

            jax.debug.print(
                '      {i:5d} {obj_p:11.6f} {obj_d:11.6f} {gap:11.6e} {mu:7.3f} {pe:10.5e} {de:10.5e}',
                i=s.bpsdp_iter, obj_p=obj_p, obj_d=obj_d, gap=gap, mu=mu, pe=primal_error, de=dual_error)

            return LoopState(s.bpsdp_iter + 1, x, y, z, mu, primal_error, dual_error, gap)

        init = LoopState(jnp.int32(0), x, jnp.zeros(n_dual), jnp.zeros(n_primal), jnp.float64(0.1),
                         jnp.float64(1e6), jnp.float64(1e6), jnp.float64(1e6))
        return lax.while_loop(cond_fun, body_fun, init)

    solve_start_time = time.perf_counter()
    state = jax.block_until_ready(jax.jit(run)(matrix, diagonal, jnp.asarray(b), jnp.asarray(c), jnp.asarray(x)))
    total_solve_time = time.perf_counter() - solve_start_time
    n_iter = int(state.bpsdp_iter)

    if n_iter == maxiter:
        print('\n    error: bpsdp did not converge\n')
        sys.exit()

    print('\n    total bpsdp iterations: %10i' % (n_iter))
    print('    total bpsdp wall time:  %10.3f s  (includes one-time JIT compilation)' % (total_solve_time))
    print('    average time / iter:    %10.3f s\n' % (total_solve_time / n_iter))

    stats = jax.devices()[0].memory_stats()          # None on CPU
    if stats is not None:
        print('peak GPU memory: %.2f GB (of %.2f GB available to JAX)'
              % (stats['peak_bytes_in_use'] / 1e9, stats['bytes_limit'] / 1e9))

    return np.array(state.x), np.array(state.y), np.array(state.z)
