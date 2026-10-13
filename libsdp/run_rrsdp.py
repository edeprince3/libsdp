"""Generic RRSDP solver using Burer-Choi exact line search with L-BFGS.

Mirrors the API of ``run_admm.py`` so both solvers are interchangeable.

The RRSDP (Rank-Restricted SDP) method parameterizes each PSD block as
X_k = R_k @ R_k^T and minimizes the augmented Lagrangian:

    L(R) = C·X - y^T·(A·X - b) + (σ/2)||A·X - b||²

via L-BFGS with an exact quartic line search (Burer & Choi).

The outer loop is a standard Augmented Lagrangian Method (ALM):
    1. Inner: minimize L(R) for fixed (y, σ) via L-BFGS.
    2. Outer: update y ← y - σ·(A·X - b), σ ← σ·scaling.

Sign convention
---------------
Residual:  res = A(X) - b          (same as ADMM code)
Lagrangian: L = CX - y^T·res + (σ/2)||res||²
Dual update: y ← y - σ·res        (gradient ascent on dual)

This ensures that at optimality y ≥ 0 and res = 0.
"""

import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from optax.tree_utils import tree_add, tree_scale, tree_add_scale, tree_vdot, tree_norm

from libsdp.jax_utilities import tree_real_matrix_inner_product
from libsdp.admm_utilities import Timer


# ─────────────────────────────────────────────────────────────────────
# Generic get_pq_fn factory
# ─────────────────────────────────────────────────────────────────────

def make_get_pq_fn(AX_fn, CX_fn):
    """Build a generic ``get_pq_fn`` closure over ``AX_fn`` and ``CX_fn``.

    The returned function computes the quartic polynomial coefficients
    for the exact line search, using only the generic AX and CX operators.

    Parameters
    ----------
    AX_fn : callable
        Linear constraint map ``X (pytree) -> y (pytree)``.
        For the sparse backend, ``AX_fn(X) = {'y': A @ flatten(X)}``.
    CX_fn : callable
        Cost function ``X (pytree) -> scalar``.
        ``CX_fn(X) = c^T · flatten(X)``.

    Returns
    -------
    get_pq_fn : callable
        ``(R_params, D_params, s_params) -> (p0, p1, p2, q0, q1, q2, a, b, c, d, e)``

    Notes
    -----
    Given R (current factors) and D (search direction), the primal at step α is:

        X(α) = (R + αD)(R + αD)^T = M_pp + α·M_pd_dp + α²·M_dd

    where:
        M_pp    = R @ R^T
        M_pd_dp = R @ D^T + D @ R^T
        M_dd    = D @ D^T

    The constraint residual is: res(α) = A(X(α)) - b = q₀ + α·q₁ + α²·q₂
    The cost is:                C(X(α)) = p₀ + α·p₁ + α²·p₂

    The augmented Lagrangian L(α) = aα⁴ + bα³ + cα² + dα + e  with coefficients
    documented in ``rrsdp_linesearch.py``.
    """

    @jax.jit
    def get_pq_fn(params, direction, s_params):
        """Compute quartic polynomial coefficients for the exact line search.

        Parameters
        ----------
        params : pytree
            Current R factor matrices (the optimization variables).
        direction : pytree
            Search direction D (same structure as params).
        s_params : dict
            Static parameters: ``{'sigma': float, 'btree': pytree, 'y': pytree}``.

        Returns
        -------
        Tuple of (p0, p1, p2, q0, q1, q2, a, b, c, d, e):
            - p0, p1, p2: cost polynomial coefficients
            - q0, q1, q2: residual polynomial coefficient pytrees
            - a, b, c, d, e: quartic Lagrangian polynomial coefficients
        """
        # Gram products (pytree of per-block matrix products)
        M_pp = tree_real_matrix_inner_product(params, params)
        M_dd = tree_real_matrix_inner_product(direction, direction)
        M_pd_dp = tree_add(
            tree_real_matrix_inner_product(params, direction),
            tree_real_matrix_inner_product(direction, params),
        )

        # Constraint residual polynomial coefficients
        # Following sosfermion convention: q₀ = b - A(M_pp) (NEGATIVE residual)
        # The quartic coefficient formulas are derived with this sign convention.
        # q₁ = A(M_pd_dp),  q₂ = A(M_dd)  (these are positive — the negative
        # sign is absorbed into the quartic coefficients via the expansion
        # res(α) = -q₀ + α·q₁ + α²·q₂ → ||res||² gives the quartic terms).
        q0 = tree_add_scale(s_params['btree'], -1, AX_fn(M_pp))  # b - A(M_pp)
        q1 = AX_fn(M_pd_dp)
        q2 = AX_fn(M_dd)

        # Cost polynomial coefficients
        p0 = CX_fn(M_pp)
        p1 = CX_fn(M_pd_dp)
        p2 = CX_fn(M_dd)

        # Assemble quartic coefficients
        sigma = s_params['sigma']
        y = s_params['y']
        y_sq0 = tree_add(y, tree_scale(sigma, q0))  # y + σ·q₀

        a = 0.5 * sigma * tree_vdot(q2, q2)
        b = sigma * tree_vdot(q1, q2)
        c = p2 - tree_vdot(y_sq0, q2) + 0.5 * sigma * tree_vdot(q1, q1)
        d = p1 - tree_vdot(y_sq0, q1)
        e = p0 + tree_vdot(y, q0) + 0.5 * sigma * tree_vdot(q0, q0)

        return p0, p1, p2, q0, q1, q2, a, b, c, d, e

    return get_pq_fn


# ─────────────────────────────────────────────────────────────────────
# Augmented Lagrangian evaluation
# ─────────────────────────────────────────────────────────────────────

def make_loss_fn(AX_fn, CX_fn):
    """Build the augmented Lagrangian loss function.

    Returns a JIT-compiled callable ``(R_params, s_params) -> scalar``.

    The Lagrangian is:
        L = C·X - y^T·(A·X - b) + (σ/2)||A·X - b||²

    where X = R @ R^T (computed per-block via tree_real_matrix_inner_product).
    """

    @jax.jit
    def loss_fn(R_params, s_params):
        X = tree_real_matrix_inner_product(R_params, R_params)
        AX_val = AX_fn(X)
        residual = tree_add_scale(AX_val, -1, s_params['btree'])  # A(X) - b

        cost = CX_fn(X)
        dual_term = tree_vdot(s_params['y'], residual)
        penalty = (s_params['sigma'] / 2) * tree_vdot(residual, residual)

        return cost - dual_term + penalty

    return loss_fn


# ─────────────────────────────────────────────────────────────────────
# Top-level RRSDP solver
# ─────────────────────────────────────────────────────────────────────

def run_rrsdp(
    AX_fn,
    CX_fn,
    b_jax,
    shapes,
    block_names,
    # --- Initialization ---
    sigma=10.0,
    primal_rank=None,
    # --- Outer ALM parameters ---
    maxiter=20,
    stopping_epsilon=1.0e-5,
    gap_stopping_epsilon=0.5e-4,
    sigma_scaling=10.0,
    # --- Inner L-BFGS parameters ---
    lbfgs_max_iter=50_000,
    lbfgs_history_size=6,
    lbfgs_tol_min=1e-4,
    # --- Output ---
    print_frequency=1,
    verbose=False,
):
    """Run the RRSDP solver on a generic SDP problem.

    API mirrors ``run_admm`` for interchangeability.

    Parameters
    ----------
    AX_fn : callable
        Linear constraint map ``X (pytree) -> y_vec (pytree)``.
    CX_fn : callable
        Cost function ``X (pytree) -> scalar``.
    b_jax : jnp.ndarray
        Constraint RHS vector.
    shapes : list of tuple
        Shape ``(n_k, n_k)`` for each primal block.
    block_names : list of str
        Name for each block (keys in the pytree dict).
    sigma : float
        Initial penalty parameter.
    primal_rank : int or None
        If not None, use rectangular R factors of shape ``(n_k, primal_rank)``
        instead of square ``(n_k, n_k)``.  This is the rank restriction that
        gives RRSDP its speed advantage for large blocks.
    maxiter : int
        Maximum number of outer ALM iterations.
    stopping_epsilon : float
        Convergence tolerance for ``||A(X) - b||``.
    gap_stopping_epsilon : float
        Convergence tolerance for the primal-dual gap.
    sigma_scaling : float
        Multiply σ by this factor each outer iteration.
    lbfgs_max_iter : int
        Max L-BFGS iterations per inner solve.
    lbfgs_history_size : int
        L-BFGS memory (number of past gradient pairs).
    lbfgs_tol_min : float
        Minimum floor for the dynamic L-BFGS gradient tolerance.
        Prevents wasting iterations when L-BFGS plateaus at high σ.
    print_frequency : int
        Print convergence info every N outer iterations.
    verbose : bool
        If True, print per-L-BFGS-iteration info.

    Returns
    -------
    R_params : dict
        Optimized R factor matrices.
    s_params : dict
        Final static parameters including y (multipliers) and sigma.
    """
    from libsdp.rrsdp_lbfgs import optimize_R_with_lbfgs

    # ── Initialize R factors ──────────────────────────────────────────
    R_params = {}
    for idx, (name, shape) in enumerate(zip(block_names, shapes)):
        n = shape[0]
        rank = primal_rank[idx] if primal_rank is not None else n
        R_params[name] = jnp.array(np.random.randn(n, rank))

    # ── Initialize static params ──────────────────────────────────────
    btree = {'y': jnp.array(b_jax)}
    y = {'y': jnp.zeros_like(jnp.array(b_jax))}

    s_params = {
        'sigma': sigma,
        'btree': btree,
        'y': y,
    }

    # ── Build closures ────────────────────────────────────────────────
    loss_fn = make_loss_fn(AX_fn, CX_fn)
    get_pq_fn = make_get_pq_fn(AX_fn, CX_fn)

    # ── Helper: compute convergence metrics ───────────────────────────
    @jax.jit
    def compute_metrics(R_params, s_params):
        X = tree_real_matrix_inner_product(R_params, R_params)
        AX_val = AX_fn(X)
        residual = tree_add_scale(AX_val, -1, s_params['btree'])
        residual_norm = jnp.sqrt(tree_vdot(residual, residual))
        primal_energy = CX_fn(X)
        dual_energy = tree_vdot(s_params['btree'], s_params['y'])
        return residual, residual_norm, primal_energy, dual_energy

    # ── Outer ALM loop ────────────────────────────────────────────────
    b_inf = float(tree_norm(btree, ord=jnp.inf))
    prev_objective = None  # Track objective change (C++ pattern)

    print(f"RRSDP Solver initialized: sigma={sigma:.2e}, "
          f"rank={'full' if primal_rank is None else primal_rank}, "
          f"lbfgs_history={lbfgs_history_size}")

    wall_time = time.time()
    for iter_count in range(maxiter):
        # Compute current residual for dynamic L-BFGS tolerance
        residual, residual_norm, primal_energy, dual_energy = \
            compute_metrics(R_params, s_params)
        primal_norm_error = float(residual_norm)

        # Dynamic L-BFGS tolerance (matches sosfermion strategy)
        # Floor at lbfgs_tol_min to avoid wasting iterations at high σ
        if iter_count == 0:
            lbfgs_tol = 0.01
        else:
            lbfgs_tol = max(0.01 * primal_norm_error, lbfgs_tol_min)

        # ── Inner L-BFGS solve ────────────────────────────────────────
        with Timer(verbose=False) as lbfgs_timer:
            R_params, num_iters, status_message = optimize_R_with_lbfgs(
                dynamic_params=R_params,
                static_params=s_params,
                loss_fn=loss_fn,
                get_pq_fn=get_pq_fn,
                max_iter=lbfgs_max_iter,
                tol=lbfgs_tol,
                history_size=lbfgs_history_size,
                verbose=verbose,
            )
            # Ensure computation is complete for timing
            jax.tree.map(lambda x: x.block_until_ready(), R_params)

        # ── Post-optimization metrics ─────────────────────────────────
        residual, residual_norm, primal_energy, dual_energy = \
            compute_metrics(R_params, s_params)

        primal_norm_error = float(residual_norm)
        primal_e_f = float(primal_energy)
        dual_e_f = float(dual_energy)
        primal_dual_gap = abs(primal_e_f - dual_e_f)
        lagrangian_val = float(loss_fn(R_params, s_params))
        mu = 1.0 / s_params['sigma']

        # Objective change (C++ convergence criterion)
        obj_change = abs(primal_e_f - prev_objective) if prev_objective is not None else float('inf')
        prev_objective = primal_e_f

        if iter_count % print_frequency == 0 or iter_count == 0:
            elapsed = lbfgs_timer.end_time - lbfgs_timer.start_time
            print(
                f"{iter_count=}",
                f"{num_iters=}",
                f"lbfgs_time={elapsed:.3e}",
                f"mu={mu:.2e}",
                f"{lagrangian_val=:.6e}",
                f"{primal_e_f=:.6e}",
                f"{primal_dual_gap=:.6e}",
                f"|dE|={obj_change:.6e}",
                f"||A(X)-b||={primal_norm_error:.6e}",
                f"{status_message=}",
            )

        # ── Convergence check (C++ pattern) ───────────────────────────
        # C++ checks: primal_error < sdp_error_convergence
        #          AND |obj_change| < sdp_objective_convergence
        converged = (primal_norm_error < stopping_epsilon and
                     obj_change < gap_stopping_epsilon)
        if converged:
            print(f"\nRRSDP converged at iteration {iter_count}:")
            print(f"  ||A(X)-b|| = {primal_norm_error:.6e} < {stopping_epsilon:.6e}")
            print(f"  |dE|       = {obj_change:.6e} < {gap_stopping_epsilon:.6e}")
            print(f"  primal_energy = {primal_e_f:.10e}")
            break

        # ── Dual variable update ──────────────────────────────────────
        # y ← y - σ·(A(X) - b)  [C++ line 245: y -= Au/mu]
        s_params['y'] = tree_add_scale(
            s_params['y'], -s_params['sigma'], residual
        )

        # ── Penalty parameter update ──────────────────────────────────
        s_params['sigma'] *= sigma_scaling

    total_wall = time.time() - wall_time
    print(f"\nRRSDP finished after {iter_count + 1} outer iterations. "
          f"Wall time: {total_wall:.4f}s")

    return R_params, s_params
