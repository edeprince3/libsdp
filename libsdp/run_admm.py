import time
import numpy as np

import jax
import jax.numpy as jnp

from libsdp.admm import (
    admm_solver,
    get_C_pytree,
    solve_for_Uparams_cg,
    solve_for_Vparams_cg,
    compute_norm_and_residual,
    admm_augmented_lagrangian,
)

def get_pytree_memory_gb(pytree):
    leaves = jax.tree_util.tree_leaves(pytree)
    # Sum bytes of all leaves
    total_bytes = sum(leaf.size * leaf.itemsize for leaf in leaves)
    # Convert to GiB (1024^3 bytes)
    return total_bytes / (1024**3)

# Adapter functions for callbacks to avoid closure over large data structures (Tracer leakage / Constant folding)
def _update_U_adapter(U, V, y, sig, btree, Ctree, AX_fn, AT_fn, cg_max_iter, tol=1e-5):
    return solve_for_Uparams_cg(
        init_U=U,
        Vparams=V,
        yparams=y,
        btree=btree,
        Ctree=Ctree,
        sigma=sig['sigma'],
        AX_fn=AX_fn,
        AT_fn=AT_fn,
        tol=tol,
        max_steps=cg_max_iter
    )

def _update_V_adapter(U, V, y, sig, btree, Ctree, AX_fn, AT_fn, cg_max_iter, tol=1e-5):
    return solve_for_Vparams_cg(
        Uparams=U,
        init_V=V,
        yparams=y,
        btree=btree,
        Ctree=Ctree,
        sigma=sig['sigma'],
        AX_fn=AX_fn,
        AT_fn=AT_fn,
        tol=tol,
        max_steps=cg_max_iter
    )

def _evaluate_adapter(U, V, y, sig, btree, AX_fn, CX_fn):
    return compute_norm_and_residual(
        Uparams=U,
        Vparams=V,
        yparams=y,
        btree=btree,
        sigma=sig['sigma'],
        AX_fn=AX_fn,
        CX_fn=CX_fn
    )

def _lagrangian_adapter(U, V, y, sig, btree, AX_fn, CX_fn):
    return admm_augmented_lagrangian(
        Uparams=U,
        Vparams=V,
        yparams=y,
        btree=btree,
        sigma=sig['sigma'],
        AX_fn=AX_fn,
        CX_fn=CX_fn
    )

def run_admm(AX_fn, AT_fn, CX_fn, b_jax, shapes, block_names, 
             U_input=None, V_input=None, y_input=None, sigma_input=None,
             sigma=10.0, primal_rank=None, maxiter=50_000, stopping_epsilon=1.0E-5, gap_stopping_epsilon=0.5E-4,
             update_frequency=1,
             sigma_scaling=1.2, tau=10, print_frequency=100, save_frequency=1000,
             cg_max_iter=1_000, cg_tol=1.0E-8,
             heuristic_factor=10.0,
             post_heuristic_sigma_scaling=1.2,
             post_heuristic_frequency=10,
             ):
    """
    Run ADMM solver given the operator functions and initial parameters.

    Args:
        AX_fn: Linear map applied to primal variables (X).
        AT_fn: Adjoint of the linear map.
        CX_fn: Cost function map.
        b_jax: Target vector for the linear constraints.
        shapes: List of shapes for each block in the primal variables.
        block_names: List of names for each block in the primal variables.
        U_input: Initial parameters for the U block (primal).
        V_input: Initial parameters for the V block (auxiliary primal).
        y_input: Initial parameters for the dual variables.
        sigma_input: Initial penalty parameter.
        sigma: Initial penalty parameter value as float.
        primal_rank : list of int, or None
            If not None, a list of per-block ranks. Each U_k and V_k factor
            is initialized as (n_k, rank_k) so that G_k = U_k @ V_k^T is
            still (n_k x n_k).  When None, full-rank (n_k x n_k) factors
            are used (equivalent to rank_k = n_k for every block).
        maxiter: Maximum number of ADMM iterations.
        stopping_epsilon: Convergence tolerance for |AX-b|_inf / (1 + |b|_inf)
        gap_stopping_epsilon: Convergence tolerance for gap
        update_frequency: Frequency of printing/checking convergence.
        sigma_scaling: Scaling factor for penalty parameter updates.
        tau: Scaling frequency parameter for penalty.
        print_frequency: Frequency of printing progress.
        save_frequency: Frequency of saving checkpoint.
        cg_max_iter: Maximum CG iterations per ADMM block update.
        cg_tol: Tolerance for CG solver.
        heuristic_factor: Factor by which to scale sigma after the initial heuristic phase.
        post_heuristic_sigma_scaling: Factor by which to scale sigma after the initial heuristic phase.
        post_heuristic_frequency: Frequency of scaling sigma after the initial heuristic phase.

    Returns:
        Uparams, Vparams, yparams, sigma
    """
    if any(x is None for x in [U_input, V_input, y_input, sigma_input]):
        print("No initial conditions provided, initializing randomly.")
        # build U, V, y, sigma, btree params
        sigma = {'sigma': float(sigma)}
        yparams = {'y': jnp.zeros_like(b_jax)}
        Uparams = {}
        Vparams = {}
        for idx in range(len(block_names)):
            n = shapes[idx][0]
            rank = primal_rank[idx] if primal_rank is not None else n
            factor_shape = (n, rank)
            print(f"Initializing {block_names[idx]} with shape {factor_shape}")
            Uparams[block_names[idx]] = jnp.array(np.random.randn(*factor_shape))
            Vparams[block_names[idx]] = jnp.array(np.random.randn(*factor_shape))
    else:
        print("Using provided initial conditions.")
        Uparams = U_input
        Vparams = V_input
        yparams = y_input
        sigma = {'sigma': sigma_input}

    try:
        from g2_v2rdm.hdf5_helper import print_memory_stats
    except ImportError:
        def print_memory_stats(msg):
            pass

    print_memory_stats("After U, V creation") 
    print(f"Memory usage of U params: {get_pytree_memory_gb(Uparams):.4f} GB")
    print(f"Memory usage of V params: {get_pytree_memory_gb(Vparams):.4f} GB")

    btree = {'y': b_jax}
    print("Constructing C pytree for ADMM solver...")
    t0 = time.time()
    dummy_Xparams = {name: jnp.zeros(shape) for name, shape in zip(block_names, shapes)}
    C_pytree = get_C_pytree(dummy_params=dummy_Xparams, CX_fn=CX_fn)
    print(f"Finished constructing C pytree in {time.time() - t0:.4f} seconds.")

    print(f"Memory usage of C pytree: {get_pytree_memory_gb(C_pytree):.4f} GB")
    print(f"Memory usage of U params: {get_pytree_memory_gb(Uparams):.4f} GB")
    print(f"Memory usage of V params: {get_pytree_memory_gb(Vparams):.4f} GB")
    print(f"Memory usage of y params: {get_pytree_memory_gb(yparams):.4f} GB")
    print(f"Memory usage of btree: {get_pytree_memory_gb(btree):.4f} GB")

    print("Starting ADMM solver...")
    
    # Use jax.tree_util.Partial to bind arguments that should be treated as Pytrees (Traced)
    # This prevents JAX from constant-folding them, which is very slow for large BCOO matrices.
    update_U_fn = jax.tree_util.Partial(_update_U_adapter, btree=btree, Ctree=C_pytree, AX_fn=AX_fn, AT_fn=AT_fn)
    update_V_fn = jax.tree_util.Partial(_update_V_adapter, btree=btree, Ctree=C_pytree, AX_fn=AX_fn, AT_fn=AT_fn)
    evaluate_fn = jax.tree_util.Partial(_evaluate_adapter, btree=btree, AX_fn=AX_fn, CX_fn=CX_fn)
    lagrangian_fn = jax.tree_util.Partial(_lagrangian_adapter, btree=btree, AX_fn=AX_fn, CX_fn=CX_fn)
    
    t0 = time.time()
    Uparams, Vparams, yparams, sigma = admm_solver(
        Uparams=Uparams,
        Vparams=Vparams,
        yparams=yparams,
        sigma=sigma,
        btree=btree,
        update_U_fn=update_U_fn,
        update_V_fn=update_V_fn,
        evaluate_fn=evaluate_fn,
        lagrangian_fn=lagrangian_fn,
        maxiter=maxiter,
        stopping_epsilon=stopping_epsilon,
        gap_stopping_epsilon=gap_stopping_epsilon,
        update_frequency=update_frequency,
        sigma_scaling=sigma_scaling,
        tau=tau,
        print_frequency=print_frequency,
        save_frequency=save_frequency,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
        heuristic_factor=heuristic_factor,
        post_heuristic_sigma_scaling=post_heuristic_sigma_scaling,
        post_heuristic_frequency=post_heuristic_frequency,
    )
    
    return Uparams, Vparams, yparams, sigma
