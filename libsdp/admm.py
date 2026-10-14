"""
A CG solver in optax to use with ADMM
"""
import jax.flatten_util
import numpy as np
import h5py
import jax
import jax.numpy as jnp

import lineax as lx

import optax
from optax.tree_utils import tree_add, tree_scale, tree_add_scale, tree_vdot, tree_norm

from functools import partial

from libsdp.admm_utilities import Timer
from libsdp.admm_utilities import tree_real_matrix_inner_product
import time


def LHS_U(Uparams: optax.Params, Vparams: optax.Params, sigma, AX_fn, AT_fn):
    G_UV = tree_real_matrix_inner_product(Uparams, Vparams)
    A_GUV = AX_fn(G_UV)
    AT_AGUV = AT_fn(A_GUV)
    deriv_residual_u = tree_scale(
        1, 
        jax.tree_util.tree_map(lambda m1, m2: m1 @ m2, AT_AGUV, Vparams)
    )
    scaled_u = tree_scale(1, Uparams)
    return tree_add(scaled_u, deriv_residual_u)

def LHS_V(Uparams: optax.Params, Vparams: optax.Params, sigma, AX_fn, AT_fn):
    G_UV = tree_real_matrix_inner_product(Uparams, Vparams)
    A_GUV = AX_fn(G_UV)
    AT_AGUV = AT_fn(A_GUV)
    deriv_residual_v = tree_scale(
        1,
        jax.tree_util.tree_map(lambda m1, m2: m1.T @ m2, AT_AGUV, Uparams)
    )
    scaled_v = tree_scale(1, Vparams)
    return tree_add(scaled_v, deriv_residual_v)

def RHS_U(Vparams: optax.Params, yparams: optax.Params, btree: optax.Params, Ctree: optax.Params, sigma, AX_fn, AT_fn):
    sV = tree_scale(1, Vparams)
    sATbV = tree_scale(
        1, 
        jax.tree_util.tree_map(
            lambda m1, m2: m1 @ m2, 
            AT_fn(btree),
            Vparams
        )
    )
    ATyV = tree_scale(1/ sigma, 
    jax.tree_util.tree_map(
            lambda m1, m2: m1 @ m2, 
            AT_fn(yparams),
            Vparams
        )
        )
    mCV = tree_scale(
        -1 / sigma, 
        jax.tree_util.tree_map(lambda m1, m2: m1 @ m2, Ctree, Vparams
                               )
        )
    return tree_add(
        tree_add(sV, sATbV),
        tree_add(ATyV, mCV)
    )

def RHS_V(Uparams: optax.Params, yparams: optax.Params, btree: optax.Params, Ctree: optax.Params, sigma, AX_fn, AT_fn):
    sU = tree_scale(1, Uparams)
    sATbU = tree_scale(
        1, 
        jax.tree_util.tree_map(
            lambda m1, m2: m1.T @ m2, 
            AT_fn(btree),
            Uparams
        )
    )
    ATyU = tree_scale(1 / sigma, 
    jax.tree_util.tree_map(
            lambda m1, m2: m1.T @ m2, 
            AT_fn(yparams),
            Uparams
        )
        )
    mCU = tree_scale(
        -1 / sigma, 
        jax.tree_util.tree_map(lambda m1, m2: m1.T @ m2, Ctree, Uparams
                               )
        )
    return tree_add(
        tree_add(sU, sATbU),
        tree_add(ATyU, mCU)
    )

def get_C_pytree(dummy_params, CX_fn):
    # 2. Create the gradient function
    # We use partial to fix 'nmo' because it is a static argument
    grad_CX = jax.grad(CX_fn)

    # 3. Evaluate the gradient
    # The input value doesn't matter for a linear function, so zeros are fine.
    C_pytree = grad_CX(dummy_params)
    
    # 4. Symmeterize the matrix (sequentially to save memory)
    # We loop over keys to allow Python to free the unsymmetrized block k
    # before allocating block k+1. This avoids holding 2x the memory.
    # We also perform the symmetrization on the CPU to avoid peak memory usage
    # on the GPU (which requires holding val, val.T, and the result).
    if isinstance(C_pytree, dict):
        keys = list(C_pytree.keys())
        for k in keys:
            val = C_pytree[k]
            val_cpu = np.array(val)
            # Free GPU memory for this block before allocating the new one
            C_pytree[k] = None
            del val
            
            val_sym = 0.5 * (val_cpu + val_cpu.T)
            C_pytree[k] = jnp.array(val_sym)
        return C_pytree
    else:
        def sym_cpu(x):
            x_cpu = np.array(x)
            return jnp.array(0.5 * (x_cpu + x_cpu.T))
        return jax.tree.map(sym_cpu, C_pytree)

# @partial(jax.jit, static_argnames=('sigma', 'AX_fn', 'CX_fn'))
# @partial(jax.jit, static_argnames=('AX_fn', 'CX_fn'))
@jax.jit
def admm_augmented_lagrangian(
    Uparams: optax.Params,
    Vparams: optax.Params,
    yparams: optax.Params,
    btree: optax.Params,
    sigma: float,
    AX_fn,
    CX_fn
):
    G_UV = tree_real_matrix_inner_product(Uparams, Vparams)
    A_GUV = AX_fn(G_UV)
    residual = tree_add_scale(A_GUV, -1, btree)
    UV_diff = tree_add_scale(Uparams, -1, Vparams) 
    alm_val = \
        CX_fn(G_UV) - \
        tree_vdot(yparams, residual) + \
        (sigma / 2) * tree_vdot(residual, residual) + \
        (sigma / 2) * tree_vdot(UV_diff, UV_diff)
    return alm_val

# @partial(jax.jit, static_argnames=('sigma', 'AX_fn', 'CX_fn'))
# @partial(jax.jit, static_argnames=('AX_fn', 'CX_fn'))
@jax.jit
def compute_norm_and_residual(
    Uparams: optax.Params,
    Vparams: optax.Params,
    yparams: optax.Params,
    btree: optax.Params,
    sigma: float,
    AX_fn,
    CX_fn
):
    G_UV = tree_real_matrix_inner_product(Uparams, Vparams)
    A_GUV = AX_fn(G_UV)
    residual = tree_add_scale(A_GUV, -1, btree)
    UV_diff = tree_add_scale(Uparams, -1, Vparams) 
    primal_energy = CX_fn(G_UV)
    dual_energy = optax.tree_utils.tree_vdot(btree, yparams)
    residual_norm_sq = optax.tree_utils.tree_vdot(residual, residual)
    UV_diff_norm_sq = optax.tree_utils.tree_vdot(UV_diff, UV_diff)

    norm_residual = jnp.sqrt(residual_norm_sq)
    norm_UV_diff = jnp.sqrt(UV_diff_norm_sq)

    return residual, norm_residual, norm_UV_diff, primal_energy, dual_energy


# @partial(jax.jit, static_argnames=('max_steps', 'AX_fn', 'AT_fn'))
@jax.jit
def solve_for_Vparams_cg(
    Uparams: optax.Params, 
    init_V: optax.Params, 
    yparams: optax.Params, 
    btree: optax.Params, 
    Ctree: optax.Params, 
    sigma: float, 
    AX_fn,
    AT_fn,
    tol: float = 1e-5,
    max_steps: int = 100
):
    """
    Solves for Vparams using Conjugate Gradient via Lineax.
    System: LHS_V(V) = RHS_V(...)
    """
    rhs_vec = RHS_V(Uparams, yparams, btree, Ctree, sigma, AX_fn, AT_fn)
    # jax.debug.print("  [Device] Starting V-update CG solve...")

    def lhs_op_fn(v):
        # jax.debug.print("CG Step: |V| = {x}", x=jax.tree_util.tree_reduce(lambda a, b: a + jnp.sum(b**2), v, 0.0)) 
        return LHS_V(Uparams, v, sigma, AX_fn, AT_fn)

    v_structure = jax.eval_shape(lambda: init_V)
    operator = lx.FunctionLinearOperator(
        lhs_op_fn, 
        v_structure,
        tags=(lx.positive_semidefinite_tag, lx.symmetric_tag)
    )
    solver = lx.CG(rtol=0., atol=tol, max_steps=max_steps)
    solution = lx.linear_solve(
        operator, 
        rhs_vec, 
        solver=solver, 
        throw=False,
        options={"y0": init_V}
    )
    # Return the solved Vparams
    # verify residual
    # final_lhs = lhs_op_fn(solution.value)
    # final_residual = tree_add_scale(rhs_vec, -1, final_lhs)
    # final_res_norm = tree_norm(final_residual)
    # Use jax.debug.print for runtime values
    # jax.debug.print("  [DEBUG] V-CG explicit residual norm: {res} (Target: {tol}) Steps: {steps}", 
    #                 res=final_res_norm, tol=tol, steps=solution.stats['num_steps'])
    
    return solution.value, solution.stats

# @partial(jax.jit, static_argnames=('max_steps', 'AX_fn', 'AT_fn'))
@jax.jit
def solve_for_Uparams_cg(
    init_U: optax.Params, 
    Vparams: optax.Params, 
    yparams: optax.Params, 
    btree: optax.Params, 
    Ctree: optax.Params, 
    sigma: float, 
    AX_fn,
    AT_fn,
    tol: float = 1e-5,
    max_steps: int = 100
):
    """
    Solves for Uparams using Conjugate Gradient via Lineax.
    System: LHS_U(U) = RHS_U(...)
    """
    
    rhs_vec = RHS_U(Vparams, yparams, btree, Ctree, sigma, AX_fn, AT_fn)
    # jax.debug.print("  [Device] Starting U-update CG solve...")

    def lhs_op_fn(u):
        # jax.debug.print("CG Step: |U| = {x}", x=jax.tree_util.tree_reduce(lambda a, b: a + jnp.sum(b**2), u, 0.0))
        return LHS_U(u, Vparams, sigma, AX_fn, AT_fn)

    u_structure = jax.eval_shape(lambda: init_U)

    operator = lx.FunctionLinearOperator(
        lhs_op_fn, 
        u_structure,
        tags=(lx.positive_semidefinite_tag, lx.symmetric_tag)
    )

    solver = lx.CG(rtol=0.0, atol=tol, max_steps=max_steps)
    solution = lx.linear_solve(
        operator, 
        rhs_vec, 
        solver=solver, 
        throw=False,
        options={'y0': init_U}
    )
    # final_lhs = lhs_op_fn(solution.value)
    # final_residual = tree_add_scale(rhs_vec, -1, final_lhs)
    # final_res_norm = tree_norm(final_residual)
    # jax.debug.print("  [DEBUG] U-CG explicit residual norm: {res} (Target: {tol}) Steps: {steps}", 
    #                 res=final_res_norm, tol=tol, steps=solution.stats['num_steps'])

    return solution.value, solution.stats

def save_checkpoint(filename, Uparams, Vparams, yparams, sigma, iter_count=None):
    if iter_count is not None:
        print(f"Saving checkpoint at iteration {iter_count}...")
    else:
        print(f"Saving checkpoint to {filename}...")

    with h5py.File(filename, 'w') as h5f:
        u_grp = h5f.create_group('Uparams')
        for k, v in Uparams.items():
            u_grp.create_dataset(k, data=np.array(v))
        
        v_grp = h5f.create_group('Vparams')
        for k, v in Vparams.items():
            v_grp.create_dataset(k, data=np.array(v))
        
        y_grp = h5f.create_group('yparams')
        for k, v in yparams.items():
            y_grp.create_dataset(k, data=np.array(v))
        
        s_grp = h5f.create_group('sigma')
        for k, v in sigma.items():
            s_grp.create_dataset(k, data=np.array(v))

def admm_solver(
    Uparams,
    Vparams,
    yparams,
    sigma,
    btree,
    update_U_fn,
    update_V_fn,
    evaluate_fn,
    lagrangian_fn,
    maxiter=20,
    stopping_epsilon=1.0E-5,
    gap_stopping_epsilon=0.5E-4,
    update_frequency=5,
    sigma_scaling=1.0,
    tau=10.0,
    print_frequency=100,
    save_frequency=None,
    save_file_name='admm_checkpoint.h5',
    cg_max_iter=1000,
    cg_tol=1e-8,
    heuristic_factor=10.0,
    post_heuristic_sigma_scaling=1.2,
    post_heuristic_frequency=10
):
    """
    Generic ADMM solver loop.
    
    Args:
        state: Initial state (U, V, y, sigma)
        update_U_fn: Callable(U, V, y, sigma, cg_max_iter) -> U, stats
        update_V_fn: Callable(U, V, y, sigma, cg_max_iter) -> V, stats
        evaluate_fn: Callable(U, V, y, sigma) -> (residual, primal_norm, uv_diff, primal_en, dual_en)
        lagrangian_fn: Callable(U, V, y, sigma) -> float
    """
    iter_count = 0
    Vparams_old = Vparams
    
    residual_at_last_update = None
    reduction_factor = 0.6
    patience_counter = 0

    heuristic_applied = False
    heuristic_iter = 0
    
    print(f"ADMM Solver initialized with cg_tol={cg_tol}")
    print("Compiling JAX functions...")
    
    # Warmup U
    # Use the real cg_max_iter so max_steps (a static equinox field in lineax CG)
    # matches the main loop — avoids recompilation on first real iteration.
    # Use tol=1e30 so CG exits in ~1 step (atol is a dynamic field, no recompile).
    print(f"  [Warmup] Tracing & Compiling update_U_fn (max_steps={cg_max_iter})...")
    t0 = time.time()
    U_dummy, _ = update_U_fn(Uparams, Vparams, yparams, sigma, cg_max_iter=cg_max_iter, tol=1e30)
    print(f"  [Warmup] Dispatch (Compilation) finished in {time.time() - t0:.4f}s. Starting execution...")
    t0 = time.time()
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), U_dummy)
    print(f"  [Warmup] Execution finished in {time.time() - t0:.4f}s.")

    # Warmup V
    print(f"  [Warmup] Tracing & Compiling update_V_fn (max_steps={cg_max_iter})...")
    t0 = time.time()
    V_dummy, _ = update_V_fn(Uparams, Vparams, yparams, sigma, cg_max_iter=cg_max_iter, tol=1e30)
    print(f"  [Warmup] Dispatch (Compilation) finished in {time.time() - t0:.4f}s. Starting execution...")
    t0 = time.time()
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), V_dummy)
    print(f"  [Warmup] Execution finished in {time.time() - t0:.4f}s.")

    # Warmup others
    print("  [Warmup] Compiling evaluate_fn and lagrangian_fn...")
    t0 = time.time()
    eval_dummy = evaluate_fn(Uparams, Vparams, yparams, sigma)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), eval_dummy)
    lag_dummy = lagrangian_fn(Uparams, Vparams, yparams, sigma)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), lag_dummy)
    del eval_dummy, lag_dummy
    print(f"  [Warmup] evaluate_fn + lagrangian_fn compiled in {time.time() - t0:.4f}s.")
    
    del U_dummy, V_dummy
    b_inf = tree_norm(btree, ord=jnp.inf)

    # Compute initial metrics to set dynamic CG tolerance for the first iteration
    _, primal_norm_error_init, uv_diff_norm_init, primal_energy_init, dual_energy_init = \
        evaluate_fn(Uparams, Vparams, yparams, sigma)
    primal_dual_gap_init = jnp.abs(primal_energy_init - dual_energy_init)
    avg_metric = (primal_dual_gap_init + primal_norm_error_init + uv_diff_norm_init) / 3.0
    if primal_norm_error_init < 1e-4:
        cg_tol = 1e-8
    else:
        cg_tol = float(jnp.minimum(1e-0, avg_metric / 100.0))
        # cg_tol = float(avg_metric / 100.0)

    print(f"\t[ADMM] Initial dynamic CG tolerance: {cg_tol:.2e}")

    print("\t[ADMM] Starting main loop...")
    wall_time = time.time()
    while iter_count < maxiter:
        with Timer(verbose=False) as cg_step_timer:
            u_iters = 0
            Uparams, u_sol_stats = update_U_fn(Uparams, Vparams, yparams, sigma, cg_max_iter=cg_max_iter, tol=cg_tol)
            u_iters += int(u_sol_stats['num_steps'])
            
            v_iters = 0
            Vparams, v_sol_stats = update_V_fn(Uparams, Vparams, yparams, sigma, cg_max_iter=cg_max_iter, tol=cg_tol)
            v_iters += int(v_sol_stats['num_steps'])
                    
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), Vparams)

        # Compute residuals and metrics for update
        residuals, primal_norm_error, uv_diff_norm, primal_energy, dual_energy = \
            evaluate_fn(Uparams, Vparams, yparams, sigma)

        # Residual update (Dual update)
        yparams = tree_add_scale(yparams, -sigma['sigma'], residuals)
        
        # Compute ADMM-specific residuals for adaptive penalty
        Vp1_V_diff = tree_add_scale(Vparams, -1, Vparams_old)
        # Original code:
        UVk1_diff = tree_add_scale(Uparams, -1, Vparams)
        admm_dual_residual_norm = tree_scale(sigma['sigma'], tree_norm(Vp1_V_diff))
        admm_primal_residual_norm = tree_norm(UVk1_diff)
        
        # admm_dual_residual_norm = sigma['sigma'] * tree_norm(Vp1_V_diff)
        # admm_primal_residual_norm = jnp.hypot(primal_norm_error, uv_diff_norm)

        
        # Re-evaluate metrics if needed after y update? 
        # Actually in original code, evaluate_fn was called TWICE. 
        # First call was lines 354-361 (before y update).
        # Second call was lines 370-377 (after y update).
        # However, `yparams` in the second call was the UPDATED yparams.
        # But `residuals` (A(G)-b) does NOT depend on yparams!
        # `primal_norm_error` (norm of residuals) does NOT depend on yparams.
        # `uv_diff_norm` (norm of U-V) does NOT depend on yparams.
        # `primal_energy` (CX) does NOT depend on yparams.
        # `dual_energy` (y dot b) DOES depend on yparams.
        
        # So we can just re-evaluate dual energy?
        # Or just call evaluate_fn again to be safe and match original exact behavior suitable for callbacks.
        
        residuals, primal_norm_error, uv_diff_norm, primal_energy, dual_energy = \
            evaluate_fn(Uparams, Vparams, yparams, sigma)

        lagrangian_val = lagrangian_fn(Uparams, Vparams, yparams, sigma)
        
        primal_dual_gap = jnp.abs(primal_energy - dual_energy)
        num_cg_iters = u_iters + v_iters

        err_primal = primal_norm_error / (1 + b_inf)
        err_primal_inf = tree_norm(residuals, ord=jnp.inf) / (1 + b_inf)
        err_gap = primal_dual_gap / (1 + jnp.abs(primal_energy) + jnp.abs(dual_energy))
        err_consensus = uv_diff_norm / (1 + tree_norm(Vparams, ord=2))

        # Dynamic CG tolerance based on current convergence metrics
        avg_metric = (primal_dual_gap + primal_norm_error + uv_diff_norm) / 3.0
        # Original code:
        # if primal_norm_error < 1e-4:
        #     dynamic_cg_tol = 1e-8
        # else:
        #     dynamic_cg_tol = float(jnp.minimum(1e-1, avg_metric / 100.0))
        
        dynamic_cg_tol = float(jnp.maximum(1e-8, jnp.minimum(1e-1, avg_metric / 100.0)))
        cg_tol = dynamic_cg_tol


        if iter_count % print_frequency == 0 or iter_count == 0 or iter_count == maxiter - 1:
            print(f"{iter_count=}", 
                  f"{num_cg_iters=}",
                  f"CG time: {(cg_step_timer.end_time - cg_step_timer.start_time):.3e}",
                  f"sigma={sigma['sigma']:.2e}",
                  f"{cg_tol=:.2e}",
                  f"{lagrangian_val=:.6e}",
                  f"{primal_energy=:.6e}", 
                  f"{primal_dual_gap=:.6e}",  
                  f"{err_gap=:.6e}",
                  f"||A(G)-b||={primal_norm_error:.6e}",
                  f"||A(G)-b||_n={err_primal:.6e}",
                  f"||A(G)-b||_inf={err_primal_inf:.6e}",
                  f"||U-V||={uv_diff_norm:.6e}",
                  f"||U-V||_n={err_consensus:.6e}",
                  f"sigma*||V_k+1 - V_k||={admm_dual_residual_norm:.6e}",
                  # f"||U_k - V_k||={admm_primal_residual_norm:.6e}",
                )
        if save_frequency is not None and iter_count % save_frequency == 0 and iter_count > 0:
            save_checkpoint(save_file_name, Uparams, Vparams, yparams, sigma, iter_count)

        iter_count += 1


        # if iter_count % update_frequency == 0 and iter_count > 0:
        #     if admm_primal_residual_norm > tau * admm_dual_residual_norm:
        #         sigma['sigma'] *= sigma_scaling
        #     elif admm_dual_residual_norm > tau * admm_primal_residual_norm:
        #         sigma['sigma'] /= sigma_scaling

        # Adaptive Penalty Update
        if primal_norm_error > 5e-3: 
            # ONLY update sigma if we are far from the solution
            if iter_count % update_frequency == 0 and iter_count > 0:
                if admm_primal_residual_norm > tau * admm_dual_residual_norm:
                    sigma['sigma'] *= sigma_scaling
                elif admm_dual_residual_norm > tau * admm_primal_residual_norm:
                    sigma['sigma'] /= sigma_scaling
        else:
            if not heuristic_applied:
                sigma['sigma'] *= heuristic_factor
                heuristic_applied = True
                heuristic_iter = iter_count
                print(f"\tApplying heuristic factor: sigma={sigma['sigma']}")
            else:
                if (iter_count - heuristic_iter) % post_heuristic_frequency == 0:
                    print(f"\tUpdating penalty parameter by {post_heuristic_sigma_scaling}")
                    sigma['sigma'] *= post_heuristic_sigma_scaling
                    sigma['sigma'] = jnp.minimum(sigma['sigma'], 5_000.0)
                    if sigma['sigma'] == 5_000.0:
                        print("\tSigma reached maximum value")

        Vparams_old = Vparams

        # Original code:
        # if primal_norm_error < stopping_epsilon and primal_dual_gap < gap_stopping_epsilon:
        
        if primal_norm_error < stopping_epsilon and primal_dual_gap < gap_stopping_epsilon and uv_diff_norm < stopping_epsilon:
        # if err_primal_inf < stopping_epsilon and primal_dual_gap < gap_stopping_epsilon:
            print(f"Stopping criteria met at iteration {iter_count}")
            print(f"Stopping epsilon: {stopping_epsilon:.6e}")
            print(f"Gap stopping epsilon: {gap_stopping_epsilon:.6e}")
            print(f"primal norm error at exit: {primal_norm_error:.6e}")
            print(f"primal dual gap at exit: {primal_dual_gap:.6e}")
            print(f"consensus error at exit: {uv_diff_norm:.6e}")
            print(f"{primal_energy=:.6e}")
            print(f"{primal_dual_gap=:.6e}")
            print(f"{err_gap=:.6e}")
            print(f"||A(G)-b||={primal_norm_error:.6e}")
            print(f"||A(G)-b||_n={err_primal:.6e}")
            print(f"||A(G)-b||_inf={err_primal_inf:.6e}")
            print(f"||U-V||={uv_diff_norm:.6e}")
            print(f"||U-V||_n={err_consensus:.6e}")

            break
            
        if sigma['sigma'] > 5000:
            print("Sigma too large, exiting.")
            break


    print(f"ADMM finished after {iter_count} iterations.")
    print(f"ADMM Wall time {time.time() - wall_time:.4f}s")
    
    save_checkpoint(save_file_name, Uparams, Vparams, yparams, sigma)
    
    return Uparams, Vparams, yparams, sigma
