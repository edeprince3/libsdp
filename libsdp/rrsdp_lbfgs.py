"""L-BFGS optimization loop for RRSDP with Burer-Choi exact line search.

Provides two modes:
1. ``lax.while_loop`` (default): JIT-compiles the entire optimization into a
   single fused kernel — eliminates Python host overhead per iteration.
2. Plain Python loop: for debugging (per-iteration printing, breakpoints).

The outer ALM loop (in ``run_rrsdp.py``) remains plain Python.
"""

import jax
import jax.numpy as jnp
import optax
import optax.tree

from functools import partial

from libsdp.rrsdp_linesearch import scale_by_burer_choi_linesearch


# ─────────────────────────────────────────────────────────────────────
# JIT-compiled L-BFGS loop via lax.while_loop
# ─────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=('func', 'opt', 'max_iter', 'get_pq_fn', 'verbose'))
def _run_lbfgs_loop(dynamic_params, context_params, tol, *,
                     func, opt, max_iter, get_pq_fn, verbose=False):
    """JIT-compiled L-BFGS loop using ``jax.lax.while_loop``.

    Static args (keyword-only): func, opt, max_iter, get_pq_fn, verbose.
    Dynamic args (positional): dynamic_params, context_params, tol.

    Parameters
    ----------
    dynamic_params : pytree
        R factor matrices to optimize.
    context_params : dict
        Static params ``{'sigma', 'btree', 'y'}`` (fixed during inner loop).
    tol : float
        Gradient norm convergence tolerance.
    func : callable
        Loss function ``(R_params, s_params) -> scalar``.
    opt : optax optimizer
        ``optax.chain(scale_by_lbfgs(), scale_by_burer_choi_linesearch())``.
    max_iter : int
        Maximum number of L-BFGS iterations.
    get_pq_fn : callable
        Quartic polynomial coefficient function.
    verbose : bool
        If True, emit ``jax.debug.print`` inside the loop (compile-time flag).

    Returns
    -------
    final_params : pytree
        Optimized R factors.
    final_state : optax state
        Final optimizer state (contains iteration count, etc.).
    """
    def loss_wrapper(params):
        return func(params, context_params)

    value_and_grad_fn = jax.value_and_grad(loss_wrapper)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fn(params)
        iter_num = optax.tree.get(state, 'count')
        gnorm = optax.tree.norm(grad)

        # Python `if` on a static bool: the print code is completely
        # excluded from the JIT trace when verbose=False (zero cost).
        if verbose:
            jax.lax.cond(
                (iter_num % 50 == 0) | (iter_num < 5),
                lambda: jax.debug.print(
                    "    [while_loop] iter {i}: L = {v:.6e}, ||grad|| = {g:.6e}",
                    i=iter_num, v=value, g=gnorm),
                lambda: None,
            )

        updates, state = opt.update(
            grad, state, params,
            value=value,
            grad=grad,
            s_params=context_params,
            get_pq_fn=get_pq_fn,
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        params, state = carry
        _, grad = value_and_grad_fn(params)
        gnorm = optax.tree.norm(grad)
        xnorm = optax.tree.norm(params)
        # Match liblbfgs convergence: ||g|| / max(1, ||x||) < epsilon
        err = gnorm / jnp.maximum(1.0, xnorm)
        iter_num = optax.tree.get(state, 'count')

        # CRITICAL: Inside lax.while_loop, use JAX logical operators
        is_first_iter = (iter_num == 0)
        not_maxed = (iter_num < max_iter)
        not_converged = (err >= tol)

        return jnp.logical_or(is_first_iter,
                              jnp.logical_and(not_maxed, not_converged))

    init_state = opt.init(dynamic_params)
    init_carry = (dynamic_params, init_state)

    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state


# ─────────────────────────────────────────────────────────────────────
# Plain Python loop (for debugging / per-iteration tracking)
# ─────────────────────────────────────────────────────────────────────

def _run_lbfgs_python(dynamic_params, context_params, tol, *,
                       func, opt, max_iter, get_pq_fn, verbose=False):
    """Plain Python L-BFGS loop — for debugging and per-iteration logging.

    Same interface as ``_run_lbfgs_loop`` but runs in Python with
    per-iteration control flow.  Slower but allows printing and breakpoints.

    Returns
    -------
    final_params, num_iters, losses, grad_norms
    """
    @jax.jit
    def loss_wrapper(params):
        return func(params, context_params)

    value_and_grad_fn = jax.value_and_grad(loss_wrapper)

    state = opt.init(dynamic_params)
    params = dynamic_params
    losses = []
    grad_norms = []

    for i in range(max_iter):
        value, grad = value_and_grad_fn(params)
        gnorm = float(optax.tree_utils.tree_norm(grad))
        xnorm = float(optax.tree_utils.tree_norm(params))
        # Match liblbfgs: ||g|| / max(1, ||x||)
        rel_gnorm = gnorm / max(1.0, xnorm)
        fval = float(value)
        losses.append(fval)
        grad_norms.append(gnorm)

        if verbose and (i % 50 == 0 or i < 5):
            print(f"    iter {i:4d}: L = {fval:.6e}, ||grad|| = {gnorm:.6e}, "
                  f"||grad||/||x|| = {rel_gnorm:.6e}")

        if rel_gnorm < float(tol):
            if verbose:
                print(f"    Converged at iter {i}: ||grad||/||x|| = {rel_gnorm:.6e}")
            break

        updates, state = opt.update(
            grad, state, params,
            value=value,
            grad=grad,
            s_params=context_params,
            get_pq_fn=get_pq_fn,
        )
        params = optax.apply_updates(params, updates)

    return params, len(losses), losses, grad_norms


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def optimize_R_with_lbfgs(
    dynamic_params,
    static_params,
    loss_fn,
    get_pq_fn,
    max_iter=50_000,
    tol=1e-5,
    history_size=6,
    verbose=False,
    use_while_loop=True,
):
    """Minimize the augmented Lagrangian over R using L-BFGS + exact line search.

    Parameters
    ----------
    dynamic_params : dict
        Current R factors (pytree dict of JAX arrays).
    static_params : dict
        Context parameters ``{'sigma', 'btree', 'y'}`` (fixed during optimization).
    loss_fn : callable
        ``(R_params, s_params) -> scalar``.
    get_pq_fn : callable
        ``(R_params, direction, s_params) -> (p0, p1, p2, q0, q1, q2, a, b, c, d, e)``.
    max_iter : int
        Maximum L-BFGS iterations.
    tol : float
        Gradient norm tolerance.
    history_size : int
        L-BFGS memory size.
    verbose : bool
        If True, print per-iteration info (only in Python mode).
    use_while_loop : bool
        If True, use ``jax.lax.while_loop`` for maximum speed.
        If False, use plain Python loop for debugging.

    Returns
    -------
    Tuple of ``(optimized_params, num_iterations, status_message)``.
    """
    # Cache the optimizer so the same object identity is reused across calls.
    # Because `opt` is a static_argname in _run_lbfgs_loop, a new optimizer
    # object triggers full recompilation of the lax.while_loop (~8 GB leak).
    cache_key = (history_size,)
    if not hasattr(optimize_R_with_lbfgs, '_opt_cache'):
        optimize_R_with_lbfgs._opt_cache = {}
    if cache_key not in optimize_R_with_lbfgs._opt_cache:
        optimize_R_with_lbfgs._opt_cache[cache_key] = optax.chain(
            optax.scale_by_lbfgs(memory_size=history_size, scale_init_precond=True),
            scale_by_burer_choi_linesearch(verbose=False),
        )
    optimizer = optimize_R_with_lbfgs._opt_cache[cache_key]

    # if use_while_loop:
    # JIT-fused loop
    final_params, final_state = _run_lbfgs_loop(
        dynamic_params,
        static_params,
        tol,
        func=loss_fn,
        opt=optimizer,
        max_iter=max_iter,
        get_pq_fn=get_pq_fn,
        verbose=verbose,
    )

    # Extract iteration count
    count_value = optax.tree.get(final_state, 'count')
    total_iterations = int(count_value) if count_value is not None else max_iter

    # Compute final gradient norm (relative, matching liblbfgs)
    # Use loss_fn directly (stable object) to avoid jax.grad re-tracing
    # a new closure on every call.
    final_grad = jax.grad(loss_fn, argnums=0)(final_params, static_params)
    grad_norm = float(optax.tree.norm(final_grad))
    x_norm = float(optax.tree.norm(final_params))
    rel_grad_norm = grad_norm / max(1.0, x_norm)

    if rel_grad_norm <= tol:
        status_message = f"Converged (||g||/||x|| {rel_grad_norm:.6e} <= tol {tol:.6e})"
    elif total_iterations >= max_iter:
        status_message = f"Max iters ({max_iter}) reached. ||g||/||x||: {rel_grad_norm:.6e}"
    else:
        status_message = f"Stopped (||g||/||x|| {rel_grad_norm:.6e}, iters {total_iterations})"

    return final_params, total_iterations, status_message

    # else:
    #     # Python loop
    #     final_params, num_iters, losses, grad_norms = _run_lbfgs_python(
    #         dynamic_params,
    #         static_params,
    #         tol,
    #         func=loss_fn,
    #         opt=optimizer,
    #         max_iter=max_iter,
    #         get_pq_fn=get_pq_fn,
    #         verbose=verbose,
    #     )

    #     grad_norm = grad_norms[-1] if grad_norms else float('inf')
    #     x_norm = float(optax.tree_utils.tree_norm(final_params))
    #     rel_grad_norm = grad_norm / max(1.0, x_norm)
    #     if rel_grad_norm <= tol:
    #         status_message = f"Converged (||g||/||x|| {rel_grad_norm:.6e} <= tol {tol:.6e})"
    #     elif num_iters >= max_iter:
    #         status_message = f"Max iters ({max_iter}) reached. ||g||/||x||: {rel_grad_norm:.6e}"
    #     else:
    #         status_message = f"Stopped (||g||/||x|| {rel_grad_norm:.6e}, iters {num_iters})"

    #     return final_params, num_iters, status_message
