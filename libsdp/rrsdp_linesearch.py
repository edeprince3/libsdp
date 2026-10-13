"""Burer-Choi exact line search for RRSDP augmented Lagrangian.

Implements an optax GradientTransformationExtraArgs that performs an exact
line search along the L-BFGS descent direction.

Mathematical background
-----------------------
In RRSDP we parameterize each PSD block as X_k = R_k @ R_k^T.
The augmented Lagrangian is:

    L(R) = C·X(R) - y^T·res(R) + (σ/2)||res(R)||²

where  res_i = <A_i, X> - b_i   (per-constraint residual).

Along a search direction D,  R(α) = R + α·D  gives:

    X(α) = (R + αD)(R + αD)^T = M_pp + α·M_pd_dp + α²·M_dd

with:
    M_pp    = R @ R^T           (current primal)
    M_pd_dp = R @ D^T + D @ R^T (cross terms)
    M_dd    = D @ D^T           (direction outer product)

The constraint residual becomes a quadratic in α:

    res(α) = A·X(α) - b = q₀ + α·q₁ + α²·q₂

where  q₀ = A(M_pp) - b,  q₁ = A(M_pd_dp),  q₂ = A(M_dd).

The cost is also quadratic: C·X(α) = p₀ + α·p₁ + α²·p₂.

So L(α) is a **quartic polynomial** aα⁴ + bα³ + cα² + dα + e with:

    a = (σ/2) · ||q₂||²
    b = σ · ⟨q₁, q₂⟩
    c = p₂ - ⟨y + σ·q₀, q₂⟩ + (σ/2) · ||q₁||²
    d = p₁ - ⟨y + σ·q₀, q₁⟩
    e = p₀ + ⟨y, q₀⟩ + (σ/2) · ||q₀||²

The optimal step α* minimizes this quartic, found by solving the cubic
derivative 4aα³ + 3bα² + 2cα + d = 0.

Reference: S. Burer and R. D. C. Monteiro, "A nonlinear programming
algorithm for solving semidefinite programs via low-rank factorization,"
Mathematical Programming, 2003.
"""

from typing import Any, NamedTuple, Union

import jax
import jax.numpy as jnp
import optax
import optax.tree
from optax._src import base


class BurerChoiInfo(NamedTuple):
    """Debugging information for the Burer-Choi linesearch step.

    Attributes:
        num_linesearch_steps: Number of linesearch steps (always 0 for
            the exact method—kept for API compatibility).
        decrease_error: Not used; kept for API compatibility.
        curvature_error: Not used; kept for API compatibility.
    """
    num_linesearch_steps: Any
    decrease_error: Any
    curvature_error: Any


class ScaleByBurerChoiState(NamedTuple):
    """Optimizer state for :func:`scale_by_burer_choi_linesearch`.

    Attributes:
        value: Augmented Lagrangian value at the optimal α.
        grad: Gradient (or descent direction) used for the step.
        alpha: Optimal step size that minimizes L(R + α·D).
        info: Debugging information.
    """
    value: Any
    grad: base.Updates
    alpha: Any
    info: BurerChoiInfo


def scale_by_burer_choi_linesearch(
    verbose: bool = False,
) -> base.GradientTransformationExtraArgs:
    """Optax transform that replaces the line search with an exact quartic solve.

    This is plugged into ``optax.lbfgs(linesearch=...)`` to replace the
    default More-Thuente line search with the Burer-Choi exact line search.

    The ``update_fn`` expects the following extra keyword arguments:

    - ``get_pq_fn``: A callable ``(params, direction, s_params) -> (..., a, b, c, d, e)``
      that computes the quartic polynomial coefficients.
    - ``s_params``: Static parameters dict ``{'sigma', 'btree', 'y'}``.

    Returns:
        An ``optax.GradientTransformationExtraArgs``.
    """

    def init_fn(params):
        val_dtype = jnp.real(jax.tree.leaves(params)[0]).dtype
        return ScaleByBurerChoiState(
            value=jnp.asarray(jnp.inf, dtype=val_dtype),
            grad=optax.tree.zeros_like(params),
            alpha=jnp.asarray(0.0, dtype=val_dtype),
            info=BurerChoiInfo(
                num_linesearch_steps=jnp.asarray(0),
                decrease_error=jnp.asarray(jnp.inf),
                curvature_error=jnp.asarray(jnp.inf),
            ),
        )

    def update_fn(updates, state, params, *, s_params, get_pq_fn, **extra_args):
        """Compute optimal step size via quartic root finding.

        Args:
            updates: Descent direction D (pytree, same structure as params).
            state: Current optimizer state.
            params: Current R factors (pytree).
            s_params: Static params ``{'sigma', 'btree', 'y'}``.
            get_pq_fn: Callable that returns
                ``(p0, p1, p2, q0, q1, q2, a, b, c, d, e)``.

        Returns:
            Tuple of ``(α * updates, new_state)``.
        """
        # Compute quartic coefficients
        _, _, _, _, _, _, a, b, c, d, e = get_pq_fn(params, updates, s_params)

        # Cubic derivative: 4a·α³ + 3b·α² + 2c·α + d = 0
        grad_coeffs = jnp.array([4 * a, 3 * b, 2 * c, d])

        # Quartic objective: a·α⁴ + b·α³ + c·α² + d·α + e
        obj_coeffs = jnp.array([a, b, c, d, e])

        # Find roots of the cubic (strip_zeros=False for JIT compatibility)
        roots = jnp.roots(grad_coeffs, strip_zeros=False)

        # Filter to real, finite roots
        is_real = jnp.abs(roots.imag) < 1.0e-8
        is_finite = jnp.isfinite(roots)
        mask = is_real & is_finite

        # Evaluate quartic at each real root
        real_roots = roots.real
        obj_values = jnp.polyval(obj_coeffs, real_roots)

        # Mask invalid roots → +inf so argmin ignores them
        masked_obj_values = jnp.where(mask, obj_values, jnp.inf)

        # Pick the root that minimizes the quartic
        min_idx = jnp.argmin(masked_obj_values)
        alpha_star = real_roots[min_idx]

        # Build new state
        val_dtype = jnp.real(jax.tree.leaves(params)[0]).dtype
        new_state = ScaleByBurerChoiState(
            value=jnp.asarray(masked_obj_values[min_idx], dtype=val_dtype),
            grad=updates,
            alpha=jnp.asarray(alpha_star, dtype=val_dtype),
            info=BurerChoiInfo(
                num_linesearch_steps=jnp.asarray(0),
                decrease_error=jnp.asarray(jnp.inf),
                curvature_error=jnp.asarray(jnp.inf),
            ),
        )

        # Scale the direction by α*
        scaled_updates = optax.tree_utils.tree_scale(alpha_star, updates)
        return scaled_updates, optax.tree.cast_like(new_state, other_tree=state)

    return base.GradientTransformationExtraArgs(init_fn, update_fn)
