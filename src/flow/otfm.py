"""Optimal-transport flow matching loss and utilities.

Implements minibatch OT coupling, the linear interpolant path,
and the MSE flow matching objective.
"""

import jax
import numpy as np
import equinox as eqx
import jax.numpy as jnp

from scipy.optimize import linear_sum_assignment


def minibatch_ot_coupling(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Pair x0 ~ N(0,I) with x1 ~ p_data via minibatch optimal transport.

    Runs on NumPy/CPU, outside JAX JIT. Both inputs shape (B, C, H, W).
    Returns a permutation of x0 that minimises total squared L2 cost to x1.
    """
    B = x0.shape[0]
    x0_flat = x0.reshape(B, -1)
    x1_flat = x1.reshape(B, -1)
    # Pairwise squared L2 cost matrix (B, B)
    cost = np.sum((x0_flat[:, None, :] - x1_flat[None, :, :]) ** 2, axis=-1)
    _, col_ind = linear_sum_assignment(cost)
    return x0[col_ind]


def sample_ot_path(
    x0: jnp.ndarray, x1: jnp.ndarray, t: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Construct the linear OT interpolant and target velocity.

    Args:
        x0: shape (B, C, H, W) — noise samples (OT-coupled)
        x1: shape (B, C, H, W) — data samples
        t:  shape (B,) — per-sample time values in [0, 1]

    Returns:
        x_t: linear interpolant at time t
        u_t: target velocity (x1 - x0, constant along the path)
    """
    t_ = t[:, None, None, None]  # broadcast over (C, H, W)
    x_t = (1.0 - t_) * x0 + t_ * x1
    u_t = x1 - x0
    return x_t, u_t


def flow_matching_loss(
    model,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    t: jnp.ndarray,
    cond: jnp.ndarray,
    cond_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the flow matching MSE loss.

    Args:
        model: Velocity-field network accepting ``(t, x_t, cond, cond_mask)``.
        x0:    shape (B, C, H, W) — noise samples, OT-coupled to x1.
        x1:    shape (B, C, H, W) — data samples.
        t:     shape (B,) — per-sample times in [0, 1].
        cond:  shape (B, cond_dim) — conditioning vectors. Pass
            ``jnp.empty((B, 0))`` when the model is unconditional.
        cond_mask: shape (B,) bool — per-sample mask. ``True`` = use
            the real condition; ``False`` = use the null embedding.

    Returns:
        Scalar mean squared error between predicted and target velocities.
    """
    x_t, u_t = sample_ot_path(x0, x1, t)
    v_t = eqx.filter_vmap(model)(t, x_t, cond, cond_mask)
    return jnp.mean((v_t - u_t) ** 2)
