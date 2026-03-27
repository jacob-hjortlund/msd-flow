"""Optimal-transport flow matching loss and utilities.

Implements the linear interpolant path and the MSE flow matching objective.
"""

import jax
import equinox as eqx
import jax.numpy as jnp


def sample_path(
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    t: jnp.ndarray,
    sigma_0: float = 0.0,
    sigma_1: float = 0.0,
    key: jax.Array | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Construct the linear interpolant and target velocity, with optional noise.

    Args:
        x0: shape (B, C, H, W) — noise samples (already coupled to x1).
        x1: shape (B, C, H, W) — data samples.
        t:  shape (B,) — per-sample time values in [0, 1].
        sigma_0: Noise std at t=0. Default 0 (deterministic).
        sigma_1: Noise std at t=1. Default 0 (deterministic).
        key: JAX PRNG key required when sigma_0 or sigma_1 is nonzero.

    Returns:
        x_t: Interpolant at time t, optionally perturbed by Gaussian noise.
        u_t: Target velocity (x1 - x0), unchanged by noise.
    """
    t_ = t[:, None, None, None]  # broadcast over (C, H, W)
    x_t = (1.0 - t_) * x0 + t_ * x1
    u_t = x1 - x0
    if sigma_0 != 0.0 or sigma_1 != 0.0:
        sigma_t = (1.0 - t_) * sigma_0 + t_ * sigma_1
        eps = jax.random.normal(key, x0.shape)
        x_t = x_t + sigma_t * eps
    return x_t, u_t


def flow_matching_loss(
    model,
    x_t: jnp.ndarray,
    u_t: jnp.ndarray,
    t: jnp.ndarray,
    cond: jnp.ndarray,
    cond_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the flow matching MSE loss.

    Args:
        model: Velocity-field network accepting ``(t, x_t, cond, cond_mask)``.
        x_t:   shape (B, C, H, W) — interpolated samples at time t.
        u_t:   shape (B, C, H, W) — target velocities (x1 - x0).
        t:     shape (B,) — per-sample times in [0, 1].
        cond:  shape (B, cond_dim) — conditioning vectors. Pass
            ``jnp.empty((B, 0))`` when the model is unconditional.
        cond_mask: shape (B,) bool — per-sample mask. ``True`` = use
            the real condition; ``False`` = use the null embedding.

    Returns:
        Scalar mean squared error between predicted and target velocities.
    """
    v_t = eqx.filter_vmap(model)(t, x_t, cond, cond_mask)
    return jnp.mean((v_t - u_t) ** 2)
