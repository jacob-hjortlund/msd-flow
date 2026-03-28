import jax
import equinox as eqx
import jax.numpy as jnp


def sample_time_uniform(
    key: jax.Array,
    batch_size: int,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> jnp.ndarray:
    """Sample times uniformly from [t_min, t_max].

    Args:
        key:        JAX PRNG key.
        batch_size: Number of time samples to draw.
        t_min:      Lower bound of the uniform distribution. Default 0.0.
        t_max:      Upper bound of the uniform distribution. Default 1.0.

    Returns:
        Array of shape (batch_size,) with values in [t_min, t_max].
    """
    return jax.random.uniform(key, (batch_size,), minval=t_min, maxval=t_max)


def sample_time_logit_normal(
    key: jax.Array,
    batch_size: int,
    mu: float = -0.8,
    sigma: float = 0.8,
    t_min: float = 1e-5,
    t_max: float = 0.99999,
) -> jnp.ndarray:
    """Sample times via a logit-normal distribution.

    Draws ``u ~ Normal(mu, sigma)`` then applies sigmoid to map to (0, 1).
    The default ``mu=-0.8, sigma=0.8`` biases samples toward the middle of the
    interval, following Esser et al. 2024 (Stable Diffusion 3).

    Args:
        key:        JAX PRNG key.
        batch_size: Number of time samples to draw.
        mu:         Mean of the underlying normal. Default -0.8.
        sigma:      Std-dev of the underlying normal. Default 0.8.
        t_min:      Lower bound of the output times. Default 1e-5.
        t_max:      Upper bound of the output times. Default 0.99999.

    Returns:
        Array of shape (batch_size,) with values in (t_min, t_max).
    """
    u = jax.random.normal(key, (batch_size,)) * sigma + mu
    t = jax.nn.sigmoid(u)
    t = jnp.clip(t, t_min, t_max)
    return t


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
