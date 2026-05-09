"""Shared helpers for CLR-compatible flow tensors."""

import jax
import jax.numpy as jnp

_SUPPORTED_X0_MODES = ("gaussian", "clr")


def project_channel_mean_zero(x: jnp.ndarray) -> jnp.ndarray:
    """Project spatial planes to zero mean independently per leading index.

    The projection subtracts the mean over the final two spatial axes only.
    For ``(B, C, H, W)`` tensors this preserves batch and channel axes, so
    every ``(sample, channel)`` plane is centered independently. For
    ``(C, H, W)`` tensors every channel plane is centered independently.

    Args:
        x: Array with at least two spatial dimensions.

    Returns:
        Array with the same shape as ``x`` and zero spatial mean over the final
        two axes.
    """
    return x - jnp.mean(x, axis=(-2, -1), keepdims=True)


def validate_x0_mode(x0_mode: str) -> str:
    """Validate an initial-noise sampling mode.

    Args:
        x0_mode: Initial-noise mode. Supported values are ``"gaussian"`` and
            ``"clr"``.

    Returns:
        The validated mode.

    Raises:
        ValueError: If ``x0_mode`` is unsupported.
    """
    if x0_mode not in _SUPPORTED_X0_MODES:
        raise ValueError(
            "x0_mode must be one of "
            f"{_SUPPORTED_X0_MODES}, got {x0_mode!r}"
        )
    return x0_mode


def sample_x0(
    key: jax.Array,
    shape: tuple[int, ...],
    x0_mode: str = "gaussian",
) -> jnp.ndarray:
    """Sample an initial flow state.

    Args:
        key: JAX PRNG key.
        shape: Shape of the sampled array.
        x0_mode: ``"gaussian"`` for standard Gaussian noise, or ``"clr"`` for
            Gaussian noise projected to zero spatial mean independently per
            sample and channel.

    Returns:
        Initial sample with shape ``shape``.

    Raises:
        ValueError: If ``x0_mode`` is unsupported.
    """
    validate_x0_mode(x0_mode)
    x0 = jax.random.normal(key, shape)
    if x0_mode == "clr":
        return project_channel_mean_zero(x0)
    return x0
