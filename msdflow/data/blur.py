"""JAX-native Gaussian blur for use inside differentiable log-posteriors.

Unlike ``scipy.ndimage.gaussian_filter``, this implementation is fully
differentiable via ``jax.grad`` and compatible with ``jax.jit``.
"""

import jax.numpy as jnp
from jax import lax


def gaussian_kernel_2d(sigma: float, kernel_size: int) -> jnp.ndarray:
    """Return a normalised 2-D Gaussian kernel.

    Args:
        sigma:       Standard deviation in pixels.
        kernel_size: Side length of the square kernel. Should be odd.

    Returns:
        Array of shape ``(kernel_size, kernel_size)`` summing to 1.
    """
    coords = jnp.arange(kernel_size) - kernel_size // 2
    g = jnp.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel = jnp.outer(g, g)
    return kernel / kernel.sum()


def gaussian_blur(
    image: jnp.ndarray,
    sigma: float,
    kernel_size: int = 9,
) -> jnp.ndarray:
    """Apply Gaussian blur to a ``(C, H, W)`` JAX array.

    Uses a depthwise convolution (each channel blurred independently) via
    ``jax.lax.conv_general_dilated``. Differentiable w.r.t. ``image``.

    Args:
        image:       Input array of shape ``(C, H, W)``.
        sigma:       Gaussian standard deviation in pixels.
        kernel_size: Side length of the convolution kernel. Should be odd;
            defaults to 9 (covers ~±4σ for σ=2).

    Returns:
        Blurred array of shape ``(C, H, W)``.
    """
    C = image.shape[0]
    kernel = gaussian_kernel_2d(sigma, kernel_size)
    # Depthwise conv kernel shape: (out_channels, in_channels/groups, kH, kW)
    kernel_4d = jnp.tile(kernel[None, None], (C, 1, 1, 1))
    image_4d = image[None]  # (1, C, H, W)
    blurred = lax.conv_general_dilated(
        image_4d,
        kernel_4d,
        window_strides=(1, 1),
        padding="SAME",
        feature_group_count=C,  # depthwise: each channel uses its own kernel
    )
    return blurred[0]  # (C, H, W)
