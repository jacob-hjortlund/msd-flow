"""Tests for msdflow.model.inceptionv3."""

import jax.numpy as jnp

from msdflow.model.inceptionv3 import reshape


def test_inceptionv3_reshape_converts_channel_first_grayscale_to_rgb():
    """reshape converts a (1, H, W) image into a 299x299 RGB image."""
    x = jnp.ones((1, 16, 16))

    out = reshape(x)

    assert out.shape == (299, 299, 3)
    assert out.dtype == x.dtype
