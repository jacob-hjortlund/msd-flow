"""Tests for shared CLR flow helpers."""

import jax
import jax.numpy as jnp
import pytest

from msdflow.flow.clr import (
    project_channel_mean_zero,
    sample_x0,
    validate_x0_mode,
)


def test_project_channel_mean_zero_preserves_shape_and_centers_channels():
    """Projection should center each sample/channel over spatial axes only."""
    x = jnp.arange(2 * 3 * 4 * 5, dtype=jnp.float32).reshape(2, 3, 4, 5)

    projected = project_channel_mean_zero(x)

    assert projected.shape == x.shape
    assert jnp.allclose(jnp.sum(projected, axis=(-2, -1)), 0.0, atol=1e-5)


def test_project_channel_mean_zero_centers_single_sample_per_channel():
    """Projection should also work for unbatched ``(C, H, W)`` samples."""
    x = jnp.arange(3 * 4 * 5, dtype=jnp.float32).reshape(3, 4, 5)

    projected = project_channel_mean_zero(x)

    assert projected.shape == x.shape
    assert jnp.allclose(jnp.sum(projected, axis=(-2, -1)), 0.0, atol=1e-5)


def test_sample_x0_clr_mode_projects_each_sample_channel():
    """CLR mode should sample Gaussian noise and project spatial means out."""
    x0 = sample_x0(jax.random.PRNGKey(0), (4, 2, 8, 8), x0_mode="clr")

    assert x0.shape == (4, 2, 8, 8)
    assert jnp.allclose(jnp.sum(x0, axis=(-2, -1)), 0.0, atol=1e-5)


def test_sample_x0_gaussian_mode_keeps_standard_gaussian_noise():
    """Gaussian mode should not force exact per-channel zero sums."""
    x0 = sample_x0(jax.random.PRNGKey(1), (4, 2, 8, 8), x0_mode="gaussian")

    assert x0.shape == (4, 2, 8, 8)
    assert not jnp.allclose(jnp.sum(x0, axis=(-2, -1)), 0.0, atol=1e-5)


def test_validate_x0_mode_rejects_unknown_mode():
    """Only gaussian and clr x0 modes should be accepted."""
    with pytest.raises(ValueError, match="x0_mode must be one of"):
        validate_x0_mode("bad-mode")
