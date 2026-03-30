"""Tests for msdflow.train.metrics."""

import jax
import pytest

import numpy as np
import equinox as eqx
import jax.numpy as jnp

from msdflow.model.unet import UNet
from msdflow.flow.interpolate import sample_path
from msdflow.train.metrics import flow_matching_loss, _to_velocity

KEY = jax.random.PRNGKey(0)

SMALL_MODEL = UNet(
    in_channels=1,
    out_channels=1,
    base_channels=4,
    channel_multipliers=[1, 2],
    num_res_blocks=1,
    num_heads=1,
    num_groups=2,
    activation=jax.nn.silu,
    key=KEY,
)

SMALL_MODEL_COND = UNet(
    in_channels=1,
    out_channels=1,
    base_channels=4,
    channel_multipliers=[1, 2],
    num_res_blocks=1,
    num_heads=1,
    num_groups=2,
    activation=jax.nn.silu,
    cond_dim=1,
    key=KEY,
)


def test_flow_matching_loss_is_scalar():
    """Verify flow matching loss is a scalar."""
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss = flow_matching_loss(SMALL_MODEL, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2))
    assert loss.shape == ()


def test_flow_matching_loss_is_positive():
    """Verify flow matching loss is non-negative."""
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss = flow_matching_loss(SMALL_MODEL, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2))
    assert loss >= 0.0


def test_flow_matching_loss_has_gradient():
    """Verify at least one gradient leaf is non-zero."""
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss, grads = eqx.filter_value_and_grad(flow_matching_loss)(
        SMALL_MODEL, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2)
    )
    grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(g != 0.0) for g in grad_leaves)


def test_flow_matching_loss_with_cond():
    """Verify flow matching loss works with conditioning."""
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.array([[0.4], [0.8]])
    cond_mask = jnp.ones(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss = flow_matching_loss(SMALL_MODEL_COND, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2))
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_flow_matching_loss_with_cond_mask_false():
    """Verify loss works when all conditions are masked (unconditional)."""
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.array([[0.4], [0.8]])
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss = flow_matching_loss(SMALL_MODEL_COND, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2))
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_flow_matching_loss_gradient_with_cond():
    """Verify gradients flow through conditional loss."""
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.array([[0.4], [0.8]])
    cond_mask = jnp.array([True, False])
    x_t, u_t = sample_path(x0, x1, t)
    loss, grads = eqx.filter_value_and_grad(flow_matching_loss)(
        SMALL_MODEL_COND, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2)
    )
    grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(g != 0.0) for g in grad_leaves)


def test_sample_path_stochastic_x_t_differs_from_deterministic():
    """With nonzero sigma, x_t must differ from the noiseless interpolant."""
    key = jax.random.PRNGKey(42)
    x0 = jnp.ones((2, 1, 4, 4)) * 2.0
    x1 = jnp.ones((2, 1, 4, 4)) * 5.0
    t = jnp.array([0.3, 0.7])
    x_t_det, _ = sample_path(x0, x1, t)
    x_t_stoch, _ = sample_path(x0, x1, t, sigma_0=0.1, sigma_1=0.1, key=key)
    assert not jnp.allclose(x_t_det, x_t_stoch)


def test_sample_path_stochastic_velocity_unchanged():
    """Velocity u_t must equal x1 - x0 regardless of sigma values."""
    key = jax.random.PRNGKey(7)
    x0 = jnp.ones((2, 1, 4, 4)) * 2.0
    x1 = jnp.ones((2, 1, 4, 4)) * 5.0
    t = jnp.array([0.3, 0.7])
    _, u_t = sample_path(x0, x1, t, sigma_0=0.5, sigma_1=0.2, key=key)
    assert jnp.allclose(u_t, x1 - x0)


def test_sample_path_zero_sigma_matches_deterministic():
    """sigma_0=0, sigma_1=0 with a key provided must give the same result as no key."""
    key = jax.random.PRNGKey(0)
    x0 = jnp.ones((2, 1, 4, 4)) * 2.0
    x1 = jnp.ones((2, 1, 4, 4)) * 5.0
    t = jnp.array([0.3, 0.7])
    x_t_a, u_t_a = sample_path(x0, x1, t)
    x_t_b, u_t_b = sample_path(x0, x1, t, sigma_0=0.0, sigma_1=0.0, key=key)
    assert jnp.allclose(x_t_a, x_t_b)
    assert jnp.allclose(u_t_a, u_t_b)


# --- _to_velocity ---


def test_to_velocity_velocity_mode_returns_pred():
    """In velocity mode, _to_velocity is an identity on pred."""
    pred = jnp.ones((2, 1, 4, 4)) * 3.0
    x_t = jnp.ones((2, 1, 4, 4)) * 1.0
    t = jnp.array([0.4, 0.6])
    v = _to_velocity(pred, x_t, t, "velocity")
    assert jnp.allclose(v, pred)


def test_to_velocity_image_mode_formula():
    """In image mode, _to_velocity applies (pred - x_t) / (1 - t)."""
    pred = jnp.ones((2, 1, 4, 4)) * 2.0
    x_t = jnp.ones((2, 1, 4, 4)) * 0.5
    t = jnp.array([0.5, 0.5])
    v = _to_velocity(pred, x_t, t, "image")
    expected = (pred - x_t) / (1.0 - t[:, None, None, None])
    assert jnp.allclose(v, expected)


def test_to_velocity_image_mode_shape():
    """Output shape matches input shape."""
    B, C, H, W = 3, 2, 8, 8
    pred = jax.random.normal(KEY, (B, C, H, W))
    x_t = jax.random.normal(KEY, (B, C, H, W))
    t = jnp.array([0.1, 0.5, 0.9])
    v = _to_velocity(pred, x_t, t, "image")
    assert v.shape == (B, C, H, W)


# --- Image-mode prediction tests ---

_KEY2 = jax.random.PRNGKey(99)
_SMALL_IMG_MODEL = UNet(
    in_channels=1,
    out_channels=1,
    base_channels=4,
    channel_multipliers=[1, 2],
    num_res_blocks=1,
    num_heads=1,
    num_groups=2,
    activation=jax.nn.silu,
    key=_KEY2,
    prediction_type="image",
)


def test_flow_matching_loss_image_mode_is_scalar():
    """Loss with an image-prediction model is a scalar."""
    B = 2
    k1, k2 = jax.random.split(_KEY2)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss = flow_matching_loss(_SMALL_IMG_MODEL, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2))
    assert loss.shape == ()


def test_flow_matching_loss_image_mode_is_finite():
    """Loss with an image-prediction model is finite."""
    B = 2
    k1, k2 = jax.random.split(_KEY2)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss = flow_matching_loss(_SMALL_IMG_MODEL, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2))
    assert jnp.isfinite(loss)


def test_flow_matching_loss_image_mode_has_gradient():
    """Gradients flow through image-mode loss."""
    B = 2
    k1, k2 = jax.random.split(_KEY2)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    _, grads = eqx.filter_value_and_grad(flow_matching_loss)(
        _SMALL_IMG_MODEL, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2)
    )
    grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(g != 0.0) for g in grad_leaves)
