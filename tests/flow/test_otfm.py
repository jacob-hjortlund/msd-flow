"""Tests for src.flow.otfm."""

import jax
import pytest

import numpy as np
import equinox as eqx
import jax.numpy as jnp

from src.model.unet import UNet
from src.flow.otfm import flow_matching_loss, sample_path

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
    in_channels=1, out_channels=1, base_channels=4,
    channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
    num_groups=2, activation=jax.nn.silu, cond_dim=1, key=KEY,
)


def test_sample_path_at_t0_gives_x0():
    """Verify interpolant at t=0 equals x0."""
    x0 = jnp.ones((2, 1, 4, 4)) * 2.0
    x1 = jnp.ones((2, 1, 4, 4)) * 5.0
    t = jnp.zeros(2)
    x_t, u_t = sample_path(x0, x1, t)
    assert jnp.allclose(x_t, x0)


def test_sample_path_at_t1_gives_x1():
    """Verify interpolant at t=1 equals x1."""
    x0 = jnp.ones((2, 1, 4, 4)) * 2.0
    x1 = jnp.ones((2, 1, 4, 4)) * 5.0
    t = jnp.ones(2)
    x_t, u_t = sample_path(x0, x1, t)
    assert jnp.allclose(x_t, x1)


def test_sample_path_velocity_is_x1_minus_x0():
    """Verify linear interpolant velocity equals x1 - x0."""
    x0 = jnp.ones((2, 1, 4, 4)) * 2.0
    x1 = jnp.ones((2, 1, 4, 4)) * 5.0
    t = jnp.array([0.3, 0.7])
    _, u_t = sample_path(x0, x1, t)
    expected = x1 - x0
    assert jnp.allclose(u_t, expected)


def test_sample_path_shapes():
    """Verify x_t and u_t shapes match the input batch shape."""
    B, C, H, W = 3, 1, 4, 4
    x0 = jnp.zeros((B, C, H, W))
    x1 = jnp.ones((B, C, H, W))
    t = jnp.array([0.1, 0.5, 0.9])
    x_t, u_t = sample_path(x0, x1, t)
    assert x_t.shape == (B, C, H, W)
    assert u_t.shape == (B, C, H, W)


def test_flow_matching_loss_is_scalar():
    """Verify flow matching loss is a scalar."""
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    loss = flow_matching_loss(SMALL_MODEL, x0, x1, t, cond, cond_mask)
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
    loss = flow_matching_loss(SMALL_MODEL, x0, x1, t, cond, cond_mask)
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
    loss, grads = eqx.filter_value_and_grad(flow_matching_loss)(
        SMALL_MODEL, x0, x1, t, cond, cond_mask
    )
    # Check at least one grad leaf is non-zero
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
    loss = flow_matching_loss(SMALL_MODEL_COND, x0, x1, t, cond, cond_mask)
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
    loss = flow_matching_loss(SMALL_MODEL_COND, x0, x1, t, cond, cond_mask)
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
    loss, grads = eqx.filter_value_and_grad(flow_matching_loss)(
        SMALL_MODEL_COND, x0, x1, t, cond, cond_mask
    )
    grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(g != 0.0) for g in grad_leaves)


def test_sample_path_stochastic_x_t_differs_from_deterministic():
    """With nonzero sigma, x_t must differ from the noiseless interpolant."""
    import jax
    key = jax.random.PRNGKey(42)
    x0 = jnp.ones((2, 1, 4, 4)) * 2.0
    x1 = jnp.ones((2, 1, 4, 4)) * 5.0
    t = jnp.array([0.3, 0.7])
    x_t_det, _ = sample_path(x0, x1, t)
    x_t_stoch, _ = sample_path(x0, x1, t, sigma_0=0.1, sigma_1=0.1, key=key)
    assert not jnp.allclose(x_t_det, x_t_stoch)


def test_sample_path_stochastic_velocity_unchanged():
    """Velocity u_t must equal x1 - x0 regardless of sigma values."""
    import jax
    key = jax.random.PRNGKey(7)
    x0 = jnp.ones((2, 1, 4, 4)) * 2.0
    x1 = jnp.ones((2, 1, 4, 4)) * 5.0
    t = jnp.array([0.3, 0.7])
    _, u_t = sample_path(x0, x1, t, sigma_0=0.5, sigma_1=0.2, key=key)
    assert jnp.allclose(u_t, x1 - x0)


def test_sample_path_zero_sigma_matches_deterministic():
    """sigma_0=0, sigma_1=0 with a key provided must give the same result as no key."""
    import jax
    key = jax.random.PRNGKey(0)
    x0 = jnp.ones((2, 1, 4, 4)) * 2.0
    x1 = jnp.ones((2, 1, 4, 4)) * 5.0
    t = jnp.array([0.3, 0.7])
    x_t_a, u_t_a = sample_path(x0, x1, t)
    x_t_b, u_t_b = sample_path(x0, x1, t, sigma_0=0.0, sigma_1=0.0, key=key)
    assert jnp.allclose(x_t_a, x_t_b)
    assert jnp.allclose(u_t_a, u_t_b)
