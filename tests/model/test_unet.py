"""Tests for msdflow.model.unet."""

import jax
import pytest

import equinox as eqx
import jax.numpy as jnp

from msdflow.model.unet import UNet

KEY = jax.random.PRNGKey(42)

# Small config for fast tests: 2 levels, 4 base channels
SMALL_CFG = dict(
    in_channels=1,
    out_channels=1,
    base_channels=4,
    channel_multipliers=[1, 2],
    num_res_blocks=1,
    num_heads=1,
    num_groups=2,
    activation=jax.nn.silu,
)

SMALL_CFG_COND = dict(
    in_channels=1,
    out_channels=1,
    base_channels=4,
    channel_multipliers=[1, 2],
    num_res_blocks=1,
    num_heads=1,
    num_groups=2,
    activation=jax.nn.silu,
    cond_dim=1,
)


def test_unet_output_shape_matches_input():
    """Verify output shape matches input spatial dimensions."""
    model = UNet(**SMALL_CFG, key=KEY)
    x = jnp.ones((1, 8, 8))
    t = jnp.array(0.5)
    out = model(t, x, jnp.empty(0), jnp.array(False))
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_unet_different_t_gives_different_output():
    """Verify distinct timesteps produce distinct UNet outputs."""
    model = UNet(**SMALL_CFG, key=KEY)
    x = jnp.ones((1, 8, 8))
    out0 = model(jnp.array(0.0), x, jnp.empty(0), jnp.array(False))
    out1 = model(jnp.array(1.0), x, jnp.empty(0), jnp.array(False))
    assert not jnp.allclose(out0, out1)


def test_unet_output_finite():
    """Verify UNet output contains only finite values for random input."""
    model = UNet(**SMALL_CFG, key=KEY)
    k, _ = jax.random.split(KEY)
    x = jax.random.normal(k, (1, 8, 8))
    out = model(jnp.array(0.5), x, jnp.empty(0), jnp.array(False))
    assert jnp.all(jnp.isfinite(out))


def test_unet_filter_vmap_over_batch():
    """Verify filter_vmap produces correct batch output shape."""
    model = UNet(**SMALL_CFG, key=KEY)
    B = 3
    k, _ = jax.random.split(KEY)
    xs = jax.random.normal(k, (B, 1, 8, 8))
    ts = jnp.linspace(0.0, 1.0, B)
    conds = jnp.empty((B, 0))
    masks = jnp.zeros(B, dtype=bool)
    outs = eqx.filter_vmap(model)(ts, xs, conds, masks)
    assert outs.shape == (B, 1, 8, 8)


def test_unet_cond_output_shape():
    """Verify conditional UNet output shape matches input."""
    model = UNet(**SMALL_CFG_COND, key=KEY)
    x = jnp.ones((1, 8, 8))
    t = jnp.array(0.5)
    cond = jnp.array([0.4])
    cond_mask = jnp.array(True)
    out = model(t, x, cond, cond_mask)
    assert out.shape == x.shape


def test_unet_cond_vs_uncond_differ():
    """Verify mask routes between real condition and null embedding."""
    model = UNet(**SMALL_CFG_COND, key=KEY)
    x = jnp.ones((1, 8, 8))
    t = jnp.array(0.5)
    cond_a = jnp.array([0.1])
    cond_b = jnp.array([0.9])

    # With mask=False, cond value should be ignored (null embedding used)
    out_uncond_a = model(t, x, cond_a, jnp.array(False))
    out_uncond_b = model(t, x, cond_b, jnp.array(False))
    assert jnp.allclose(out_uncond_a, out_uncond_b), (
        "Unconditional outputs should be identical regardless of cond value"
    )

    # With mask=True, different cond values should give different outputs
    out_cond_a = model(t, x, cond_a, jnp.array(True))
    out_cond_b = model(t, x, cond_b, jnp.array(True))
    assert not jnp.allclose(out_cond_a, out_cond_b), (
        "Conditional outputs should differ for different cond values"
    )


def test_unet_cond_different_cond_gives_different_output():
    """Verify distinct condition values produce distinct outputs."""
    model = UNet(**SMALL_CFG_COND, key=KEY)
    x = jnp.ones((1, 8, 8))
    t = jnp.array(0.5)
    out_a = model(t, x, jnp.array([0.1]), jnp.array(True))
    out_b = model(t, x, jnp.array([0.9]), jnp.array(True))
    assert not jnp.allclose(out_a, out_b)


def test_unet_cond_vmap_over_batch():
    """Verify filter_vmap works with conditional UNet."""
    model = UNet(**SMALL_CFG_COND, key=KEY)
    B = 3
    k, _ = jax.random.split(KEY)
    xs = jax.random.normal(k, (B, 1, 8, 8))
    ts = jnp.linspace(0.0, 1.0, B)
    conds = jnp.array([[0.1], [0.5], [0.9]])
    masks = jnp.array([True, False, True])
    outs = eqx.filter_vmap(model)(ts, xs, conds, masks)
    assert outs.shape == (B, 1, 8, 8)


def test_unet_cond_dim0_backward_compat():
    """Verify cond_dim=0 UNet works with dummy cond/mask args."""
    model = UNet(**SMALL_CFG, key=KEY)
    x = jnp.ones((1, 8, 8))
    t = jnp.array(0.5)
    cond = jnp.empty(0)
    cond_mask = jnp.array(False)
    out = model(t, x, cond, cond_mask)
    assert out.shape == x.shape


def test_unet_cond_dim_gt1_raises():
    """Verify cond_dim > 1 raises ValueError."""
    with pytest.raises(ValueError, match="not supported"):
        UNet(**{**SMALL_CFG, "cond_dim": 2}, key=KEY)


_KEY = jax.random.PRNGKey(0)
_SMALL = dict(
    in_channels=1, out_channels=1, base_channels=4,
    channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
    num_groups=2, activation=jax.nn.silu,
)


def test_unet_prediction_type_default():
    """UNet defaults to velocity prediction."""
    model = UNet(**_SMALL, key=_KEY)
    assert model.prediction_type == "velocity"


def test_unet_prediction_type_image():
    """UNet accepts prediction_type='image'."""
    model = UNet(**_SMALL, key=_KEY, prediction_type="image")
    assert model.prediction_type == "image"


def test_unet_prediction_type_invalid():
    """UNet raises ValueError for unknown prediction_type."""
    with pytest.raises(ValueError, match="prediction_type"):
        UNet(**_SMALL, key=_KEY, prediction_type="score")
