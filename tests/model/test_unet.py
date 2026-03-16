import jax
import pytest

import equinox as eqx
import jax.numpy as jnp

from src.model.unet import UNet

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


def test_unet_output_shape_matches_input():
    model = UNet(**SMALL_CFG, key=KEY)
    x = jnp.ones((1, 8, 8))
    t = jnp.array(0.5)
    out = model(x, t)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_unet_different_t_gives_different_output():
    model = UNet(**SMALL_CFG, key=KEY)
    x = jnp.ones((1, 8, 8))
    out0 = model(x, jnp.array(0.0))
    out1 = model(x, jnp.array(1.0))
    assert not jnp.allclose(out0, out1)


def test_unet_output_finite():
    model = UNet(**SMALL_CFG, key=KEY)
    k, _ = jax.random.split(KEY)
    x = jax.random.normal(k, (1, 8, 8))
    out = model(x, jnp.array(0.5))
    assert jnp.all(jnp.isfinite(out))


def test_unet_filter_vmap_over_batch():
    """filter_vmap maps over a batch of (x_t, t) pairs."""
    model = UNet(**SMALL_CFG, key=KEY)
    B = 3
    k, _ = jax.random.split(KEY)
    xs = jax.random.normal(k, (B, 1, 8, 8))
    ts = jnp.linspace(0.0, 1.0, B)
    outs = eqx.filter_vmap(model)(xs, ts)
    assert outs.shape == (B, 1, 8, 8)
