"""Tests for src.model.ncsnpp."""

import jax
import pytest

import equinox as eqx
import jax.numpy as jnp

from src.model.ncsnpp import NCSNpp

KEY = jax.random.PRNGKey(42)

# Small config for fast tests
SMALL_CFG = dict(
    in_channels=1,
    out_channels=1,
    base_channels=8,
    channel_multipliers=[1, 2],
    num_res_blocks=1,
    attn_resolutions=[4],
    dropout=0.0,
    num_groups=2,
    num_heads=1,
    activation=jax.nn.swish,
    fourier_scale=16.0,
    skip_rescale=True,
    image_size=8,
)


def test_ncsnpp_output_shape_matches_input():
    """Verify output shape matches input spatial dimensions and out_channels."""
    model = NCSNpp(**SMALL_CFG, key=KEY)
    x = jnp.ones((1, 8, 8))
    t = jnp.array(0.5)
    out = model(t, x)
    assert out.shape == (1, 8, 8), f"Expected (1, 8, 8), got {out.shape}"


def test_ncsnpp_different_t_gives_different_output():
    """Verify distinct timesteps produce distinct outputs."""
    model = NCSNpp(**SMALL_CFG, key=KEY)
    x = jnp.ones((1, 8, 8))
    out0 = model(jnp.array(0.0), x)
    out1 = model(jnp.array(1.0), x)
    assert not jnp.allclose(out0, out1)


def test_ncsnpp_output_finite():
    """Verify output contains only finite values for random input."""
    model = NCSNpp(**SMALL_CFG, key=KEY)
    k, _ = jax.random.split(KEY)
    x = jax.random.normal(k, (1, 8, 8))
    out = model(jnp.array(0.5), x)
    assert jnp.all(jnp.isfinite(out))


def test_ncsnpp_filter_vmap_over_batch():
    """Verify filter_vmap produces correct batch output shape."""
    model = NCSNpp(**SMALL_CFG, key=KEY)
    B = 3
    k, _ = jax.random.split(KEY)
    xs = jax.random.normal(k, (B, 1, 8, 8))
    ts = jnp.linspace(0.0, 1.0, B)
    outs = eqx.filter_vmap(model)(ts, xs)
    assert outs.shape == (B, 1, 8, 8)


def test_ncsnpp_three_levels():
    """Verify NCSN++ works with 3 resolution levels."""
    cfg = dict(
        in_channels=1, out_channels=1, base_channels=8,
        channel_multipliers=[1, 2, 2], num_res_blocks=1,
        attn_resolutions=[2], dropout=0.0, num_groups=2,
        num_heads=1, activation=jax.nn.swish, fourier_scale=16.0,
        skip_rescale=True, image_size=8,
    )
    model = NCSNpp(**cfg, key=KEY)
    x = jnp.ones((1, 8, 8))
    out = model(jnp.array(0.5), x)
    assert out.shape == (1, 8, 8)


def test_ncsnpp_multichannel():
    """Verify NCSN++ works with multiple input/output channels."""
    cfg = {**SMALL_CFG, "in_channels": 3, "out_channels": 3}
    model = NCSNpp(**cfg, key=KEY)
    x = jnp.ones((3, 8, 8))
    out = model(jnp.array(0.5), x)
    assert out.shape == (3, 8, 8)


def test_ncsnpp_gradient_flows():
    """Verify gradients flow through the model."""
    model = NCSNpp(**SMALL_CFG, key=KEY)
    x = jnp.ones((1, 8, 8))
    t = jnp.array(0.5)

    def loss_fn(m):
        return jnp.sum(m(t, x))

    grads = eqx.filter_grad(loss_fn)(model)
    grad_arrays = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    has_nonzero = any(jnp.any(g != 0.0) for g in grad_arrays)
    assert has_nonzero, "At least some gradients should be non-zero"


from src.flow.otfm import flow_matching_loss


def test_ncsnpp_flow_matching_loss():
    """Verify NCSNpp plugs into flow_matching_loss without error."""
    model = NCSNpp(**SMALL_CFG, key=KEY)
    B = 2
    k1, k2, k3 = jax.random.split(KEY, 3)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jax.random.uniform(k3, (B,))
    loss = flow_matching_loss(model, x0, x1, t)
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
