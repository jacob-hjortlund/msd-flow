"""Tests for msdflow.model.ncsnpp."""

import jax
import pytest

import equinox as eqx
import jax.numpy as jnp

from msdflow.model.ncsnpp import NCSNpp

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

SMALL_CFG_COND = dict(
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
    cond_dim=1,
)


def test_ncsnpp_output_shape_matches_input():
    """Verify output shape matches input spatial dimensions and out_channels."""
    model = NCSNpp(**SMALL_CFG, key=KEY)
    x = jnp.ones((1, 8, 8))
    t = jnp.array(0.5)
    out = model(t, x, jnp.empty(0), jnp.array(False), jax.random.PRNGKey(0))
    assert out.shape == (1, 8, 8), f"Expected (1, 8, 8), got {out.shape}"


def test_ncsnpp_different_t_gives_different_output():
    """Verify distinct timesteps produce distinct outputs."""
    model = NCSNpp(**SMALL_CFG, key=KEY)
    x = jnp.ones((1, 8, 8))
    out0 = model(jnp.array(0.0), x, jnp.empty(0), jnp.array(False), jax.random.PRNGKey(0))
    out1 = model(jnp.array(1.0), x, jnp.empty(0), jnp.array(False), jax.random.PRNGKey(0))
    assert not jnp.allclose(out0, out1)


def test_ncsnpp_output_finite():
    """Verify output contains only finite values for random input."""
    model = NCSNpp(**SMALL_CFG, key=KEY)
    k, _ = jax.random.split(KEY)
    x = jax.random.normal(k, (1, 8, 8))
    out = model(jnp.array(0.5), x, jnp.empty(0), jnp.array(False), jax.random.PRNGKey(0))
    assert jnp.all(jnp.isfinite(out))


def test_ncsnpp_filter_vmap_over_batch():
    """Verify filter_vmap produces correct batch output shape."""
    model = NCSNpp(**SMALL_CFG, key=KEY)
    B = 3
    k, _ = jax.random.split(KEY)
    xs = jax.random.normal(k, (B, 1, 8, 8))
    ts = jnp.linspace(0.0, 1.0, B)
    keys = jax.random.split(jax.random.PRNGKey(0), B)
    outs = eqx.filter_vmap(model)(ts, xs, jnp.empty((B, 0)), jnp.zeros(B, dtype=bool), keys)
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
    out = model(jnp.array(0.5), x, jnp.empty(0), jnp.array(False), jax.random.PRNGKey(0))
    assert out.shape == (1, 8, 8)


def test_ncsnpp_multichannel():
    """Verify NCSN++ works with multiple input/output channels."""
    cfg = {**SMALL_CFG, "in_channels": 3, "out_channels": 3}
    model = NCSNpp(**cfg, key=KEY)
    x = jnp.ones((3, 8, 8))
    out = model(jnp.array(0.5), x, jnp.empty(0), jnp.array(False), jax.random.PRNGKey(0))
    assert out.shape == (3, 8, 8)


def test_ncsnpp_gradient_flows():
    """Verify gradients flow through the model."""
    model = NCSNpp(**SMALL_CFG, key=KEY)
    x = jnp.ones((1, 8, 8))
    t = jnp.array(0.5)

    def loss_fn(m):
        return jnp.sum(m(t, x, jnp.empty(0), jnp.array(False), jax.random.PRNGKey(0)))

    grads = eqx.filter_grad(loss_fn)(model)
    grad_arrays = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    has_nonzero = any(jnp.any(g != 0.0) for g in grad_arrays)
    assert has_nonzero, "At least some gradients should be non-zero"


def test_ncsnpp_cond_output_shape():
    """Verify conditional NCSNpp output shape matches input."""
    model = NCSNpp(**SMALL_CFG_COND, key=KEY)
    x = jnp.ones((1, 8, 8))
    t = jnp.array(0.5)
    cond = jnp.array([0.4])
    cond_mask = jnp.array(True)
    out = model(t, x, cond, cond_mask, jax.random.PRNGKey(0))
    assert out.shape == (1, 8, 8)


def test_ncsnpp_cond_vs_uncond_differ():
    """Verify mask routes between real condition and null embedding."""
    model = NCSNpp(**SMALL_CFG_COND, key=KEY)
    x = jnp.ones((1, 8, 8))
    t = jnp.array(0.5)
    cond_a = jnp.array([0.1])
    cond_b = jnp.array([0.9])

    # mask=False: cond value ignored, null embedding used — outputs must match
    out_uncond_a = model(t, x, cond_a, jnp.array(False), jax.random.PRNGKey(0))
    out_uncond_b = model(t, x, cond_b, jnp.array(False), jax.random.PRNGKey(0))
    assert jnp.allclose(out_uncond_a, out_uncond_b), (
        "Unconditional outputs should be identical regardless of cond value"
    )

    # mask=True: cond is used — different conds must give different outputs
    out_cond_a = model(t, x, cond_a, jnp.array(True), jax.random.PRNGKey(0))
    out_cond_b = model(t, x, cond_b, jnp.array(True), jax.random.PRNGKey(0))
    assert not jnp.allclose(out_cond_a, out_cond_b), (
        "Conditional outputs should differ for different cond values"
    )


def test_ncsnpp_cond_vmap_over_batch():
    """Verify filter_vmap works with conditional NCSNpp."""
    model = NCSNpp(**SMALL_CFG_COND, key=KEY)
    B = 3
    k, _ = jax.random.split(KEY)
    xs = jax.random.normal(k, (B, 1, 8, 8))
    ts = jnp.linspace(0.0, 1.0, B)
    conds = jnp.array([[0.1], [0.5], [0.9]])
    masks = jnp.array([True, False, True])
    keys = jax.random.split(jax.random.PRNGKey(0), B)
    outs = eqx.filter_vmap(model)(ts, xs, conds, masks, keys)
    assert outs.shape == (B, 1, 8, 8)


def test_ncsnpp_cond_dim_gt1_raises():
    """Verify cond_dim > 1 raises ValueError."""
    with pytest.raises(ValueError, match="not supported"):
        NCSNpp(**{**SMALL_CFG, "cond_dim": 2}, key=KEY)


def test_ncsnpp_cond_dim0_backward_compat():
    """Verify cond_dim=0 NCSNpp works with dummy cond/mask args."""
    model = NCSNpp(**SMALL_CFG, key=KEY)
    x = jnp.ones((1, 8, 8))
    t = jnp.array(0.5)
    cond = jnp.empty(0)
    cond_mask = jnp.array(False)
    out = model(t, x, cond, cond_mask, jax.random.PRNGKey(0))
    assert out.shape == (1, 8, 8)


from msdflow.flow.interpolate import sample_path
from msdflow.train.metrics import flow_matching_loss


def test_ncsnpp_flow_matching_loss():
    """Verify NCSNpp plugs into flow_matching_loss without error."""
    model = NCSNpp(**SMALL_CFG, key=KEY)
    B = 2
    k1, k2, k3 = jax.random.split(KEY, 3)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jax.random.uniform(k3, (B,))
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss = flow_matching_loss(model, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), B))
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"


def test_ncsnpp_prediction_type_default():
    """NCSNpp defaults to velocity prediction."""
    model = NCSNpp(**SMALL_CFG, key=KEY)
    assert model.prediction_type == "velocity"


def test_ncsnpp_prediction_type_image():
    """NCSNpp accepts prediction_type='image'."""
    model = NCSNpp(**SMALL_CFG, key=KEY, prediction_type="image")
    assert model.prediction_type == "image"


def test_ncsnpp_prediction_type_invalid():
    """NCSNpp raises ValueError for unknown prediction_type."""
    with pytest.raises(ValueError, match="prediction_type"):
        NCSNpp(**SMALL_CFG, key=KEY, prediction_type="score")


def test_ncsnpp_attention_dtype_bfloat16_smoke():
    """NCSNpp with attention_dtype=bfloat16 builds and runs; output dtype matches input."""
    cfg = dict(SMALL_CFG)
    cfg["attention_dtype"] = jnp.bfloat16
    model = NCSNpp(**cfg, key=KEY)
    k, _ = jax.random.split(KEY)
    x = jax.random.normal(k, (1, 8, 8)).astype(jnp.float32)
    out = model(jnp.array(0.5), x, jnp.empty(0), jnp.array(False), jax.random.PRNGKey(0))
    assert out.shape == (1, 8, 8)
    assert out.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(out))


def test_ncsnpp_attention_implementation_passthrough():
    """attention_implementation reaches every AttnBlockNCSN inside NCSNpp."""
    cfg = dict(SMALL_CFG)
    cfg["attention_implementation"] = "xla"
    model = NCSNpp(**cfg, key=KEY)
    # Bottleneck attention is the only guaranteed attention site for SMALL_CFG.
    assert model.mid_attn.attn.implementation == "xla"
