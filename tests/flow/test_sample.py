"""Tests for src.flow.sample."""

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import pytest
from src.model.unet import UNet
from src.flow.sample import sample

KEY = jax.random.PRNGKey(0)

SMALL_MODEL = UNet(
    in_channels=1, out_channels=1, base_channels=4,
    channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
    num_groups=2, activation=jax.nn.silu, key=KEY,
)

SMALL_MODEL_COND = UNet(
    in_channels=1, out_channels=1, base_channels=4,
    channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
    num_groups=2, activation=jax.nn.silu, cond_dim=1, key=KEY,
)


def test_sample_output_shape():
    """Verify sample output shape matches the requested shape."""
    out = sample(
        model=SMALL_MODEL,
        shape=(1, 8, 8),
        key=KEY,
        solver=diffrax.Euler,
        dt0=0.1,
        t0=0.0,
        t1=1.0,
        stepsize_controller=diffrax.ConstantStepSize,
        stepsize_controller_cfg={},
    )
    assert out.shape == (1, 8, 8)


def test_sample_output_finite():
    """Verify sample output contains only finite values."""
    out = sample(
        model=SMALL_MODEL,
        shape=(1, 8, 8),
        key=KEY,
        solver=diffrax.Euler,
        dt0=0.1,
        t0=0.0,
        t1=1.0,
        stepsize_controller=diffrax.ConstantStepSize,
        stepsize_controller_cfg={},
    )
    assert jnp.all(jnp.isfinite(out))


def test_sample_batched_via_vmap():
    """Verify batched sampling via vmap produces correct batch shape."""
    keys = jax.random.split(KEY, 3)
    batched_sample = jax.vmap(
        lambda k: sample(
            SMALL_MODEL, (1, 8, 8), k,
            diffrax.Euler, 0.1, 0.0, 1.0, diffrax.ConstantStepSize, {},
        )
    )
    outs = batched_sample(keys)
    assert outs.shape == (3, 1, 8, 8)


def test_sample_conditional():
    """Verify conditional sampling produces correct shape and finite output."""
    out = sample(
        model=SMALL_MODEL_COND,
        shape=(1, 8, 8),
        key=KEY,
        solver=diffrax.Euler,
        dt0=0.1,
        t0=0.0,
        t1=1.0,
        stepsize_controller=diffrax.ConstantStepSize,
        stepsize_controller_cfg={},
        cond=jnp.array([0.4]),
    )
    assert out.shape == (1, 8, 8)
    assert jnp.all(jnp.isfinite(out))


def test_sample_guided():
    """Verify guided sampling (scale > 1) produces correct shape and finite output."""
    out = sample(
        model=SMALL_MODEL_COND,
        shape=(1, 8, 8),
        key=KEY,
        solver=diffrax.Euler,
        dt0=0.1,
        t0=0.0,
        t1=1.0,
        stepsize_controller=diffrax.ConstantStepSize,
        stepsize_controller_cfg={},
        cond=jnp.array([0.4]),
        guidance_scale=2.0,
    )
    assert out.shape == (1, 8, 8)
    assert jnp.all(jnp.isfinite(out))


def test_sample_guided_differs_from_unguided():
    """Verify guided (scale=3) and unguided (scale=1) outputs differ."""
    kwargs = dict(
        model=SMALL_MODEL_COND,
        shape=(1, 8, 8),
        key=KEY,
        solver=diffrax.Euler,
        dt0=0.1,
        t0=0.0,
        t1=1.0,
        stepsize_controller=diffrax.ConstantStepSize,
        stepsize_controller_cfg={},
        cond=jnp.array([0.4]),
    )
    out_unguided = sample(**kwargs, guidance_scale=1.0)
    out_guided = sample(**kwargs, guidance_scale=3.0)
    assert not jnp.allclose(out_unguided, out_guided)


_SMALL_IMG = UNet(
    in_channels=1, out_channels=1, base_channels=4,
    channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
    num_groups=2, activation=jax.nn.silu, key=KEY,
    prediction_type="image",
)

_SMALL_IMG_COND = UNet(
    in_channels=1, out_channels=1, base_channels=4,
    channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
    num_groups=2, activation=jax.nn.silu, cond_dim=1, key=KEY,
    prediction_type="image",
)


def test_sample_image_prediction_shape():
    """sample() with an image-prediction model returns correct shape."""
    out = sample(
        model=_SMALL_IMG,
        shape=(1, 8, 8),
        key=KEY,
        solver=diffrax.Euler,
        dt0=0.1,
        t0=0.0,
        t1=0.9,
        stepsize_controller=diffrax.ConstantStepSize,
        stepsize_controller_cfg={},
    )
    assert out.shape == (1, 8, 8)


def test_sample_image_prediction_finite():
    """sample() with an image-prediction model returns finite values."""
    out = sample(
        model=_SMALL_IMG,
        shape=(1, 8, 8),
        key=KEY,
        solver=diffrax.Euler,
        dt0=0.1,
        t0=0.0,
        t1=0.9,
        stepsize_controller=diffrax.ConstantStepSize,
        stepsize_controller_cfg={},
    )
    assert jnp.all(jnp.isfinite(out))


def test_sample_image_prediction_guided_shape():
    """Guided sampling with an image-prediction model returns correct shape."""
    out = sample(
        model=_SMALL_IMG_COND,
        shape=(1, 8, 8),
        key=KEY,
        solver=diffrax.Euler,
        dt0=0.1,
        t0=0.0,
        t1=0.9,
        stepsize_controller=diffrax.ConstantStepSize,
        stepsize_controller_cfg={},
        cond=jnp.array([0.4]),
        guidance_scale=2.0,
    )
    assert out.shape == (1, 8, 8)
    assert jnp.all(jnp.isfinite(out))
