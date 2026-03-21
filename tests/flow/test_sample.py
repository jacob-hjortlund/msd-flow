"""Tests for src.flow.sample."""

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import pytest
from src.model.unet import UNet
from src.flow.sample import make_ode_term, sample

KEY = jax.random.PRNGKey(0)

SMALL_MODEL = UNet(
    in_channels=1, out_channels=1, base_channels=4, image_size=8,
    channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
    num_groups=2, activation=jax.nn.silu, key=KEY,
)


def test_make_ode_term_returns_ode_term():
    """Verify make_ode_term returns a diffrax ODETerm instance."""
    term = make_ode_term(SMALL_MODEL)
    assert isinstance(term, diffrax.ODETerm)


def test_sample_output_shape():
    """Verify sample output shape matches the requested shape."""
    out = sample(
        model=SMALL_MODEL,
        shape=(1, 8, 8),
        key=KEY,
        solver=diffrax.Euler(),
        dt0=0.1,
        t0=0.0,
        t1=1.0,
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    assert out.shape == (1, 8, 8)


def test_sample_output_finite():
    """Verify sample output contains only finite values."""
    out = sample(
        model=SMALL_MODEL,
        shape=(1, 8, 8),
        key=KEY,
        solver=diffrax.Euler(),
        dt0=0.1,
        t0=0.0,
        t1=1.0,
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    assert jnp.all(jnp.isfinite(out))


def test_sample_batched_via_vmap():
    """Verify batched sampling via vmap produces correct batch shape."""
    keys = jax.random.split(KEY, 3)
    batched_sample = jax.vmap(
        lambda k: sample(
            SMALL_MODEL, (1, 8, 8), k,
            diffrax.Euler(), 0.1, 0.0, 1.0, diffrax.ConstantStepSize(),
        )
    )
    outs = batched_sample(keys)
    assert outs.shape == (3, 1, 8, 8)
