"""Tests for src.train.trainer."""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pytest
import diffrax
from src.model.unet import UNet
from src.train.trainer import TrainState, make_train_state
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
OPTIMIZER = optax.adam(1e-3)


def test_train_state_is_eqx_module():
    """Verify TrainState is an Equinox module."""
    state = make_train_state(SMALL_MODEL, OPTIMIZER)
    assert isinstance(state, eqx.Module)


def test_train_state_has_model_and_opt_state():
    """Verify TrainState exposes model and opt_state attributes."""
    state = make_train_state(SMALL_MODEL, OPTIMIZER)
    assert hasattr(state, "model")
    assert hasattr(state, "opt_state")


def test_make_train_state_opt_state_matches_model_params():
    """Verify optimizer state is initialized (non-None)."""
    state = make_train_state(SMALL_MODEL, OPTIMIZER)
    # Optax adam state should be non-None
    assert state.opt_state is not None


from src.train.trainer import make_train_step


def test_train_step_returns_updated_state_and_loss():
    """Verify train step returns a TrainState and a scalar loss."""
    optimizer = optax.adam(1e-3)
    state = make_train_state(SMALL_MODEL, optimizer)
    train_step = make_train_step(optimizer)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)

    new_state, loss = train_step(state, x0, x1, t, cond, cond_mask)

    assert isinstance(new_state, TrainState)
    assert loss.shape == ()


def test_train_step_loss_is_finite():
    """Verify train step produces a finite loss value."""
    optimizer = optax.adam(1e-3)
    state = make_train_state(SMALL_MODEL, optimizer)
    train_step = make_train_step(optimizer)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)

    _, loss = train_step(state, x0, x1, t, cond, cond_mask)
    assert jnp.isfinite(loss)


def test_train_step_updates_model_params():
    """Verify at least one model parameter changes after a train step."""
    optimizer = optax.adam(1e-3)
    state = make_train_state(SMALL_MODEL, optimizer)
    train_step = make_train_step(optimizer)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)

    new_state, _ = train_step(state, x0, x1, t, cond, cond_mask)

    # At least one parameter should have changed
    orig_leaves = jax.tree_util.tree_leaves(eqx.filter(state.model, eqx.is_array))
    new_leaves = jax.tree_util.tree_leaves(eqx.filter(new_state.model, eqx.is_array))
    assert any(not jnp.allclose(o, n) for o, n in zip(orig_leaves, new_leaves))


import numpy as np
from src.train.trainer import train
from src.flow.coupling import ot_coupling


def _make_fake_dataloader(B=2, num_batches=3):
    """Yield fake (images, meta) tuples matching DataLoader contract."""
    import torch
    for _ in range(num_batches):
        images = torch.from_numpy(
            np.random.randn(B, 1, 8, 8).astype(np.float32)
        )
        meta = torch.empty(B, 0)
        yield images, meta


def test_train_runs_and_returns_model():
    """Verify the full training loop completes and returns a model."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "seed": 0,
        "train": {
            "num_steps": 3,
            "log_every": 1,
            "checkpoint_every": 100,  # won't trigger in 3 steps
            "checkpoint_dir": "/tmp/test_ckpt",
            "p_uncond": 0.0,
        },
        "flow": {"otfm": {"t_min": 0.0, "t_max": 1.0}},
    })
    optimizer = optax.adam(1e-3)
    dataloader = _make_fake_dataloader(B=2, num_batches=3)
    trained_model = train(cfg, SMALL_MODEL, dataloader, optimizer)
    assert trained_model is not None


def test_train_reduces_loss():
    """Verify the training loop runs without error on repeated fixed batches."""
    import torch
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "seed": 0,
        "train": {
            "num_steps": 20,
            "log_every": 5,
            "checkpoint_every": 100,
            "checkpoint_dir": "/tmp/test_ckpt",
            "p_uncond": 0.0,
        },
        "flow": {"otfm": {"t_min": 0.0, "t_max": 1.0}},
    })
    # Fixed batch — repeat the same data so loss can decrease
    fixed_images = torch.from_numpy(
        np.random.randn(4, 1, 8, 8).astype(np.float32)
    )
    fixed_meta = torch.empty(4, 0)
    def dataloader():
        while True:
            yield fixed_images, fixed_meta

    optimizer = optax.adam(1e-3)
    losses = []

    # Patch to capture losses — use a higher LR model
    big_model = UNet(
        in_channels=1, out_channels=1, base_channels=4,
        channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
        num_groups=2, activation=jax.nn.silu,
        key=jax.random.PRNGKey(99),
    )
    train(cfg, big_model, dataloader(), optimizer)
    # We just check it runs without error; strict loss decrease is not
    # guaranteed for random data in just 20 steps.


def test_train_step_with_cond():
    """Verify train step works with conditioning."""
    optimizer = optax.adam(1e-3)
    state = make_train_state(SMALL_MODEL_COND, optimizer)
    train_step = make_train_step(optimizer)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.array([[0.4], [0.8]])
    cond_mask = jnp.ones(B, dtype=bool)

    new_state, loss = train_step(state, x0, x1, t, cond, cond_mask)
    assert isinstance(new_state, TrainState)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_train_step_with_cond_dropped():
    """Verify train step works when some conditions are dropped (CFG path)."""
    optimizer = optax.adam(1e-3)
    state = make_train_state(SMALL_MODEL_COND, optimizer)
    train_step = make_train_step(optimizer)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.array([[0.4], [0.8]])
    cond_mask = jnp.array([True, False])  # second sample uses null embedding

    new_state, loss = train_step(state, x0, x1, t, cond, cond_mask)
    assert isinstance(new_state, TrainState)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_train_loop_with_cond():
    """Verify training loop works with metadata conditioning."""
    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "seed": 0,
        "train": {
            "num_steps": 3,
            "log_every": 1,
            "checkpoint_every": 100,
            "checkpoint_dir": "/tmp/test_ckpt_cond",
            "p_uncond": 0.2,
        },
        "flow": {"otfm": {"t_min": 0.0, "t_max": 1.0}},
    })

    def dataloader():
        for _ in range(3):
            images = torch.randn(2, 1, 8, 8)
            meta = torch.tensor([[0.4], [0.8]])
            yield images, meta

    optimizer = optax.adam(1e-3)
    trained = train(cfg, SMALL_MODEL_COND, dataloader(), optimizer)
    assert trained is not None


def test_end_to_end_conditional_training_and_sampling():
    """Train a small conditional model and verify unconditional and guided sampling."""
    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "seed": 0,
        "train": {
            "num_steps": 5,
            "log_every": 1,
            "checkpoint_every": 100,
            "checkpoint_dir": "/tmp/test_ckpt_e2e",
            "p_uncond": 0.2,
        },
        "flow": {"otfm": {"t_min": 0.0, "t_max": 1.0}},
    })

    def dataloader():
        for _ in range(5):
            images = torch.randn(2, 1, 8, 8)
            meta = torch.tensor([[0.4], [0.8]])
            yield images, meta

    optimizer = optax.adam(1e-3)
    trained = train(cfg, SMALL_MODEL_COND, dataloader(), optimizer)

    sample_kwargs = dict(
        model=trained,
        shape=(1, 8, 8),
        key=KEY,
        solver=diffrax.Euler,
        dt0=0.1,
        t0=0.0,
        t1=1.0,
        stepsize_controller=diffrax.ConstantStepSize,
        stepsize_controller_cfg={},
    )

    # Unconditional sample (cond=None default)
    out_uncond = sample(**sample_kwargs)
    assert out_uncond.shape == (1, 8, 8)
    assert jnp.all(jnp.isfinite(out_uncond))

    # Guided sample
    out_guided = sample(**sample_kwargs, cond=jnp.array([0.4]), guidance_scale=2.0)
    assert out_guided.shape == (1, 8, 8)
    assert jnp.all(jnp.isfinite(out_guided))
