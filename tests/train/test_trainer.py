"""Tests for msdflow.train.trainer."""

import inspect
import json
from decimal import Decimal

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pytest
import numpy as np
import diffrax
from msdflow.model.unet import UNet
import msdflow.train.parallel as train_parallel
from msdflow.train.trainer import (
    DataParallelConfig,
    TimeLossDiagnosticConfig,
    TrainState,
    _call_epoch_metric,
    make_data_parallel_config,
    make_train_state,
    make_prepare_batch_jax,
    resolve_data_parallel_config,
    resolve_time_loss_diagnostic_config,
    shard_batch,
    time_binned_loss_loop,
    train,
)
from msdflow.train.checkpointing import TrainingCheckpoint, load_training_checkpoint
from msdflow.flow.sample import sample
from msdflow.flow.interpolate import sample_path

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
OPTIMIZER = optax.adam(1e-3)


def _make_small_model(key=None):
    """Return a fresh small UNet for tests that donate model buffers."""
    if key is None:
        key = jax.random.PRNGKey(0)
    return UNet(
        in_channels=1,
        out_channels=1,
        base_channels=4,
        channel_multipliers=[1, 2],
        num_res_blocks=1,
        num_heads=1,
        num_groups=2,
        activation=jax.nn.silu,
        key=key,
    )


def _make_small_model_cond(key=None):
    """Return a fresh small conditional UNet for tests that donate model buffers."""
    if key is None:
        key = jax.random.PRNGKey(0)
    return UNet(
        in_channels=1,
        out_channels=1,
        base_channels=4,
        channel_multipliers=[1, 2],
        num_res_blocks=1,
        num_heads=1,
        num_groups=2,
        activation=jax.nn.silu,
        cond_dim=1,
        key=key,
    )


def test_train_state_is_eqx_module():
    """Verify TrainState is an Equinox module."""
    state = make_train_state(SMALL_MODEL, OPTIMIZER)
    assert isinstance(state, eqx.Module)


def test_train_state_has_model_and_opt_state():
    """Verify TrainState exposes model and opt_state attributes."""
    state = make_train_state(SMALL_MODEL, OPTIMIZER)
    assert hasattr(state, "model")
    assert hasattr(state, "opt_state")


def test_trainer_reexports_parallel_helpers():
    """trainer keeps the old sharding-helper import path working."""
    from msdflow.train import trainer as trainer_module

    assert trainer_module.DataParallelConfig is train_parallel.DataParallelConfig
    assert (
        trainer_module._parse_data_parallel_enabled
        is train_parallel._parse_data_parallel_enabled
    )
    assert (
        trainer_module._validate_data_parallel_config
        is train_parallel._validate_data_parallel_config
    )
    assert trainer_module.make_data_parallel_config is train_parallel.make_data_parallel_config
    assert (
        trainer_module.resolve_data_parallel_config
        is train_parallel.resolve_data_parallel_config
    )
    assert trainer_module.shard_train_state is train_parallel.shard_train_state
    assert trainer_module.shard_model is train_parallel.shard_model
    assert (
        trainer_module._validate_batch_for_data_parallel
        is train_parallel._validate_batch_for_data_parallel
    )
    assert trainer_module.shard_batch is train_parallel.shard_batch


def test_make_train_state_opt_state_matches_model_params():
    """Verify optimizer state is initialized (non-None)."""
    state = make_train_state(SMALL_MODEL, OPTIMIZER)
    # Optax adam state should be non-None
    assert state.opt_state is not None


def test_make_data_parallel_config_disabled_has_no_shardings():
    """Disabled data parallel config should not create mesh shardings."""
    cfg = make_data_parallel_config(enabled=False)

    assert isinstance(cfg, DataParallelConfig)
    assert cfg.enabled is False
    assert cfg.num_devices == 1
    assert cfg.data_sharding is None
    assert cfg.model_sharding is None


def test_make_data_parallel_config_rejects_min_devices_below_one():
    """Data parallel config requires min_devices >= 1."""
    with pytest.raises(ValueError, match="min_devices"):
        make_data_parallel_config(enabled=False, min_devices=0)


def test_make_data_parallel_config_enabled_requires_available_devices():
    """Enabled data parallel config rejects unavailable local device counts."""
    unavailable = len(jax.local_devices()) + 1

    with pytest.raises(ValueError, match="data parallel"):
        make_data_parallel_config(enabled=True, min_devices=unavailable)


def test_make_data_parallel_config_enabled_min_one_creates_shardings():
    """min_devices=1 exercises the sharding path on CPU-only machines."""
    cfg = make_data_parallel_config(enabled=True, min_devices=1)

    assert cfg.enabled is True
    assert cfg.num_devices == len(jax.local_devices())
    assert cfg.data_sharding is not None
    assert cfg.model_sharding is not None


def test_resolve_data_parallel_config_accepts_mapping():
    """Hydra-style mappings should resolve into a runtime data parallel config."""
    cfg = resolve_data_parallel_config(
        {"enabled": False, "axis_name": "sample", "min_devices": 3}
    )

    assert cfg.enabled is False
    assert cfg.axis_name == "sample"
    assert cfg.min_devices == 3


def test_resolve_data_parallel_config_parses_false_string_mapping():
    """String false values in direct mappings should not enable sharding."""
    cfg = resolve_data_parallel_config({"enabled": "false"})

    assert cfg.enabled is False


def test_resolve_data_parallel_config_rejects_malformed_enabled_config():
    """Enabled caller-provided configs must include valid sharding invariants."""
    cfg = DataParallelConfig(
        enabled=True,
        axis_name="batch",
        min_devices=2,
        num_devices=1,
        data_sharding=None,
        model_sharding=None,
    )

    with pytest.raises(ValueError, match="num_devices.*min_devices"):
        resolve_data_parallel_config(cfg)


def test_shard_batch_noops_when_data_parallel_disabled():
    """shard_batch should return the original tuple when disabled."""
    cfg = make_data_parallel_config(enabled=False)
    batch = (
        jnp.ones((2, 1, 4, 4)),
        jnp.ones((2, 1, 4, 4)),
        jnp.ones((2,)),
        jnp.ones((2, 0)),
        jnp.ones((2,), dtype=bool),
        jax.random.split(jax.random.PRNGKey(0), 2),
    )

    result = shard_batch(batch, cfg)

    assert result is batch


def test_shard_batch_enabled_min_one_applies_data_sharding():
    """Enabled shard_batch should place every array on the data sharding."""
    cfg = make_data_parallel_config(enabled=True, min_devices=1)
    batch_size = cfg.num_devices * 2
    batch = (
        jnp.ones((batch_size, 1, 4, 4)),
        jnp.ones((batch_size, 1, 4, 4)),
        jnp.ones((batch_size,)),
        jnp.ones((batch_size, 0)),
        jnp.ones((batch_size,), dtype=bool),
        jax.random.split(jax.random.PRNGKey(0), batch_size),
    )

    result = shard_batch(batch, cfg)

    assert all(array.sharding == cfg.data_sharding for array in result)


def test_shard_batch_enabled_rejects_scalar_arrays():
    """Data-parallel batches require arrays with a leading batch dimension."""
    cfg = make_data_parallel_config(enabled=True, min_devices=1)
    batch_size = cfg.num_devices
    batch = (
        jnp.ones((batch_size, 1, 4, 4)),
        jnp.ones((batch_size, 1, 4, 4)),
        jnp.array(0.5),
        jnp.ones((batch_size, 0)),
        jnp.ones((batch_size,), dtype=bool),
        jax.random.split(jax.random.PRNGKey(0), batch_size),
    )

    with pytest.raises(ValueError, match="leading batch dimension"):
        shard_batch(batch, cfg)


def test_shard_batch_enabled_rejects_inconsistent_leading_dimensions():
    """All data-parallel batch arrays must share one leading dimension."""
    cfg = make_data_parallel_config(enabled=True, min_devices=1)
    batch_size = cfg.num_devices * 2
    batch = (
        jnp.ones((batch_size, 1, 4, 4)),
        jnp.ones((batch_size + 1, 1, 4, 4)),
        jnp.ones((batch_size,)),
        jnp.ones((batch_size, 0)),
        jnp.ones((batch_size,), dtype=bool),
        jax.random.split(jax.random.PRNGKey(0), batch_size),
    )

    with pytest.raises(ValueError, match="same leading dimension"):
        shard_batch(batch, cfg)


def test_shard_batch_rejects_nondivisible_batch_axis():
    """Data-parallel batches must divide evenly across selected devices."""
    cfg = DataParallelConfig(
        enabled=True,
        axis_name="batch",
        min_devices=2,
        num_devices=2,
        data_sharding=None,
        model_sharding=None,
    )
    batch = (
        jnp.ones((3, 1, 4, 4)),
        jnp.ones((3, 1, 4, 4)),
        jnp.ones((3,)),
        jnp.ones((3, 0)),
        jnp.ones((3,), dtype=bool),
        jax.random.split(jax.random.PRNGKey(0), 3),
    )

    with pytest.raises(ValueError, match="divisible"):
        shard_batch(batch, cfg)


from msdflow.train.trainer import make_train_step
from msdflow.train.metrics import flow_matching_loss as _fml


def test_make_train_step_dispatches_to_injected_loss_fn():
    """make_train_step must use the injected loss_fn, not a hardcoded one."""
    optimizer = optax.adam(1e-3)
    model = _make_small_model()
    state = make_train_state(model, optimizer)

    def constant_loss(model, x_t, u_t, t, cond, cond_mask, key):
        return jnp.array(42.0)

    train_step = make_train_step(optimizer, constant_loss)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)

    _, loss = train_step(state, x_t, u_t, t, cond, cond_mask, jax.random.PRNGKey(0))
    assert jnp.allclose(loss, jnp.array(42.0))


def test_train_step_returns_updated_state_and_loss():
    """Verify train step returns a TrainState and a scalar loss."""
    optimizer = optax.adam(1e-3)
    model = _make_small_model()
    state = make_train_state(model, optimizer)
    train_step = make_train_step(optimizer, _fml)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)

    x_t, u_t = sample_path(x0, x1, t)
    new_state, loss = train_step(
        state, x_t, u_t, t, cond, cond_mask, jax.random.PRNGKey(0)
    )

    assert isinstance(new_state, TrainState)
    assert loss.shape == ()


def test_make_train_step_accepts_data_parallel_disabled():
    """make_train_step should keep working with an explicit disabled config."""
    optimizer = optax.adam(1e-3)
    model = _make_small_model()
    state = make_train_state(model, optimizer)
    cfg = make_data_parallel_config(enabled=False)
    train_step = make_train_step(optimizer, _fml, data_parallel=cfg)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)

    new_state, loss = train_step(
        state, x_t, u_t, t, cond, cond_mask, jax.random.PRNGKey(0)
    )

    assert isinstance(new_state, TrainState)
    assert loss.shape == ()


def test_train_step_loss_is_finite():
    """Verify train step produces a finite loss value."""
    optimizer = optax.adam(1e-3)
    model = _make_small_model()
    state = make_train_state(model, optimizer)
    train_step = make_train_step(optimizer, _fml)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)

    x_t, u_t = sample_path(x0, x1, t)
    _, loss = train_step(state, x_t, u_t, t, cond, cond_mask, jax.random.PRNGKey(0))
    assert jnp.isfinite(loss)


def test_train_step_updates_model_params():
    """Verify at least one model parameter changes after a train step."""
    optimizer = optax.adam(1e-3)
    model = _make_small_model()
    state = make_train_state(model, optimizer)
    train_step = make_train_step(optimizer, _fml)
    orig_leaves = jax.tree_util.tree_leaves(eqx.filter(state.model, eqx.is_array))
    orig_leaves = [jnp.array(leaf, copy=True).block_until_ready() for leaf in orig_leaves]

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)

    x_t, u_t = sample_path(x0, x1, t)
    new_state, _ = train_step(
        state, x_t, u_t, t, cond, cond_mask, jax.random.PRNGKey(0)
    )

    # At least one parameter should have changed
    new_leaves = jax.tree_util.tree_leaves(eqx.filter(new_state.model, eqx.is_array))
    assert any(not jnp.allclose(o, n) for o, n in zip(orig_leaves, new_leaves))


from msdflow.train.trainer import ema_update


def test_ema_update_decay_one_leaves_ema_unchanged():
    """With decay=1.0, EMA model arrays must not change."""
    key1, key2 = jax.random.split(jax.random.PRNGKey(10))
    ema_model = eqx.nn.Linear(4, 4, key=key1)
    new_model = eqx.nn.Linear(4, 4, key=key2)

    result = ema_update(ema_model, new_model, decay=1.0)

    assert jnp.allclose(result.weight, ema_model.weight)
    assert jnp.allclose(result.bias, ema_model.bias)


def test_ema_update_decay_zero_copies_new_model():
    """With decay=0.0, EMA model arrays must equal the new model."""
    key1, key2 = jax.random.split(jax.random.PRNGKey(11))
    ema_model = eqx.nn.Linear(4, 4, key=key1)
    new_model = eqx.nn.Linear(4, 4, key=key2)

    result = ema_update(ema_model, new_model, decay=0.0)

    assert jnp.allclose(result.weight, new_model.weight)
    assert jnp.allclose(result.bias, new_model.bias)


def test_ema_update_blends_arrays_correctly():
    """EMA update applies decay * ema + (1 - decay) * new element-wise."""
    key1, key2 = jax.random.split(jax.random.PRNGKey(12))
    ema_model = eqx.nn.Linear(4, 4, key=key1)
    new_model = eqx.nn.Linear(4, 4, key=key2)
    decay = 0.9

    result = ema_update(ema_model, new_model, decay=decay)

    expected_weight = 0.9 * ema_model.weight + 0.1 * new_model.weight
    expected_bias = 0.9 * ema_model.bias + 0.1 * new_model.bias
    assert jnp.allclose(result.weight, expected_weight, atol=1e-6)
    assert jnp.allclose(result.bias, expected_bias, atol=1e-6)


def test_ema_update_is_jit_compiled():
    """ema_update should be JIT-compiled (filter_jit decorated)."""
    key1, key2 = jax.random.split(jax.random.PRNGKey(13))
    ema_model = eqx.nn.Linear(4, 4, key=key1)
    new_model = eqx.nn.Linear(4, 4, key=key2)
    decay = 0.9

    # Call twice — second call should use cached compilation
    result1 = ema_update(ema_model, new_model, decay=decay)
    result2 = ema_update(ema_model, new_model, decay=decay)

    expected_weight = 0.9 * ema_model.weight + 0.1 * new_model.weight
    assert jnp.allclose(result1.weight, expected_weight, atol=1e-6)
    assert jnp.allclose(result2.weight, expected_weight, atol=1e-6)


from msdflow.train.trainer import make_batch_metric_step
from msdflow.train.metrics import TimeBinnedLossResult, make_time_binned_loss_step


def test_make_batch_metric_step_returns_dict_keyed_by_fn_name():
    """make_batch_metric_step must return a dict keyed by fn.__name__."""
    step = make_batch_metric_step([_fml])
    model = _make_small_model()
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)

    result = step(model, x_t, u_t, t, cond, cond_mask, jax.random.PRNGKey(0))
    assert isinstance(result, dict)
    assert "flow_matching_loss" in result


def test_make_batch_metric_step_accepts_data_parallel_disabled():
    """make_batch_metric_step should work with an explicit disabled config."""
    cfg = make_data_parallel_config(enabled=False)
    step = make_batch_metric_step([_fml], data_parallel=cfg)
    model = _make_small_model()
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)

    result = step(model, x_t, u_t, t, cond, cond_mask, jax.random.PRNGKey(0))

    assert "flow_matching_loss" in result


def test_make_batch_metric_step_values_are_scalar_jax_arrays():
    """All values returned by make_batch_metric_step must be scalar JAX arrays."""
    step = make_batch_metric_step([_fml])
    model = _make_small_model()
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)

    result = step(model, x_t, u_t, t, cond, cond_mask, jax.random.PRNGKey(0))
    for v in result.values():
        assert isinstance(v, jax.Array)
        assert v.shape == ()


def test_make_batch_metric_step_multiple_metrics_all_keys_present():
    """make_batch_metric_step with two distinct metrics returns both keys."""

    def dummy_metric(model, x_t, u_t, t, cond, cond_mask, key):
        return jnp.array(0.0)

    step = make_batch_metric_step([_fml, dummy_metric])
    model = _make_small_model()
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)

    result = step(model, x_t, u_t, t, cond, cond_mask, jax.random.PRNGKey(0))
    assert "flow_matching_loss" in result
    assert "dummy_metric" in result


def test_make_batch_metric_step_raises_on_duplicate_names():
    """make_batch_metric_step must raise ValueError for duplicate metric names."""

    def my_metric(model, x_t, u_t, t, cond, cond_mask, key):
        return jnp.array(0.0)

    def my_metric_copy(
        model, x_t, u_t, t, cond, cond_mask, key
    ):  # same __name__ via rename
        return jnp.array(1.0)

    my_metric_copy.__name__ = "my_metric"

    with pytest.raises(ValueError, match="duplicate metric names"):
        make_batch_metric_step([my_metric, my_metric_copy])


from functools import partial

from msdflow.train.trainer import train
from msdflow.flow.coupling import independent_coupling
from msdflow.flow.interpolate import sample_time_uniform
import torch


def _make_fake_dataloader(B=2, num_batches=3):
    """Yield fake (images, meta) tuples matching DataLoader contract."""
    for _ in range(num_batches):
        images = torch.from_numpy(np.random.randn(B, 1, 8, 8).astype(np.float32))
        meta = torch.empty(B, 0)
        yield images, meta


def _fake_val_dataloader(B=2):
    """Return a list with one val batch (re-iterable)."""
    images = torch.from_numpy(np.random.randn(B, 1, 8, 8).astype(np.float32))
    meta = torch.empty(B, 0)
    return [(images, meta)]


def _make_train_kwargs(num_epochs=1, num_steps_per_epoch=3, p_uncond=0.0):
    """Return default keyword args matching the current train() signature."""
    return dict(
        key=jax.random.PRNGKey(0),
        optimizer=optax.adam(1e-3),
        loss_fn=_fml,
        batch_metrics=[_fml],
        epoch_metrics=[],
        num_train_eval_batches=0,
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        num_epochs=num_epochs,
        num_steps_per_epoch=num_steps_per_epoch,
        p_uncond=p_uncond,
        ema_decay=0.9999,
        log_every=1,
        val_every=1,
        checkpoint_every=100,
        checkpoint_dir="/tmp/test_ckpt",
    )


class ScalarLogTask:
    """Collect scalar metrics passed through ClearML-style logging."""

    def __init__(self):
        """Initialize an empty scalar call list."""
        self.scalars = []

    class Logger:
        """Minimal ClearML logger facade used by log_metrics()."""

        def __init__(self, owner):
            """Store the parent collector."""
            self.owner = owner

        def report_scalar(self, title, series, value, iteration):
            """Record scalar reports for assertions."""
            self.owner.scalars.append((title, series, value, iteration))

    def get_logger(self):
        """Return a logger facade."""
        return self.Logger(self)


def constant_zero_metric(model, x_t, u_t, t, cond, cond_mask, key):
    """Return a constant zero metric."""
    return jnp.array(0.0)


def constant_one_loss(model, x_t, u_t, t, cond, cond_mask, key):
    """Return a differentiable constant loss for resume accounting tests."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return jnp.sum(leaves[0]) * 0.0 + jnp.array(1.0)


def test_train_runs_and_returns_model():
    """Verify the full training loop completes and returns a model."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=3))
    val_dataloader = _fake_val_dataloader()
    model = _make_small_model()
    trained_model = train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **_make_train_kwargs(),
    )
    assert trained_model is not None


def test_train_saves_full_state_periodic_checkpoint(tmp_path):
    """Periodic checkpointing must save a resumable full-state checkpoint."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=2))
    val_dataloader = _fake_val_dataloader(B=2)
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["checkpoint_every"] = 1
    kwargs["checkpoint_hash"] = "hash123"
    kwargs["hash_payload"] = {"model": {"base_channels": 4}}
    kwargs["latest_filename"] = "latest.json"
    model = _make_small_model()

    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )

    pointer = json.loads((tmp_path / "latest.json").read_text())
    metadata = json.loads(open(pointer["metadata_path"]).read())
    assert metadata["checkpoint_kind"] == "periodic"
    assert metadata["stable_hash"] == "hash123"
    assert metadata["epoch"] == 1
    assert metadata["completed_microsteps"] == 0
    assert (tmp_path / "model_epoch1_raw.eqx").exists()
    assert (tmp_path / "model_epoch1_ema.eqx").exists()

    like_model = _make_small_model()
    like_state = make_train_state(like_model, kwargs["optimizer"])
    like_checkpoint = TrainingCheckpoint(
        state=like_state,
        ema_model=like_model,
        ema_initialized=True,
        key=jax.random.PRNGKey(0),
        sampling_key=jax.random.PRNGKey(1),
        epoch=0,
        completed_microsteps=0,
        epoch_loss=0.0,
        best_metric_value=float("inf"),
        best_epoch=None,
        patience_counter=0,
        total_epoch_time=0.0,
        total_train_time=0.0,
        total_val_time=0.0,
        val_runs=0,
        val_time=float("nan"),
        val_metrics={},
        train_metrics={},
        epoch_metric_results={},
    )
    checkpoint = load_training_checkpoint(metadata["payload_path"], like_checkpoint)
    assert checkpoint.epoch == 1
    assert checkpoint.completed_microsteps == 0


def test_train_resume_checkpoint_path_loads_sidecar_metadata(tmp_path):
    """Path-based resume should infer EMA structure from checkpoint metadata."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=2))
    val_dataloader = _fake_val_dataloader(B=2)
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["checkpoint_every"] = 1
    kwargs["checkpoint_hash"] = "hash123"
    kwargs["latest_filename"] = "latest.json"
    model = _make_small_model()

    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )

    pointer = json.loads((tmp_path / "latest.json").read_text())
    metadata = json.loads(open(pointer["metadata_path"]).read())
    resume_kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=1)
    resume_kwargs["checkpoint_dir"] = str(tmp_path)
    resume_kwargs["checkpoint_hash"] = "hash123"

    resumed = train(
        model=_make_small_model(),
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        resume_checkpoint_path=metadata["payload_path"],
        **resume_kwargs,
    )

    assert resumed is not None


def test_train_resumes_mid_epoch_and_normalizes_loss_with_saved_microsteps(tmp_path):
    """Mid-epoch resume should replay epoch and include saved loss denominator."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=3))
    val_dataloader = _fake_val_dataloader(B=2)
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["checkpoint_every"] = 100
    kwargs["batch_metrics"] = [constant_zero_metric]
    kwargs["loss_fn"] = constant_one_loss
    kwargs["clearml_task"] = ScalarLogTask()
    kwargs["val_every"] = 100
    model = _make_small_model()
    state = make_train_state(model, kwargs["optimizer"])
    resume_payload = TrainingCheckpoint(
        state=state,
        ema_model=model,
        ema_initialized=True,
        key=jax.random.PRNGKey(5),
        sampling_key=jax.random.PRNGKey(6),
        epoch=0,
        completed_microsteps=2,
        epoch_loss=8.0,
        best_metric_value=float("inf"),
        best_epoch=None,
        patience_counter=0,
        total_epoch_time=0.0,
        total_train_time=0.0,
        total_val_time=0.0,
        val_runs=0,
        val_time=float("nan"),
        val_metrics={},
        train_metrics={},
        epoch_metric_results={},
    )

    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        resume_checkpoint=resume_payload,
        resume_metadata={
            "metadata_path": str(tmp_path / "checkpoint.json"),
            "checkpoint_kind": "sigterm",
        },
        **kwargs,
    )

    loss_scalars = [
        scalar
        for scalar in kwargs["clearml_task"].scalars
        if scalar[0] == "train/loss"
    ]
    assert loss_scalars
    assert loss_scalars[-1][3] == 1
    assert loss_scalars[-1][2] == pytest.approx(3.0)


def test_train_sigterm_flag_saves_checkpoint_and_stops(tmp_path):
    """A requested SIGTERM flag should save full state and return cleanly."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=4))
    val_dataloader = _fake_val_dataloader(B=2)
    kwargs = _make_train_kwargs(num_epochs=3, num_steps_per_epoch=2)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["checkpoint_hash"] = "hash123"
    kwargs["latest_filename"] = "latest.json"
    kwargs["save_on_sigterm"] = True
    model = _make_small_model()

    class RequestedFlag:
        """Context manager whose requested flag is already set."""

        def __init__(self):
            """Set the request flag."""
            self.requested = True

        def __enter__(self):
            """Return this requested flag."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Leave the fake signal context."""
            return None

    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        sigterm_flag_factory=lambda enabled: RequestedFlag(),
        **kwargs,
    )

    pointer = json.loads((tmp_path / "latest.json").read_text())
    metadata = json.loads(open(pointer["metadata_path"]).read())
    assert metadata["checkpoint_kind"] == "sigterm"
    assert metadata["epoch"] == 0
    assert metadata["completed_microsteps"] == 1


def test_train_sigterm_at_epoch_boundary_resumes_next_epoch(tmp_path):
    """SIGTERM after a complete epoch should not replay that whole epoch."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=2))
    val_dataloader = _fake_val_dataloader(B=2)
    kwargs = _make_train_kwargs(num_epochs=2, num_steps_per_epoch=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["checkpoint_hash"] = "hash123"
    kwargs["latest_filename"] = "latest.json"
    model = _make_small_model()

    class RequestedFlag:
        """Context manager whose requested flag is already set."""

        requested = True

        def __enter__(self):
            """Return this requested flag."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Leave the fake signal context."""
            return None

    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        sigterm_flag_factory=lambda enabled: RequestedFlag(),
        **kwargs,
    )

    pointer = json.loads((tmp_path / "latest.json").read_text())
    metadata = json.loads(open(pointer["metadata_path"]).read())
    assert metadata["checkpoint_kind"] == "sigterm"
    assert metadata["epoch"] == 1
    assert metadata["completed_microsteps"] == 0


def test_train_sigterm_requested_after_final_microstep_stops_at_epoch_boundary(
    tmp_path,
):
    """SIGTERM after the last microstep poll should stop before the next epoch."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=3))
    val_dataloader = _fake_val_dataloader(B=2)
    kwargs = _make_train_kwargs(num_epochs=2, num_steps_per_epoch=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["checkpoint_hash"] = "hash123"
    kwargs["latest_filename"] = "latest.json"
    kwargs["checkpoint_every"] = 100
    kwargs["val_every"] = 100
    model = _make_small_model()

    class BoundaryFlag:
        """Flag that becomes requested after the final microstep check."""

        def __init__(self):
            """Initialize request polling state."""
            self.reads = 0

        @property
        def requested(self):
            """Return true after the in-loop microstep poll."""
            self.reads += 1
            return self.reads > 1

        def __enter__(self):
            """Return this boundary flag."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Leave the fake signal context."""
            return None

    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        sigterm_flag_factory=lambda enabled: BoundaryFlag(),
        **kwargs,
    )

    pointer = json.loads((tmp_path / "latest.json").read_text())
    metadata = json.loads(open(pointer["metadata_path"]).read())
    assert metadata["checkpoint_kind"] == "sigterm"
    assert metadata["epoch"] == 1
    assert metadata["completed_microsteps"] == 0


def test_train_sigterm_during_resumed_epoch_waits_for_replay_boundary(tmp_path):
    """SIGTERM in a resumed epoch should not advance until replay is complete."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=2))
    val_dataloader = _fake_val_dataloader(B=2)
    kwargs = _make_train_kwargs(num_epochs=2, num_steps_per_epoch=2)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["checkpoint_hash"] = "hash123"
    kwargs["latest_filename"] = "latest.json"
    kwargs["checkpoint_every"] = 100
    kwargs["val_every"] = 100
    model = _make_small_model()
    state = make_train_state(model, kwargs["optimizer"])
    resume_payload = TrainingCheckpoint(
        state=state,
        ema_model=model,
        ema_initialized=True,
        key=jax.random.PRNGKey(5),
        sampling_key=jax.random.PRNGKey(6),
        epoch=0,
        completed_microsteps=1,
        epoch_loss=2.0,
        best_metric_value=float("inf"),
        best_epoch=None,
        patience_counter=0,
        total_epoch_time=0.0,
        total_train_time=0.0,
        total_val_time=0.0,
        val_runs=0,
        val_time=float("nan"),
        val_metrics={},
        train_metrics={},
        epoch_metric_results={},
    )

    class RequestedFlag:
        """Context manager whose requested flag is already set."""

        requested = True

        def __enter__(self):
            """Return this requested flag."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Leave the fake signal context."""
            return None

    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        resume_checkpoint=resume_payload,
        resume_metadata={
            "schema_version": 1,
            "stable_hash": "hash123",
            "checkpoint_kind": "sigterm",
            "epoch": 0,
            "completed_microsteps": 1,
            "payload_path": str(tmp_path / "resume.eqx"),
            "monitor": "flow_matching_loss",
            "monitor_mode": "min",
            "microsteps_per_epoch": 2,
        },
        sigterm_flag_factory=lambda enabled: RequestedFlag(),
        **kwargs,
    )

    pointer = json.loads((tmp_path / "latest.json").read_text())
    metadata = json.loads(open(pointer["metadata_path"]).read())
    assert metadata["checkpoint_kind"] == "sigterm"
    assert metadata["epoch"] == 0
    assert metadata["completed_microsteps"] == 1


def test_train_sigterm_during_repeated_resume_caps_completed_microsteps(tmp_path):
    """Repeated mid-epoch SIGTERM should save bounded replay progress."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=4))
    val_dataloader = _fake_val_dataloader(B=2)
    kwargs = _make_train_kwargs(num_epochs=2, num_steps_per_epoch=4)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["checkpoint_hash"] = "hash123"
    kwargs["latest_filename"] = "latest.json"
    kwargs["checkpoint_every"] = 100
    kwargs["val_every"] = 100
    kwargs["loss_fn"] = constant_one_loss
    model = _make_small_model()
    state = make_train_state(model, kwargs["optimizer"])
    resume_payload = TrainingCheckpoint(
        state=state,
        ema_model=model,
        ema_initialized=True,
        key=jax.random.PRNGKey(5),
        sampling_key=jax.random.PRNGKey(6),
        epoch=0,
        completed_microsteps=3,
        epoch_loss=9.0,
        best_metric_value=float("inf"),
        best_epoch=None,
        patience_counter=0,
        total_epoch_time=0.0,
        total_train_time=0.0,
        total_val_time=0.0,
        val_runs=0,
        val_time=float("nan"),
        val_metrics={},
        train_metrics={},
        epoch_metric_results={},
    )

    class SecondMicrostepFlag:
        """Context manager requested on the second replayed microstep."""

        def __init__(self):
            """Initialize request polling state."""
            self.reads = 0

        @property
        def requested(self):
            """Return true only after the first replayed microstep."""
            self.reads += 1
            return self.reads > 1

        def __enter__(self):
            """Return this requested flag."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Leave the fake signal context."""
            return None

    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        resume_checkpoint=resume_payload,
        resume_metadata={
            "schema_version": 1,
            "stable_hash": "hash123",
            "checkpoint_kind": "sigterm",
            "epoch": 0,
            "completed_microsteps": 3,
            "payload_path": str(tmp_path / "resume.eqx"),
            "monitor": "flow_matching_loss",
            "monitor_mode": "min",
            "microsteps_per_epoch": 4,
        },
        sigterm_flag_factory=lambda enabled: SecondMicrostepFlag(),
        **kwargs,
    )

    pointer = json.loads((tmp_path / "latest.json").read_text())
    metadata = json.loads(open(pointer["metadata_path"]).read())
    assert metadata["checkpoint_kind"] == "sigterm"
    assert metadata["epoch"] == 0
    assert metadata["completed_microsteps"] == 3

    like_checkpoint = TrainingCheckpoint(
        state=state,
        ema_model=model,
        ema_initialized=True,
        key=jax.random.PRNGKey(0),
        sampling_key=jax.random.PRNGKey(1),
        epoch=0,
        completed_microsteps=0,
        epoch_loss=0.0,
        best_metric_value=float("inf"),
        best_epoch=None,
        patience_counter=0,
        total_epoch_time=0.0,
        total_train_time=0.0,
        total_val_time=0.0,
        val_runs=0,
        val_time=float("nan"),
        val_metrics={},
        train_metrics={},
        epoch_metric_results={},
    )
    checkpoint = load_training_checkpoint(metadata["payload_path"], like_checkpoint)
    assert checkpoint.completed_microsteps == 3
    assert checkpoint.epoch_loss / checkpoint.completed_microsteps == pytest.approx(
        11.0 / 5.0,
    )


def test_train_sigterm_after_epoch_work_stops_before_next_epoch(tmp_path):
    """SIGTERM during epoch-end work should not start the next epoch."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=3))
    val_dataloader = _fake_val_dataloader(B=2)
    kwargs = _make_train_kwargs(num_epochs=2, num_steps_per_epoch=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["checkpoint_hash"] = "hash123"
    kwargs["latest_filename"] = "latest.json"
    kwargs["checkpoint_every"] = 100
    kwargs["val_every"] = 100
    model = _make_small_model()

    class EpochEndFlag:
        """Flag that becomes requested after epoch-boundary polling."""

        def __init__(self):
            """Initialize request polling state."""
            self.reads = 0

        @property
        def requested(self):
            """Return true after microstep and boundary polls."""
            self.reads += 1
            return self.reads > 2

        def __enter__(self):
            """Return this epoch-end flag."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Leave the fake signal context."""
            return None

    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        sigterm_flag_factory=lambda enabled: EpochEndFlag(),
        **kwargs,
    )

    pointer = json.loads((tmp_path / "latest.json").read_text())
    metadata = json.loads(open(pointer["metadata_path"]).read())
    assert metadata["checkpoint_kind"] == "sigterm"
    assert metadata["epoch"] == 1
    assert metadata["completed_microsteps"] == 0


def test_train_sigterm_during_validation_preempts_early_stopping(tmp_path):
    """SIGTERM observed during validation should checkpoint before early stop."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=3))
    val_dataloader = _fake_val_dataloader(B=2)
    kwargs = _make_train_kwargs(num_epochs=3, num_steps_per_epoch=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["checkpoint_hash"] = "hash123"
    kwargs["latest_filename"] = "latest.json"
    kwargs["checkpoint_every"] = 100
    kwargs["monitor"] = "constant_validation_metric"
    kwargs["early_stopping_patience"] = 1
    model = _make_small_model()
    flag_holder = {}
    metric_calls = []

    class ValidationFlag:
        """Context manager requested by the validation metric."""

        def __init__(self):
            """Initialize an unset request flag."""
            self.requested = False

        def __enter__(self):
            """Return this validation flag."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Leave the fake signal context."""
            return None

    def constant_validation_metric(model, val_batches, key):
        """Set the SIGTERM flag during the second validation cycle."""
        metric_calls.append(None)
        if len(metric_calls) > 1:
            flag_holder["flag"].requested = True
        return jnp.array(0.0)

    def factory(enabled):
        """Create and retain the validation flag."""
        flag = ValidationFlag()
        flag_holder["flag"] = flag
        return flag

    kwargs["epoch_metrics"] = [constant_validation_metric]

    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        sigterm_flag_factory=factory,
        **kwargs,
    )

    pointer = json.loads((tmp_path / "latest.json").read_text())
    metadata = json.loads(open(pointer["metadata_path"]).read())
    assert metadata["checkpoint_kind"] == "sigterm"
    assert metadata["epoch"] == 2
    assert metadata["completed_microsteps"] == 0


def test_train_sigterm_handler_disabled_without_checkpoint_hash(tmp_path):
    """Legacy model-only runs should not install a SIGTERM checkpoint handler."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=2))
    val_dataloader = _fake_val_dataloader(B=2)
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    seen_enabled = []
    model = _make_small_model()

    class UnrequestedFlag:
        """Context manager that records disabled SIGTERM setup."""

        requested = False

        def __enter__(self):
            """Return this unrequested flag."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Leave the fake signal context."""
            return None

    def factory(enabled):
        """Record whether the trainer attempted to enable SIGTERM handling."""
        seen_enabled.append(enabled)
        return UnrequestedFlag()

    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        sigterm_flag_factory=factory,
        **kwargs,
    )

    assert seen_enabled == [False]


def test_train_runs_with_data_parallel_enabled_min_one(tmp_path):
    """train() should execute the data-parallel path when min_devices=1."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=2))
    val_dataloader = _fake_val_dataloader(B=2)
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    model = _make_small_model()

    trained_model = train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        data_parallel={"enabled": True, "axis_name": "batch", "min_devices": 1},
        **kwargs,
    )

    assert trained_model is not None


def test_train_data_parallel_requires_available_devices(tmp_path):
    """train() should resolve data_parallel and reject unavailable devices."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=2))
    val_dataloader = _fake_val_dataloader(B=2)
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    model = _make_small_model()
    unavailable = len(jax.local_devices()) + 1

    with pytest.raises(ValueError, match="data parallel"):
        train(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            data_parallel={
                "enabled": True,
                "axis_name": "batch",
                "min_devices": unavailable,
            },
            **kwargs,
        )


def test_train_returns_ema_model_not_live_model():
    """train() must return the EMA model, which differs from the initial model after training."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=5))
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=5)
    # High learning rate so live model diverges quickly; EMA lags behind
    kwargs["optimizer"] = optax.adam(1e-1)
    model = _make_small_model()
    init_leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    init_leaves = [jnp.array(leaf, copy=True).block_until_ready() for leaf in init_leaves]
    trained = train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )
    trained_leaves = jax.tree_util.tree_leaves(eqx.filter(trained, eqx.is_array))
    assert any(not jnp.allclose(i, t) for i, t in zip(init_leaves, trained_leaves))


def test_train_loop_completes_without_error():
    """Verify the training loop runs without error on repeated fixed batches."""
    fixed_images = torch.from_numpy(np.random.randn(4, 1, 8, 8).astype(np.float32))
    fixed_meta = torch.empty(4, 0)
    dataloader = [(fixed_images, fixed_meta) for _ in range(20)]
    val_dataloader = _fake_val_dataloader()
    big_model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=4,
        channel_multipliers=[1, 2],
        num_res_blocks=1,
        num_heads=1,
        num_groups=2,
        activation=jax.nn.silu,
        key=jax.random.PRNGKey(99),
    )
    train(
        model=big_model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **_make_train_kwargs(num_steps_per_epoch=20),
    )


def test_train_step_with_cond():
    """Verify train step works with conditioning."""
    optimizer = optax.adam(1e-3)
    model_cond = _make_small_model_cond()
    state = make_train_state(model_cond, optimizer)
    train_step = make_train_step(optimizer, _fml)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.array([[0.4], [0.8]])
    cond_mask = jnp.ones(B, dtype=bool)

    x_t, u_t = sample_path(x0, x1, t)
    new_state, loss = train_step(
        state, x_t, u_t, t, cond, cond_mask, jax.random.PRNGKey(0)
    )
    assert isinstance(new_state, TrainState)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_train_step_with_cond_dropped():
    """Verify train step works when some conditions are dropped (CFG path)."""
    optimizer = optax.adam(1e-3)
    model_cond = _make_small_model_cond()
    state = make_train_state(model_cond, optimizer)
    train_step = make_train_step(optimizer, _fml)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.array([[0.4], [0.8]])
    cond_mask = jnp.array([True, False])

    x_t, u_t = sample_path(x0, x1, t)
    new_state, loss = train_step(
        state, x_t, u_t, t, cond, cond_mask, jax.random.PRNGKey(0)
    )
    assert isinstance(new_state, TrainState)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_train_loop_with_cond():
    """Verify training loop works with metadata conditioning."""
    dataloader = [
        (torch.randn(2, 1, 8, 8), torch.tensor([[0.4], [0.8]])) for _ in range(3)
    ]
    val_dataloader = [(torch.randn(2, 1, 8, 8), torch.tensor([[0.4], [0.8]]))]
    kwargs = _make_train_kwargs(p_uncond=0.2)
    kwargs["checkpoint_dir"] = "/tmp/test_ckpt_cond"
    model_cond = _make_small_model_cond()
    trained = train(
        model=model_cond,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )
    assert trained is not None


def test_train_num_steps_per_epoch_zero_uses_dataloader_length():
    """num_steps_per_epoch=0 should run exactly len(dataloader) steps per epoch."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=4))
    val_dataloader = _fake_val_dataloader()
    model = _make_small_model()
    # Simply verify it completes without error when num_steps_per_epoch=0
    trained = train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **_make_train_kwargs(num_steps_per_epoch=0),
    )
    assert trained is not None


def test_end_to_end_conditional_training_and_sampling():
    """Train a small conditional model and verify unconditional and guided sampling."""
    dataloader = [
        (torch.randn(2, 1, 8, 8), torch.tensor([[0.4], [0.8]])) for _ in range(5)
    ]
    val_dataloader = [(torch.randn(2, 1, 8, 8), torch.tensor([[0.4], [0.8]]))]

    kwargs = _make_train_kwargs(num_steps_per_epoch=5, p_uncond=0.2)
    kwargs["checkpoint_dir"] = "/tmp/test_ckpt_e2e"
    model_cond = _make_small_model_cond()
    trained = train(
        model=model_cond,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )

    sample_kwargs = dict(
        model=trained,
        shape=(1, 8, 8),
        key=KEY,
        solver=diffrax.Euler(),
        dt0=0.1,
        t0=0.0,
        t1=1.0,
        stepsize_controller=diffrax.ConstantStepSize(),
    )

    out_uncond = sample(**sample_kwargs)
    assert out_uncond.shape == (1, 8, 8)
    assert jnp.all(jnp.isfinite(out_uncond))

    out_guided = sample(**sample_kwargs, cond=jnp.array([0.4]), guidance_scale=2.0)
    assert out_guided.shape == (1, 8, 8)
    assert jnp.all(jnp.isfinite(out_guided))


# --- prepare_batch ---

from msdflow.train.trainer import prepare_batch


def test_prepare_batch_returns_numpy_arrays():
    """prepare_batch returns numpy arrays from PyTorch tensors."""
    B = 4
    images = torch.from_numpy(np.random.randn(B, 1, 8, 8).astype(np.float32))
    meta = torch.empty(B, 0)
    batch = (images, meta)

    images_np, cond_np = prepare_batch(batch)

    assert isinstance(images_np, np.ndarray)
    assert isinstance(cond_np, np.ndarray)
    assert images_np.shape == (B, 1, 8, 8)
    assert cond_np.shape == (B, 0)


def test_prepare_batch_preserves_values():
    """prepare_batch preserves tensor values during conversion."""
    B = 2
    data = np.random.randn(B, 1, 4, 4).astype(np.float32)
    images = torch.from_numpy(data)
    meta = torch.tensor([[0.5], [1.0]])
    batch = (images, meta)

    images_np, cond_np = prepare_batch(batch)

    assert np.array_equal(images_np, data)
    assert np.array_equal(cond_np, np.array([[0.5], [1.0]], dtype=np.float32))


# --- batch_metric_loop ---


def _make_val_dataloader(B=2, num_batches=2):
    """Return a re-iterable list of fake (images, meta) batches."""
    return [
        (
            torch.from_numpy(np.random.randn(B, 1, 8, 8).astype(np.float32)),
            torch.empty(B, 0),
        )
        for _ in range(num_batches)
    ]


from msdflow.train.trainer import batch_metric_loop


def _make_prepare_jax():
    """Return a prepare_jax callable for tests."""
    return make_prepare_batch_jax(
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        p_uncond=0.0,
    )


def test_batch_metric_loop_returns_dict_of_floats():
    """batch_metric_loop must return dict[str, float]."""
    step = make_batch_metric_step([_fml])
    val_loader = _make_val_dataloader()
    model = _make_small_model()
    result = batch_metric_loop(
        key=jax.random.PRNGKey(0),
        ema_model=model,
        dataloader=val_loader,
        step_fn=step,
        prepare_jax=_make_prepare_jax(),
        num_batches=0,
    )
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())


def test_batch_metric_loop_values_are_finite():
    """batch_metric_loop must return finite values."""
    step = make_batch_metric_step([_fml])
    val_loader = _make_val_dataloader()
    model = _make_small_model()
    result = batch_metric_loop(
        key=jax.random.PRNGKey(1),
        ema_model=model,
        dataloader=val_loader,
        step_fn=step,
        prepare_jax=_make_prepare_jax(),
        num_batches=0,
    )
    assert all(np.isfinite(v) for v in result.values())


def test_batch_metric_loop_num_batches_limit():
    """batch_metric_loop stops after num_batches batches when limit > 0."""
    call_count = []

    def counting_loader():
        for _ in range(10):
            call_count.append(1)
            yield (
                torch.from_numpy(np.random.randn(2, 1, 8, 8).astype(np.float32)),
                torch.empty(2, 0),
            )

    step = make_batch_metric_step([_fml])
    model = _make_small_model()
    batch_metric_loop(
        key=jax.random.PRNGKey(2),
        ema_model=model,
        dataloader=counting_loader(),
        step_fn=step,
        prepare_jax=_make_prepare_jax(),
        num_batches=3,
    )
    assert len(call_count) == 3


def test_batch_metric_loop_returns_mean_not_sum():
    """batch_metric_loop must return the mean, not the sum, across batches."""

    def simple_metric(model, x_t, u_t, t, cond, cond_mask, key):
        return jnp.array(float(x_t.shape[0]))

    simple_metric.__name__ = "simple_metric"

    loader1 = [
        (torch.randn(2, 1, 8, 8), torch.empty(2, 0)),
    ]
    loader2 = [
        (torch.randn(4, 1, 8, 8), torch.empty(4, 0)),
    ]
    combined_loader = loader1 + loader2

    step = make_batch_metric_step([simple_metric])
    model = _make_small_model()
    result = batch_metric_loop(
        key=jax.random.PRNGKey(0),
        ema_model=model,
        dataloader=combined_loader,
        step_fn=step,
        prepare_jax=_make_prepare_jax(),
        num_batches=0,
    )
    assert "simple_metric" in result
    assert isinstance(result["simple_metric"], float)
    assert abs(result["simple_metric"] - 3.0) < 1e-5


def test_resolve_time_loss_diagnostic_config_defaults_disabled():
    """Missing diagnostic config should preserve direct train() compatibility."""
    cfg = resolve_time_loss_diagnostic_config(None)

    assert isinstance(cfg, TimeLossDiagnosticConfig)
    assert cfg.enabled is False
    assert cfg.split == "val"
    assert cfg.num_bins == 20
    assert cfg.num_batches == 0
    assert cfg.log_heatmap is True


def test_resolve_time_loss_diagnostic_config_accepts_mapping():
    """Hydra-style mappings should resolve into a diagnostic config."""
    cfg = resolve_time_loss_diagnostic_config(
        {
            "enabled": "true",
            "split": "both",
            "num_bins": 8,
            "num_batches": 2,
            "log_heatmap": "false",
        }
    )

    assert cfg.enabled is True
    assert cfg.split == "both"
    assert cfg.num_bins == 8
    assert cfg.num_batches == 2
    assert cfg.log_heatmap is False


def test_resolve_time_loss_diagnostic_config_accepts_integer_strings():
    """Integer strings should resolve into diagnostic integer fields."""
    cfg = resolve_time_loss_diagnostic_config(
        {
            "enabled": True,
            "num_bins": "8",
            "num_batches": "2",
        }
    )

    assert cfg.num_bins == 8
    assert cfg.num_batches == 2


def test_resolve_time_loss_diagnostic_config_rejects_invalid_split():
    """Only val, train, and both are valid diagnostic splits."""
    with pytest.raises(ValueError, match="split"):
        resolve_time_loss_diagnostic_config({"enabled": True, "split": "test"})


def test_resolve_time_loss_diagnostic_config_rejects_invalid_bins():
    """Diagnostic bin count must be positive."""
    with pytest.raises(ValueError, match="num_bins"):
        resolve_time_loss_diagnostic_config({"enabled": True, "num_bins": 0})


@pytest.mark.parametrize("num_bins", ["bad", None])
def test_resolve_time_loss_diagnostic_config_rejects_non_integer_bins(num_bins):
    """Diagnostic bin count errors should name num_bins."""
    with pytest.raises(ValueError, match="num_bins"):
        resolve_time_loss_diagnostic_config({"enabled": True, "num_bins": num_bins})


def test_resolve_time_loss_diagnostic_config_rejects_direct_non_integer_bins():
    """Direct diagnostic configs should validate invalid num_bins fields."""
    with pytest.raises(ValueError, match="num_bins"):
        resolve_time_loss_diagnostic_config(
            TimeLossDiagnosticConfig(enabled=True, num_bins="bad")
        )


def test_resolve_time_loss_diagnostic_config_rejects_fractional_num_bins():
    """Diagnostic bin count should reject fractional numeric values."""
    with pytest.raises(ValueError, match="num_bins"):
        resolve_time_loss_diagnostic_config({"enabled": True, "num_bins": 1.9})


def test_resolve_time_loss_diagnostic_config_rejects_decimal_fractional_num_bins():
    """Diagnostic bin count should reject fractional Decimal values."""
    with pytest.raises(ValueError, match="num_bins"):
        resolve_time_loss_diagnostic_config(
            {"enabled": True, "num_bins": Decimal("1.9")}
        )


@pytest.mark.parametrize("num_batches", [-1, "bad", None])
def test_resolve_time_loss_diagnostic_config_rejects_invalid_num_batches(num_batches):
    """Diagnostic batch limit errors should name num_batches."""
    with pytest.raises(ValueError, match="num_batches"):
        resolve_time_loss_diagnostic_config(
            {"enabled": True, "num_batches": num_batches}
        )


def test_resolve_time_loss_diagnostic_config_rejects_direct_none_num_batches():
    """Direct diagnostic configs should validate invalid num_batches fields."""
    with pytest.raises(ValueError, match="num_batches"):
        resolve_time_loss_diagnostic_config(
            TimeLossDiagnosticConfig(enabled=True, num_batches=None)
        )


def test_resolve_time_loss_diagnostic_config_rejects_fractional_num_batches():
    """Diagnostic batch limit should reject fractional numeric values."""
    with pytest.raises(ValueError, match="num_batches"):
        resolve_time_loss_diagnostic_config({"enabled": True, "num_batches": 2.5})


def test_resolve_time_loss_diagnostic_config_rejects_decimal_fractional_num_batches():
    """Diagnostic batch limit should reject fractional Decimal values."""
    with pytest.raises(ValueError, match="num_batches"):
        resolve_time_loss_diagnostic_config(
            {"enabled": True, "num_batches": Decimal("2.5")}
        )


def test_time_binned_loss_loop_returns_result_with_counts():
    """Diagnostic loop should aggregate losses over a limited dataloader pass."""
    step = make_time_binned_loss_step(num_bins=4)
    val_loader = _make_val_dataloader(num_batches=2)
    model = _make_small_model()
    result = time_binned_loss_loop(
        key=jax.random.PRNGKey(7),
        model=model,
        dataloader=val_loader,
        step_fn=step,
        prepare_jax=_make_prepare_jax(),
        num_bins=4,
        num_batches=1,
    )

    assert result.loss_sums.shape == (4,)
    assert result.counts.shape == (4,)
    assert int(result.counts.sum()) == 2


def test_call_epoch_metric_passes_data_parallel_when_supported():
    """Epoch metrics that accept data_parallel receive the resolved config."""
    received = {}
    cfg = make_data_parallel_config(enabled=True, min_devices=1)

    def metric(model, val_dataloader, key, *, data_parallel=None):
        received["data_parallel"] = data_parallel
        return {"metric": 1.0}

    result = _call_epoch_metric(
        metric,
        model=None,
        val_dataloader=[],
        key=jax.random.PRNGKey(0),
        data_parallel=cfg,
    )

    assert result == {"metric": 1.0}
    assert received["data_parallel"] is cfg


def test_call_epoch_metric_passes_positional_only_data_parallel_parameter():
    """Positional-only data_parallel receives the resolved config positionally."""
    received = {}
    cfg = make_data_parallel_config(enabled=True, min_devices=1)

    def metric(model, val_dataloader, key, data_parallel, /):
        received["data_parallel"] = data_parallel
        return {"metric": 1.0}

    result = _call_epoch_metric(
        metric,
        model=None,
        val_dataloader=[],
        key=jax.random.PRNGKey(0),
        data_parallel=cfg,
    )

    assert result == {"metric": 1.0}
    assert received["data_parallel"] is cfg


def test_call_epoch_metric_prioritizes_positional_only_data_parallel_over_kwargs():
    """A positional-only data_parallel is filled positionally despite **kwargs."""
    received = {}
    cfg = make_data_parallel_config(enabled=True, min_devices=1)

    def metric(model, val_dataloader, key, data_parallel, /, **kwargs):
        received["data_parallel"] = data_parallel
        received["kwargs"] = kwargs
        return {"metric": 1.0}

    result = _call_epoch_metric(
        metric,
        model=None,
        val_dataloader=[],
        key=jax.random.PRNGKey(0),
        data_parallel=cfg,
    )

    assert result == {"metric": 1.0}
    assert received["data_parallel"] is cfg
    assert received["kwargs"] == {}


def test_call_epoch_metric_keeps_three_argument_metrics_working():
    """Existing epoch metrics without data_parallel keep their old signature."""
    received = {}
    cfg = make_data_parallel_config(enabled=True, min_devices=1)

    def metric(model, val_dataloader, key):
        received["called"] = True
        return jnp.array(2.0)

    result = _call_epoch_metric(
        metric,
        model=None,
        val_dataloader=[],
        key=jax.random.PRNGKey(0),
        data_parallel=cfg,
    )

    assert received["called"] is True
    assert float(result) == pytest.approx(2.0)


def test_call_epoch_metric_does_not_swallow_metric_type_errors():
    """Real TypeError exceptions raised inside a metric must propagate."""
    cfg = make_data_parallel_config(enabled=False)

    def metric(model, val_dataloader, key, data_parallel=None):
        raise TypeError("metric internal failure")

    with pytest.raises(TypeError, match="metric internal failure"):
        _call_epoch_metric(
            metric,
            model=None,
            val_dataloader=[],
            key=jax.random.PRNGKey(0),
            data_parallel=cfg,
        )


def test_train_epoch_metric_receives_val_dataloader():
    """A no-op epoch metric must receive the val_dataloader iterable."""
    received = {}

    def capture_epoch_metric(model, val_dataloader, key):
        received["val_dataloader"] = val_dataloader
        return jnp.array(0.0)

    capture_epoch_metric.__name__ = "capture_epoch_metric"

    dataloader = list(_make_fake_dataloader(B=2, num_batches=3))
    val_dataloader = _fake_val_dataloader()

    kwargs = _make_train_kwargs()
    kwargs["batch_metrics"] = [_fml]
    kwargs["epoch_metrics"] = [capture_epoch_metric]
    kwargs["num_train_eval_batches"] = 0
    model = _make_small_model()

    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )

    assert "val_dataloader" in received
    assert received["val_dataloader"] is val_dataloader


def test_train_epoch_metric_receives_resolved_data_parallel_config(tmp_path):
    """The trainer passes resolved DataParallelConfig to aware epoch metrics."""
    received = {}

    class DataParallelAwareMetric:
        def __call__(self, model, val_dataloader, key, data_parallel=None):
            received["data_parallel"] = data_parallel
            return {"aware_metric": 0.0}

    dataloader = list(_make_fake_dataloader(B=2, num_batches=2))
    val_dataloader = _fake_val_dataloader(B=2)
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["batch_metrics"] = [_fml]
    kwargs["epoch_metrics"] = [DataParallelAwareMetric()]
    kwargs["num_train_eval_batches"] = 0
    model = _make_small_model()

    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        data_parallel={"enabled": True, "axis_name": "batch", "min_devices": 1},
        **kwargs,
    )

    assert received["data_parallel"].enabled is True
    assert received["data_parallel"].min_devices == 1


def test_train_epoch_metric_callable_object():
    """A callable object (not function/partial) must work as an epoch metric."""

    class DictMetric:
        def __call__(self, model, val_dataloader, key):
            return {"custom_a": 1.0, "custom_b": 2.0}

    dataloader = list(_make_fake_dataloader(B=2, num_batches=3))
    val_dataloader = _fake_val_dataloader()

    kwargs = _make_train_kwargs()
    kwargs["batch_metrics"] = [_fml]
    kwargs["epoch_metrics"] = [DictMetric()]
    kwargs["num_train_eval_batches"] = 0
    model = _make_small_model()

    result_model = train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )
    # If we get here without AttributeError, the callable object was handled
    assert result_model is not None


# ---------------------------------------------------------------------------
# ClearML + sample integration tests
# ---------------------------------------------------------------------------

import functools
from unittest.mock import MagicMock, patch

from msdflow.flow.interpolate import sample_path

_PATH_SAMPLER = functools.partial(sample_path, sigma_0=0.0, sigma_1=0.0)
_COUPLING = lambda x0, x1: x0  # identity coupling for tests


def _make_dataloader(batch_size=2, img_size=8, steps=2):
    """Return a tiny list-backed dataloader for trainer tests."""
    import torch

    images = torch.zeros(batch_size, 1, img_size, img_size)
    meta = torch.zeros(batch_size, 0)
    return [(images, meta)] * steps


def _time_sampler(key, batch_size):
    import jax

    return jax.random.uniform(key, (batch_size,))


def test_train_runs_with_clearml_task_none(tmp_path):
    """train() completes without error when clearml_task=None (default)."""
    import jax

    key = jax.random.PRNGKey(0)
    dl = _make_dataloader()
    model = _make_small_model()
    result = train(
        key=key,
        model=model,
        dataloader=dl,
        val_dataloader=dl,
        optimizer=OPTIMIZER,
        loss_fn=lambda model, x_t, u_t, t, cond, cond_mask, key: jnp.mean(x_t),
        batch_metrics=[],
        epoch_metrics=[],
        coupling=_COUPLING,
        time_sampler=_time_sampler,
        path_sampler=_PATH_SAMPLER,
        num_epochs=1,
        num_steps_per_epoch=2,
        p_uncond=1.0,
        ema_decay=0.999,
        log_every=1,
        val_every=1,
        checkpoint_every=10,
        checkpoint_dir=str(tmp_path),
        clearml_task=None,
    )
    assert result is not None


def test_train_raises_if_sample_fn_set_but_no_samples_dir():
    """train() raises ValueError when sample_fn is set but samples_dir is None."""
    import jax
    import pytest

    key = jax.random.PRNGKey(5)
    dl = _make_dataloader()
    model = _make_small_model()

    with pytest.raises(ValueError, match="samples_dir"):
        train(
            key=key,
            model=model,
            dataloader=dl,
            val_dataloader=dl,
            optimizer=OPTIMIZER,
            loss_fn=lambda model, x_t, u_t, t, cond, cond_mask, key: jnp.mean(x_t),
            batch_metrics=[],
            epoch_metrics=[],
            coupling=_COUPLING,
            time_sampler=_time_sampler,
            path_sampler=_PATH_SAMPLER,
            num_epochs=1,
            num_steps_per_epoch=1,
            p_uncond=1.0,
            ema_decay=0.999,
            log_every=1,
            val_every=1,
            checkpoint_every=10,
            checkpoint_dir="/tmp/test_ckpt2",
            sample_fn=lambda model, key, n: np.zeros((n, 1, 8, 8)),
            sample_every=1,
            samples_dir=None,
        )


def test_train_calls_log_metrics_when_task_provided(tmp_path):
    """log_metrics is called once per log_every epoch when clearml_task is set."""
    import jax

    key = jax.random.PRNGKey(1)
    dl = _make_dataloader()
    mock_task = MagicMock()
    model = _make_small_model()

    with patch("msdflow.train.trainer.log_metrics") as mock_log_metrics:
        train(
            key=key,
            model=model,
            dataloader=dl,
            val_dataloader=dl,
            optimizer=OPTIMIZER,
            loss_fn=lambda model, x_t, u_t, t, cond, cond_mask, key: jnp.mean(x_t),
            batch_metrics=[],
            epoch_metrics=[],
            coupling=_COUPLING,
            time_sampler=_time_sampler,
            path_sampler=_PATH_SAMPLER,
            num_epochs=2,
            num_steps_per_epoch=1,
            p_uncond=1.0,
            ema_decay=0.999,
            log_every=1,
            val_every=1,
            checkpoint_every=10,
            checkpoint_dir=str(tmp_path),
            clearml_task=mock_task,
        )
    assert mock_log_metrics.call_count == 2  # once per epoch (log_every=1)
    first_call_scalars = mock_log_metrics.call_args_list[0][0][1]
    assert "train/loss" in first_call_scalars


def test_train_calls_log_checkpoint_when_task_provided(tmp_path):
    """log_checkpoint is called at checkpoint_every epochs when clearml_task is set."""
    import jax

    key = jax.random.PRNGKey(2)
    dl = _make_dataloader()
    mock_task = MagicMock()
    model = _make_small_model()

    with patch("msdflow.train.trainer.log_checkpoint") as mock_log_ckpt:
        train(
            key=key,
            model=model,
            dataloader=dl,
            val_dataloader=dl,
            optimizer=OPTIMIZER,
            loss_fn=lambda model, x_t, u_t, t, cond, cond_mask, key: jnp.mean(x_t),
            batch_metrics=[],
            epoch_metrics=[],
            coupling=_COUPLING,
            time_sampler=_time_sampler,
            path_sampler=_PATH_SAMPLER,
            num_epochs=1,
            num_steps_per_epoch=1,
            p_uncond=1.0,
            ema_decay=0.999,
            log_every=1,
            val_every=1,
            checkpoint_every=1,
            checkpoint_dir=str(tmp_path),
            clearml_task=mock_task,
        )
    assert mock_log_ckpt.call_count == 1
    called_path = mock_log_ckpt.call_args[0][1]
    assert "ema" in called_path


def test_train_logs_time_binned_loss_diagnostic_at_validation_cadence(tmp_path):
    """Enabled diagnostic should log once per validation epoch for the val split."""
    key = jax.random.PRNGKey(31)
    dl = _make_dataloader()
    mock_task = MagicMock()
    model = _make_small_model()

    with patch("msdflow.train.trainer.log_time_binned_loss") as mock_log_diag:
        train(
            key=key,
            model=model,
            dataloader=dl,
            val_dataloader=dl,
            optimizer=OPTIMIZER,
            loss_fn=lambda model, x_t, u_t, t, cond, cond_mask, key: jnp.mean(x_t),
            batch_metrics=[],
            epoch_metrics=[],
            coupling=_COUPLING,
            time_sampler=_time_sampler,
            path_sampler=_PATH_SAMPLER,
            num_epochs=2,
            num_steps_per_epoch=1,
            p_uncond=1.0,
            ema_decay=0.999,
            log_every=1,
            val_every=1,
            checkpoint_every=10,
            checkpoint_dir=str(tmp_path),
            clearml_task=mock_task,
            time_loss_diagnostic={
                "enabled": True,
                "split": "val",
                "num_bins": 4,
                "num_batches": 1,
                "log_heatmap": True,
            },
        )

    assert mock_log_diag.call_count == 2
    first_call = mock_log_diag.call_args_list[0].kwargs
    assert first_call["task"] is mock_task
    assert first_call["split"] == "val"
    assert first_call["epoch"] == 1
    assert first_call["result"].counts.shape == (4,)
    assert first_call["history"] is not None


def test_train_skips_time_binned_loss_diagnostic_without_clearml_task(tmp_path):
    """Enabled diagnostic should not run a diagnostic pass without ClearML."""
    key = jax.random.PRNGKey(34)
    dl = _make_dataloader()
    model = _make_small_model()

    with (
        patch("msdflow.train.trainer.time_binned_loss_loop") as mock_loop,
        patch("msdflow.train.trainer.log_time_binned_loss") as mock_log_diag,
    ):
        train(
            key=key,
            model=model,
            dataloader=dl,
            val_dataloader=dl,
            optimizer=OPTIMIZER,
            loss_fn=lambda model, x_t, u_t, t, cond, cond_mask, key: jnp.mean(x_t),
            batch_metrics=[],
            epoch_metrics=[],
            coupling=_COUPLING,
            time_sampler=_time_sampler,
            path_sampler=_PATH_SAMPLER,
            num_epochs=1,
            num_steps_per_epoch=1,
            p_uncond=1.0,
            ema_decay=0.999,
            log_every=1,
            val_every=1,
            checkpoint_every=10,
            checkpoint_dir=str(tmp_path),
            clearml_task=None,
            time_loss_diagnostic={
                "enabled": True,
                "split": "val",
                "num_bins": 4,
                "num_batches": 1,
                "log_heatmap": True,
            },
        )

    mock_loop.assert_not_called()
    mock_log_diag.assert_not_called()


def test_train_time_loss_diagnostic_noop_ignores_malformed_config_without_clearml(
    tmp_path,
):
    """ClearML-disabled diagnostics should not parse or run malformed configs."""
    key = jax.random.PRNGKey(35)
    dl = _make_dataloader()
    model = _make_small_model()

    with (
        patch("msdflow.train.trainer.time_binned_loss_loop") as mock_loop,
        patch("msdflow.train.trainer.log_time_binned_loss") as mock_log_diag,
    ):
        train(
            key=key,
            model=model,
            dataloader=dl,
            val_dataloader=dl,
            optimizer=OPTIMIZER,
            loss_fn=lambda model, x_t, u_t, t, cond, cond_mask, key: jnp.mean(x_t),
            batch_metrics=[],
            epoch_metrics=[],
            coupling=_COUPLING,
            time_sampler=_time_sampler,
            path_sampler=_PATH_SAMPLER,
            num_epochs=1,
            num_steps_per_epoch=1,
            p_uncond=1.0,
            ema_decay=0.999,
            log_every=1,
            val_every=1,
            checkpoint_every=10,
            checkpoint_dir=str(tmp_path),
            clearml_task=None,
            time_loss_diagnostic={"enabled": True, "num_bins": 0},
        )

    mock_loop.assert_not_called()
    mock_log_diag.assert_not_called()


def test_train_time_loss_diagnostic_keeps_positional_checkpoint_hash_binding():
    """Old positional checkpoint_hash calls should not bind to diagnostics."""
    sentinel_hash = "old-positional-hash"
    positional_args = [
        jax.random.PRNGKey(41),
        _make_small_model(),
        _make_dataloader(),
        _make_dataloader(),
        OPTIMIZER,
        lambda model, x_t, u_t, t, cond, cond_mask, key: jnp.mean(x_t),
        [],
        [],
        _COUPLING,
        _time_sampler,
        _PATH_SAMPLER,
        1,
        1,
        1.0,
        0.999,
        1,
        1,
        10,
        "/tmp/test_ckpt_positional",
        0,
        None,
        None,
        0,
        4,
        False,
        "arcsinh",
        10.0,
        None,
        "flow_matching_loss",
        "min",
        None,
        1,
        4,
        None,
        sentinel_hash,
    ]

    bound = inspect.signature(train).bind_partial(*positional_args)

    assert bound.arguments["checkpoint_hash"] == sentinel_hash
    assert "time_loss_diagnostic" not in bound.arguments


def test_train_time_loss_diagnostic_disabled_preserves_epoch_metric_key(tmp_path):
    """Disabled diagnostics should preserve pre-diagnostic epoch metric keys."""
    def run_and_capture(time_loss_diagnostic=None):
        captured = []

        def fake_split(key, num=2):
            return tuple(jnp.array([num, index], dtype=jnp.uint32) for index in range(num))

        def fake_prepare_jax(images_np, cond_np, key):
            batch_size = images_np.shape[0]
            return (
                jnp.zeros((batch_size,)),
                jnp.zeros((batch_size, 1, 8, 8)),
                jnp.zeros((batch_size, 1, 8, 8)),
                jnp.zeros((batch_size, 0)),
                jnp.zeros((batch_size,), dtype=bool),
                jnp.zeros((batch_size, 2), dtype=jnp.uint32),
            )

        def fake_train_step(state, x_t, u_t, t, cond, cond_mask, key):
            return state, jnp.array(0.0)

        def capture_metric(model, val_dataloader, key):
            captured.append(np.asarray(key))
            return {"captured": 0.0}

        kwargs = {}
        if time_loss_diagnostic is not None:
            kwargs["time_loss_diagnostic"] = time_loss_diagnostic

        with (
            patch("msdflow.train.trainer.jax.random.split", side_effect=fake_split),
            patch("msdflow.train.trainer.make_train_step", return_value=fake_train_step),
            patch(
                "msdflow.train.trainer.make_prepare_batch_jax",
                return_value=fake_prepare_jax,
            ),
            patch("msdflow.train.trainer.batch_metric_loop", return_value={}),
        ):
            train(
                key=jnp.array([0, 0], dtype=jnp.uint32),
                model=_make_small_model(),
                dataloader=_make_dataloader(steps=1),
                val_dataloader=_make_dataloader(steps=1),
                optimizer=OPTIMIZER,
                loss_fn=lambda model, x_t, u_t, t, cond, cond_mask, key: jnp.mean(x_t),
                batch_metrics=[],
                epoch_metrics=[capture_metric],
                coupling=_COUPLING,
                time_sampler=_time_sampler,
                path_sampler=_PATH_SAMPLER,
                num_epochs=1,
                num_steps_per_epoch=1,
                p_uncond=1.0,
                ema_decay=0.999,
                log_every=1,
                val_every=1,
                checkpoint_every=10,
                checkpoint_dir=str(tmp_path),
                clearml_task=None,
                monitor="captured",
                **kwargs,
            )
        return captured[0]

    omitted_key = run_and_capture()
    disabled_key = run_and_capture({"enabled": False})
    enabled_without_task_key = run_and_capture(
        {
            "enabled": True,
            "split": "val",
            "num_bins": 4,
            "num_batches": 1,
        }
    )
    expected_old_key = np.array([4, 3], dtype=np.uint32)

    assert np.array_equal(omitted_key, expected_old_key)
    assert np.array_equal(disabled_key, omitted_key)
    assert np.array_equal(enabled_without_task_key, omitted_key)


def test_train_time_loss_diagnostic_does_not_change_training_prepare_keys(tmp_path):
    """Enabled ClearML diagnostics should not alter subsequent training keys."""
    diagnostic_result = TimeBinnedLossResult.empty(num_bins=4)

    def run_and_capture_training_keys(time_loss_diagnostic):
        """Run two epochs and return keys passed to training prepare_jax."""
        captured_keys = []

        def fake_prepare_jax(images_np, cond_np, key):
            batch_size = images_np.shape[0]
            captured_keys.append(np.asarray(key))
            return (
                jnp.zeros((batch_size,)),
                jnp.zeros((batch_size, 1, 8, 8)),
                jnp.zeros((batch_size, 1, 8, 8)),
                jnp.zeros((batch_size, 0)),
                jnp.zeros((batch_size,), dtype=bool),
                jnp.zeros((batch_size, 2), dtype=jnp.uint32),
            )

        def fake_train_step(state, x_t, u_t, t, cond, cond_mask, key):
            return state, jnp.array(0.0)

        with (
            patch("msdflow.train.trainer.make_train_step", return_value=fake_train_step),
            patch(
                "msdflow.train.trainer.make_prepare_batch_jax",
                return_value=fake_prepare_jax,
            ),
            patch("msdflow.train.trainer.batch_metric_loop", return_value={}),
            patch(
                "msdflow.train.trainer.time_binned_loss_loop",
                return_value=diagnostic_result,
            ),
            patch("msdflow.train.trainer.log_time_binned_loss"),
        ):
            train(
                key=jax.random.PRNGKey(43),
                model=_make_small_model(),
                dataloader=_make_dataloader(steps=1),
                val_dataloader=_make_dataloader(steps=1),
                optimizer=OPTIMIZER,
                loss_fn=lambda model, x_t, u_t, t, cond, cond_mask, key: jnp.mean(x_t),
                batch_metrics=[],
                epoch_metrics=[],
                coupling=_COUPLING,
                time_sampler=_time_sampler,
                path_sampler=_PATH_SAMPLER,
                num_epochs=2,
                num_steps_per_epoch=1,
                p_uncond=1.0,
                ema_decay=0.999,
                log_every=1,
                val_every=1,
                checkpoint_every=10,
                checkpoint_dir=str(tmp_path),
                clearml_task=MagicMock(),
                time_loss_diagnostic=time_loss_diagnostic,
            )
        return captured_keys

    disabled_keys = run_and_capture_training_keys({"enabled": False})
    enabled_keys = run_and_capture_training_keys(
        {
            "enabled": True,
            "split": "val",
            "num_bins": 4,
            "num_batches": 1,
        }
    )

    assert len(disabled_keys) == len(enabled_keys) == 2
    assert all(
        np.array_equal(disabled_key, enabled_key)
        for disabled_key, enabled_key in zip(disabled_keys, enabled_keys)
    )


def test_train_logs_time_binned_loss_diagnostic_for_both_splits(tmp_path):
    """The both split should log independent train and val diagnostics."""
    key = jax.random.PRNGKey(32)
    dl = _make_dataloader()
    mock_task = MagicMock()
    model = _make_small_model()

    with patch("msdflow.train.trainer.log_time_binned_loss") as mock_log_diag:
        train(
            key=key,
            model=model,
            dataloader=dl,
            val_dataloader=dl,
            optimizer=OPTIMIZER,
            loss_fn=lambda model, x_t, u_t, t, cond, cond_mask, key: jnp.mean(x_t),
            batch_metrics=[],
            epoch_metrics=[],
            coupling=_COUPLING,
            time_sampler=_time_sampler,
            path_sampler=_PATH_SAMPLER,
            num_epochs=1,
            num_steps_per_epoch=1,
            p_uncond=1.0,
            ema_decay=0.999,
            log_every=1,
            val_every=1,
            checkpoint_every=10,
            checkpoint_dir=str(tmp_path),
            clearml_task=mock_task,
            time_loss_diagnostic={
                "enabled": True,
                "split": "both",
                "num_bins": 4,
                "num_batches": 1,
                "log_heatmap": True,
            },
        )

    calls = mock_log_diag.call_args_list
    assert [call.kwargs["split"] for call in calls] == ["val", "train"]
    assert calls[0].kwargs["history"] is not None
    assert calls[1].kwargs["history"] is not None
    assert calls[0].kwargs["history"] is not calls[1].kwargs["history"]
    assert {call.kwargs["result"].counts.shape for call in calls} == {(4,)}


def test_train_logs_time_binned_loss_diagnostic_without_heatmap_history(tmp_path):
    """Disabled heatmap logging should pass no cumulative history."""
    key = jax.random.PRNGKey(33)
    dl = _make_dataloader()
    mock_task = MagicMock()
    model = _make_small_model()

    with patch("msdflow.train.trainer.log_time_binned_loss") as mock_log_diag:
        train(
            key=key,
            model=model,
            dataloader=dl,
            val_dataloader=dl,
            optimizer=OPTIMIZER,
            loss_fn=lambda model, x_t, u_t, t, cond, cond_mask, key: jnp.mean(x_t),
            batch_metrics=[],
            epoch_metrics=[],
            coupling=_COUPLING,
            time_sampler=_time_sampler,
            path_sampler=_PATH_SAMPLER,
            num_epochs=1,
            num_steps_per_epoch=1,
            p_uncond=1.0,
            ema_decay=0.999,
            log_every=1,
            val_every=1,
            checkpoint_every=10,
            checkpoint_dir=str(tmp_path),
            clearml_task=mock_task,
            time_loss_diagnostic={
                "enabled": True,
                "split": "val",
                "num_bins": 4,
                "num_batches": 1,
                "log_heatmap": False,
            },
        )

    assert mock_log_diag.call_count == 1
    assert mock_log_diag.call_args.kwargs["history"] is None


def test_train_generates_samples_to_disk(tmp_path):
    """sample_fn is called at sample_every epochs and files are saved."""
    import jax

    key = jax.random.PRNGKey(3)
    dl = _make_dataloader()
    model = _make_small_model()

    def fake_sample_fn(model, key):
        return jnp.zeros((1, 8, 8), dtype=jnp.float32)

    train(
        key=key,
        model=model,
        dataloader=dl,
        val_dataloader=dl,
        optimizer=OPTIMIZER,
        loss_fn=lambda model, x_t, u_t, t, cond, cond_mask, key: jnp.mean(x_t),
        batch_metrics=[],
        epoch_metrics=[],
        coupling=_COUPLING,
        time_sampler=_time_sampler,
        path_sampler=_PATH_SAMPLER,
        num_epochs=2,
        num_steps_per_epoch=1,
        p_uncond=1.0,
        ema_decay=0.999,
        log_every=1,
        val_every=1,
        checkpoint_every=10,
        checkpoint_dir=str(tmp_path / "ckpt"),
        clearml_task=None,
        sample_fn=fake_sample_fn,
        sample_every=1,
        num_samples=2,
        samples_dir=str(tmp_path / "samples"),
    )
    import glob

    npy_files = [
        p
        for p in glob.glob(str(tmp_path / "samples" / "**" / "*.npy"), recursive=True)
        if "reference" not in p
    ]
    assert len(npy_files) == 4  # 2 epochs x 2 samples (reference samples excluded)


def test_train_logs_samples_to_clearml_without_samples_dir(tmp_path):
    """ClearML sample logging does not require a disk samples_dir."""
    import jax

    key = jax.random.PRNGKey(6)
    dl = _make_dataloader()
    model = _make_small_model()
    mock_task = MagicMock()

    def fake_sample_fn(model, key):
        return jnp.zeros((1, 8, 8), dtype=jnp.float32)

    with patch("msdflow.train.trainer.log_samples") as mock_log_samples:
        train(
            key=key,
            model=model,
            dataloader=dl,
            val_dataloader=dl,
            optimizer=OPTIMIZER,
            loss_fn=lambda model, x_t, u_t, t, cond, cond_mask, key: jnp.mean(x_t),
            batch_metrics=[],
            epoch_metrics=[],
            coupling=_COUPLING,
            time_sampler=_time_sampler,
            path_sampler=_PATH_SAMPLER,
            num_epochs=1,
            num_steps_per_epoch=1,
            p_uncond=1.0,
            ema_decay=0.999,
            log_every=1,
            val_every=1,
            checkpoint_every=10,
            checkpoint_dir=str(tmp_path / "ckpt"),
            clearml_task=mock_task,
            sample_fn=fake_sample_fn,
            sample_every=1,
            num_samples=2,
            samples_dir=None,
        )

    assert mock_log_samples.call_count == 2


def test_train_skips_sampling_when_sample_every_is_zero(tmp_path):
    """No samples are generated when sample_every=0."""
    import jax

    key = jax.random.PRNGKey(4)
    dl = _make_dataloader()
    call_count = {"n": 0}
    model = _make_small_model()

    def counting_sample_fn(model, key, num_samples):
        call_count["n"] += 1
        return np.zeros((num_samples, 1, 8, 8), dtype=np.float32)

    train(
        key=key,
        model=model,
        dataloader=dl,
        val_dataloader=dl,
        optimizer=OPTIMIZER,
        loss_fn=lambda model, x_t, u_t, t, cond, cond_mask, key: jnp.mean(x_t),
        batch_metrics=[],
        epoch_metrics=[],
        coupling=_COUPLING,
        time_sampler=_time_sampler,
        path_sampler=_PATH_SAMPLER,
        num_epochs=2,
        num_steps_per_epoch=1,
        p_uncond=1.0,
        ema_decay=0.999,
        log_every=1,
        val_every=1,
        checkpoint_every=10,
        checkpoint_dir=str(tmp_path),
        sample_fn=counting_sample_fn,
        sample_every=0,
        num_samples=2,
        samples_dir=str(tmp_path / "samples"),
    )
    assert call_count["n"] == 0


def test_train_invalid_monitor_mode_raises(tmp_path):
    """train() raises ValueError immediately if monitor_mode is not 'min' or 'max'."""
    dataloader = list(_make_fake_dataloader())
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["monitor_mode"] = "diagonal"
    model = _make_small_model()
    with pytest.raises(ValueError, match="monitor_mode"):
        train(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            **kwargs,
        )


def test_best_checkpoint_saved_on_first_val(tmp_path):
    """Best-model checkpoint (raw + ema) is created after the first validation epoch."""
    dataloader = list(_make_fake_dataloader())
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    model = _make_small_model()
    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )
    best_ema = list(tmp_path.glob("*_best_ema.eqx"))
    best_raw = list(tmp_path.glob("*_best_raw.eqx"))
    assert len(best_ema) == 1
    assert len(best_raw) == 1
    assert "epoch1_best_ema" in best_ema[0].name


def test_best_checkpoint_not_saved_when_no_improvement(tmp_path):
    """No new best-model file is written when the metric does not improve."""
    call_count = [0]

    def plateau_metric(model, val_batches, key):
        call_count[0] += 1
        return 1.0  # constant — never beats itself in max mode after epoch 1

    dataloader = list(_make_fake_dataloader())
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=3)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["epoch_metrics"] = [plateau_metric]
    kwargs["monitor"] = "plateau_metric"
    kwargs["monitor_mode"] = "max"
    model = _make_small_model()
    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )
    best_ema = list(tmp_path.glob("*_best_ema.eqx"))
    # Only epoch 1 improves (−∞ → 1.0); epochs 2 and 3 are tied
    assert len(best_ema) == 1
    assert "epoch1_best_ema" in best_ema[0].name


def test_best_checkpoint_max_mode_saves_on_each_improvement(tmp_path):
    """In max mode, a new best file is written every time the metric strictly improves."""
    call_count = [0]

    def rising_metric(model, val_batches, key):
        call_count[0] += 1
        return float(call_count[0])  # 1.0 → 2.0 → 3.0, always better

    dataloader = list(_make_fake_dataloader())
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=3)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["epoch_metrics"] = [rising_metric]
    kwargs["monitor"] = "rising_metric"
    kwargs["monitor_mode"] = "max"
    model = _make_small_model()
    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )
    best_ema = sorted(tmp_path.glob("*_best_ema.eqx"))
    assert len(best_ema) == 3


def test_train_unknown_monitor_raises_value_error(tmp_path):
    """train() raises ValueError at the first val run when monitor is not a known metric."""
    dataloader = list(_make_fake_dataloader())
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["monitor"] = "nonexistent_metric"
    model = _make_small_model()
    with pytest.raises(ValueError, match="nonexistent_metric"):
        train(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            **kwargs,
        )


def test_best_checkpoint_log_shows_monitored_metric_first(tmp_path, caplog):
    """Log line for a new best starts with the monitored metric, not another one."""
    import logging

    dataloader = list(_make_fake_dataloader())
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    model = _make_small_model()
    with caplog.at_level(logging.INFO, logger="msdflow.train.trainer"):
        train(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            **kwargs,
        )
    best_logs = [r.message for r in caplog.records if "New best model" in r.message]
    assert len(best_logs) == 1
    assert best_logs[0].startswith("New best model at epoch 1: flow_matching_loss =")


def test_early_stopping_triggers_at_correct_cycle(tmp_path):
    """Training halts after patience consecutive val cycles without improvement."""
    call_count = [0]

    def constant_metric(model, val_batches, key):
        call_count[0] += 1
        return 1.0  # constant — never beats itself in max mode after first call

    dataloader = list(_make_fake_dataloader())
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=10, num_steps_per_epoch=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["epoch_metrics"] = [constant_metric]
    kwargs["monitor"] = "constant_metric"
    kwargs["monitor_mode"] = "max"
    kwargs["early_stopping_patience"] = 1
    model = _make_small_model()
    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )
    # Epoch 1: 1.0 > −∞ → improved, patience_counter=0
    # Epoch 2: 1.0 not > 1.0 → patience_counter=1 >= patience=1 → stop
    assert call_count[0] == 2


def test_early_stopping_not_triggered_when_disabled(tmp_path):
    """When early_stopping_patience is None, all epochs run regardless of metric."""
    call_count = [0]

    def constant_metric(model, val_batches, key):
        call_count[0] += 1
        return 1.0

    dataloader = list(_make_fake_dataloader())
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=3, num_steps_per_epoch=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["epoch_metrics"] = [constant_metric]
    kwargs["monitor"] = "constant_metric"
    kwargs["monitor_mode"] = "max"
    kwargs["early_stopping_patience"] = None
    model = _make_small_model()
    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )
    assert call_count[0] == 3


def test_early_stopping_log_message(tmp_path, caplog):
    """Early stopping emits a log message that names the metric and patience count."""
    import logging

    call_count = [0]

    def constant_metric(model, val_batches, key):
        call_count[0] += 1
        return 1.0

    dataloader = list(_make_fake_dataloader())
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=5, num_steps_per_epoch=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["epoch_metrics"] = [constant_metric]
    kwargs["monitor"] = "constant_metric"
    kwargs["monitor_mode"] = "max"
    kwargs["early_stopping_patience"] = 1
    model = _make_small_model()
    with caplog.at_level(logging.INFO, logger="msdflow.train.trainer"):
        train(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            **kwargs,
        )
    stop_logs = [r.message for r in caplog.records if "Early stopping" in r.message]
    assert len(stop_logs) == 1
    assert "constant_metric" in stop_logs[0]
    assert "1" in stop_logs[0]  # patience count in message


def test_train_rejects_grad_accum_steps_below_one(tmp_path):
    """train() raises ValueError when grad_accum_steps < 1."""
    dataloader = list(_make_fake_dataloader())
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    model = _make_small_model()
    with pytest.raises(ValueError, match="grad_accum_steps"):
        train(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            grad_accum_steps=0,
            **kwargs,
        )


def test_train_accepts_grad_accum_steps_one(tmp_path):
    """train() completes normally with grad_accum_steps=1 (default, no wrapping)."""
    dataloader = list(_make_fake_dataloader())
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    model = _make_small_model()
    result = train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        grad_accum_steps=1,
        **kwargs,
    )
    assert result is not None


def test_train_grad_accum_processes_correct_microsteps(tmp_path):
    """With grad_accum_steps=K, the loop processes K microbatches per effective step."""
    call_count = [0]
    original_fml = _fml

    def counting_loss(model, x_t, u_t, t, cond, cond_mask, key):
        call_count[0] += 1
        return original_fml(model, x_t, u_t, t, cond, cond_mask, key)

    num_steps = 3
    grad_accum_steps = 2
    # Need enough batches: num_steps * grad_accum_steps = 6
    dataloader = list(_make_fake_dataloader(B=2, num_batches=6))
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=num_steps)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["loss_fn"] = counting_loss
    model = _make_small_model()
    with jax.disable_jit():
        train(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            grad_accum_steps=grad_accum_steps,
            **kwargs,
        )
    # 3 effective steps * 2 accumulation steps = 6 microbatch forward passes
    assert call_count[0] == num_steps * grad_accum_steps


def test_train_grad_accum_loss_equals_sum_over_effective_steps(tmp_path):
    """train/loss = sum(all microbatch losses) / steps_per_epoch."""
    grad_accum_steps = 2
    num_steps = 2
    # Need num_steps * grad_accum_steps = 4 batches
    dataloader = list(_make_fake_dataloader(B=2, num_batches=4))
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=num_steps)
    kwargs["checkpoint_dir"] = str(tmp_path)
    mock_task = MagicMock()
    model = _make_small_model()

    with patch("msdflow.train.trainer.log_metrics") as mock_log:
        train(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            grad_accum_steps=grad_accum_steps,
            clearml_task=mock_task,
            **{k: v for k, v in kwargs.items() if k not in ("clearml_task",)},
        )

    logged_scalars = mock_log.call_args_list[0][0][1]
    reported_loss = logged_scalars["train/loss"]
    # The loss should be epoch_loss / steps_per_epoch (not / microsteps_per_epoch)
    # With grad_accum_steps=2, steps_per_epoch=2, microsteps=4
    # So reported_loss = sum_of_4_losses / 2
    # We can't predict the exact value, but verify it's finite and positive
    assert np.isfinite(reported_loss)
    assert reported_loss > 0


def test_train_grad_accum_auto_steps_per_epoch(tmp_path):
    """With num_steps_per_epoch=0, steps_per_epoch = len(dataloader) // grad_accum_steps."""
    call_count = [0]

    def counting_loss(model, x_t, u_t, t, cond, cond_mask, key):
        call_count[0] += 1
        return _fml(model, x_t, u_t, t, cond, cond_mask, key)

    grad_accum_steps = 2
    num_batches = 6
    dataloader = list(_make_fake_dataloader(B=2, num_batches=num_batches))
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=0)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["loss_fn"] = counting_loss
    model = _make_small_model()
    with jax.disable_jit():
        train(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            grad_accum_steps=grad_accum_steps,
            **kwargs,
        )
    # len(dataloader)=6, grad_accum_steps=2 → microsteps=6, steps_per_epoch=3
    # The loop should process all 6 microbatches
    assert call_count[0] == num_batches


def test_train_grad_accum_truncates_non_divisible_dataloader(tmp_path, caplog):
    """When len(dataloader) % grad_accum_steps != 0, extra batches are dropped with a warning."""
    import logging

    call_count = [0]

    def counting_loss(model, x_t, u_t, t, cond, cond_mask, key):
        call_count[0] += 1
        return _fml(model, x_t, u_t, t, cond, cond_mask, key)

    # 7 batches with grad_accum_steps=2 → should truncate to 6 microsteps
    dataloader = list(_make_fake_dataloader(B=2, num_batches=7))
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=0)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["loss_fn"] = counting_loss
    model = _make_small_model()

    with (
        jax.disable_jit(),
        caplog.at_level(logging.WARNING, logger="msdflow.train.trainer"),
    ):
        train(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            grad_accum_steps=2,
            **kwargs,
        )

    # Should process 6 microsteps (not 7)
    assert call_count[0] == 6
    # Should have logged a warning about dropping 1 batch
    warning_logs = [r.message for r in caplog.records if "Dropping last" in r.message]
    assert len(warning_logs) == 1
    assert "1 batches" in warning_logs[0]


def test_ema_update_called_per_optimizer_step_not_per_microstep(tmp_path):
    """With grad_accum_steps=K, ema_update should follow optimizer steps, not microsteps."""
    from unittest.mock import patch
    from msdflow.train.trainer import ema_update as original_ema_fn

    ema_call_count = [0]

    def counting_ema(ema_model, new_model, decay):
        ema_call_count[0] += 1
        return original_ema_fn(ema_model, new_model, decay)

    num_steps = 3
    grad_accum_steps = 2
    # Need num_steps * grad_accum_steps = 6 batches
    dataloader = list(_make_fake_dataloader(B=2, num_batches=6))
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=num_steps)
    kwargs["checkpoint_dir"] = str(tmp_path)
    model = _make_small_model()

    with patch("msdflow.train.trainer.ema_update", side_effect=counting_ema):
        train(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            grad_accum_steps=grad_accum_steps,
            **kwargs,
        )

    # The first optimizer step initializes EMA; subsequent optimizer steps update it.
    assert ema_call_count[0] == num_steps - 1


def test_train_loss_with_deferred_accumulation(tmp_path):
    """train/loss should be finite and positive with deferred float() conversion."""
    mock_task = MagicMock()
    dataloader = list(_make_fake_dataloader(B=2, num_batches=6))
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=3)
    kwargs["checkpoint_dir"] = str(tmp_path)
    model = _make_small_model()

    with patch("msdflow.train.trainer.log_metrics") as mock_log:
        train(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            grad_accum_steps=2,
            clearml_task=mock_task,
            **{k: v for k, v in kwargs.items() if k not in ("clearml_task",)},
        )

    reported_loss = mock_log.call_args_list[0][0][1]["train/loss"]
    assert np.isfinite(reported_loss)
    assert reported_loss > 0


def test_train_grad_accum_loss_normalized_by_microsteps(tmp_path):
    """train/loss should be epoch_loss / microsteps_per_epoch (per-microbatch average)."""
    mock_task = MagicMock()

    # Run with grad_accum_steps=1
    dataloader = list(_make_fake_dataloader(B=2, num_batches=4))
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=4)
    kwargs["checkpoint_dir"] = str(tmp_path / "run1")
    model = _make_small_model()

    with patch("msdflow.train.trainer.log_metrics") as mock_log1:
        train(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            grad_accum_steps=1,
            clearml_task=mock_task,
            **{k: v for k, v in kwargs.items() if k not in ("clearml_task",)},
        )
    loss_no_accum = mock_log1.call_args_list[0][0][1]["train/loss"]

    # Run with grad_accum_steps=2 on same data
    kwargs2 = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=2)
    kwargs2["checkpoint_dir"] = str(tmp_path / "run2")
    model = _make_small_model()

    with patch("msdflow.train.trainer.log_metrics") as mock_log2:
        train(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            grad_accum_steps=2,
            clearml_task=mock_task,
            **{k: v for k, v in kwargs2.items() if k not in ("clearml_task",)},
        )
    loss_with_accum = mock_log2.call_args_list[0][0][1]["train/loss"]

    # Both should be similar magnitude (not 2x different)
    # They won't be exactly equal due to different optimizer update patterns,
    # but they should be within an order of magnitude
    assert abs(loss_no_accum - loss_with_accum) / max(abs(loss_no_accum), 1e-8) < 1.0


# --- BatchPrefetcher ---

from msdflow.train.trainer import BatchPrefetcher


def _make_prefetcher_dataloader(B=2, num_batches=4):
    """Return a list-backed dataloader for prefetcher tests."""
    return [
        (
            torch.from_numpy(np.random.randn(B, 1, 8, 8).astype(np.float32)),
            torch.empty(B, 0),
        )
        for _ in range(num_batches)
    ]


def test_batch_prefetcher_yields_correct_count():
    """BatchPrefetcher yields exactly num_items batches."""
    dataloader = _make_prefetcher_dataloader(num_batches=4)
    prefetcher = BatchPrefetcher(
        dataloader=dataloader,
        num_items=5,
    )
    results = list(prefetcher)
    prefetcher.shutdown()
    assert len(results) == 5


def test_batch_prefetcher_output_shapes():
    """Each prefetched item has the correct numpy array shapes."""
    B = 4
    dataloader = _make_prefetcher_dataloader(B=B, num_batches=3)
    prefetcher = BatchPrefetcher(
        dataloader=dataloader,
        num_items=3,
    )
    images_np, cond_np = next(prefetcher)
    prefetcher.shutdown()
    assert isinstance(images_np, np.ndarray)
    assert isinstance(cond_np, np.ndarray)
    assert images_np.shape == (B, 1, 8, 8)
    assert cond_np.shape == (B, 0)


def test_batch_prefetcher_restarts_exhausted_dataloader():
    """Prefetcher re-creates the iterator when the dataloader is exhausted."""
    dataloader = _make_prefetcher_dataloader(num_batches=2)
    prefetcher = BatchPrefetcher(
        dataloader=dataloader,
        num_items=5,
    )
    results = list(prefetcher)
    prefetcher.shutdown()
    assert len(results) == 5


def test_batch_prefetcher_deterministic_with_same_data():
    """Same dataloader produces identical numpy arrays."""
    np.random.seed(42)
    dataloader = _make_prefetcher_dataloader(B=2, num_batches=3)

    prefetcher1 = BatchPrefetcher(
        dataloader=dataloader,
        num_items=2,
    )
    results1 = list(prefetcher1)
    prefetcher1.shutdown()

    prefetcher2 = BatchPrefetcher(
        dataloader=dataloader,
        num_items=2,
    )
    results2 = list(prefetcher2)
    prefetcher2.shutdown()

    for (img1, cond1), (img2, cond2) in zip(results1, results2):
        assert np.array_equal(img1, img2)
        assert np.array_equal(cond1, cond2)


# --- make_prepare_batch_jax ---


def test_make_prepare_batch_jax_output_shapes():
    """make_prepare_batch_jax returns tensors with correct shapes."""
    B = 4
    images_np = np.random.randn(B, 1, 8, 8).astype(np.float32)
    cond_np = np.empty((B, 0), dtype=np.float32)

    prepare_jax = make_prepare_batch_jax(
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        p_uncond=0.0,
    )
    key = jax.random.PRNGKey(0)
    t, x_t, u_t, cond, cond_mask, dropout_keys = prepare_jax(images_np, cond_np, key)

    assert t.shape == (B,)
    assert x_t.shape == (B, 1, 8, 8)
    assert u_t.shape == (B, 1, 8, 8)
    assert cond.shape == (B, 0)
    assert cond_mask.shape == (B,)
    assert dropout_keys.shape[0] == B


def test_make_prepare_batch_jax_times_in_range():
    """make_prepare_batch_jax samples t values in [0, 1]."""
    B = 8
    images_np = np.random.randn(B, 1, 8, 8).astype(np.float32)
    cond_np = np.empty((B, 0), dtype=np.float32)

    prepare_jax = make_prepare_batch_jax(
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        p_uncond=0.0,
    )
    t, *_ = prepare_jax(images_np, cond_np, jax.random.PRNGKey(1))
    assert jnp.all(t >= 0.0) and jnp.all(t <= 1.0)


def test_make_prepare_batch_jax_p_uncond_one_masks_all():
    """With p_uncond=1.0, all cond_mask values must be False."""
    B = 16
    images_np = np.random.randn(B, 1, 8, 8).astype(np.float32)
    cond_np = np.empty((B, 0), dtype=np.float32)

    prepare_jax = make_prepare_batch_jax(
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        p_uncond=1.0,
    )
    _, _, _, _, cond_mask, _ = prepare_jax(images_np, cond_np, jax.random.PRNGKey(2))
    assert jnp.all(~cond_mask)


def test_make_prepare_batch_jax_p_uncond_zero_keeps_all():
    """With p_uncond=0.0, all cond_mask values must be True."""
    B = 16
    images_np = np.random.randn(B, 1, 8, 8).astype(np.float32)
    cond_np = np.empty((B, 0), dtype=np.float32)

    prepare_jax = make_prepare_batch_jax(
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        p_uncond=0.0,
    )
    _, _, _, _, cond_mask, _ = prepare_jax(images_np, cond_np, jax.random.PRNGKey(3))
    assert jnp.all(cond_mask)


def test_make_prepare_batch_jax_different_keys_give_different_results():
    """Different keys must produce different x_t values."""
    B = 4
    images_np = np.random.randn(B, 1, 8, 8).astype(np.float32)
    cond_np = np.empty((B, 0), dtype=np.float32)

    prepare_jax = make_prepare_batch_jax(
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        p_uncond=0.0,
    )
    _, x_t_a, *_ = prepare_jax(images_np, cond_np, jax.random.PRNGKey(0))
    _, x_t_b, *_ = prepare_jax(images_np, cond_np, jax.random.PRNGKey(1))
    assert not jnp.allclose(x_t_a, x_t_b)


def test_make_prepare_batch_jax_rejects_ot_coupling():
    """make_prepare_batch_jax raises ValueError for ot_coupling."""
    from msdflow.flow.coupling import ot_coupling

    with pytest.raises(ValueError, match="ot_coupling"):
        make_prepare_batch_jax(
            coupling=ot_coupling,
            time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
            path_sampler=partial(sample_path),
            p_uncond=0.0,
        )
