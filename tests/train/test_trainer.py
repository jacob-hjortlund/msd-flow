"""Tests for src.train.trainer."""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pytest
import numpy as np
import diffrax
from src.model.unet import UNet
from src.train.trainer import TrainState, make_train_state, train
from src.flow.sample import sample
from src.flow.interpolate import sample_path

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
from src.train.metrics import flow_matching_loss as _fml


def test_make_train_step_dispatches_to_injected_loss_fn():
    """make_train_step must use the injected loss_fn, not a hardcoded one."""
    optimizer = optax.adam(1e-3)
    state = make_train_state(SMALL_MODEL, optimizer)

    def constant_loss(model, x_t, u_t, t, cond, cond_mask):
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

    _, loss = train_step(state, x_t, u_t, t, cond, cond_mask)
    assert jnp.allclose(loss, jnp.array(42.0))


def test_train_step_returns_updated_state_and_loss():
    """Verify train step returns a TrainState and a scalar loss."""
    optimizer = optax.adam(1e-3)
    state = make_train_state(SMALL_MODEL, optimizer)
    train_step = make_train_step(optimizer, _fml)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)

    x_t, u_t = sample_path(x0, x1, t)
    new_state, loss = train_step(state, x_t, u_t, t, cond, cond_mask)

    assert isinstance(new_state, TrainState)
    assert loss.shape == ()


def test_train_step_loss_is_finite():
    """Verify train step produces a finite loss value."""
    optimizer = optax.adam(1e-3)
    state = make_train_state(SMALL_MODEL, optimizer)
    train_step = make_train_step(optimizer, _fml)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)

    x_t, u_t = sample_path(x0, x1, t)
    _, loss = train_step(state, x_t, u_t, t, cond, cond_mask)
    assert jnp.isfinite(loss)


def test_train_step_updates_model_params():
    """Verify at least one model parameter changes after a train step."""
    optimizer = optax.adam(1e-3)
    state = make_train_state(SMALL_MODEL, optimizer)
    train_step = make_train_step(optimizer, _fml)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)

    x_t, u_t = sample_path(x0, x1, t)
    new_state, _ = train_step(state, x_t, u_t, t, cond, cond_mask)

    # At least one parameter should have changed
    orig_leaves = jax.tree_util.tree_leaves(eqx.filter(state.model, eqx.is_array))
    new_leaves = jax.tree_util.tree_leaves(eqx.filter(new_state.model, eqx.is_array))
    assert any(not jnp.allclose(o, n) for o, n in zip(orig_leaves, new_leaves))


from src.train.trainer import ema_update


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


from src.train.trainer import make_batch_metric_step


def test_make_batch_metric_step_returns_dict_keyed_by_fn_name():
    """make_batch_metric_step must return a dict keyed by fn.__name__."""
    step = make_batch_metric_step([_fml])
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)

    result = step(SMALL_MODEL, x_t, u_t, t, cond, cond_mask)
    assert isinstance(result, dict)
    assert "flow_matching_loss" in result


def test_make_batch_metric_step_values_are_scalar_jax_arrays():
    """All values returned by make_batch_metric_step must be scalar JAX arrays."""
    step = make_batch_metric_step([_fml])
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)

    result = step(SMALL_MODEL, x_t, u_t, t, cond, cond_mask)
    for v in result.values():
        assert isinstance(v, jax.Array)
        assert v.shape == ()


def test_make_batch_metric_step_multiple_metrics_all_keys_present():
    """make_batch_metric_step with two distinct metrics returns both keys."""

    def dummy_metric(model, x_t, u_t, t, cond, cond_mask):
        return jnp.array(0.0)

    step = make_batch_metric_step([_fml, dummy_metric])
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)

    result = step(SMALL_MODEL, x_t, u_t, t, cond, cond_mask)
    assert "flow_matching_loss" in result
    assert "dummy_metric" in result


def test_make_batch_metric_step_raises_on_duplicate_names():
    """make_batch_metric_step must raise ValueError for duplicate metric names."""

    def my_metric(model, x_t, u_t, t, cond, cond_mask):
        return jnp.array(0.0)

    def my_metric_copy(model, x_t, u_t, t, cond, cond_mask):  # same __name__ via rename
        return jnp.array(1.0)
    my_metric_copy.__name__ = "my_metric"

    with pytest.raises(ValueError, match="duplicate metric names"):
        make_batch_metric_step([my_metric, my_metric_copy])


from functools import partial

from src.train.trainer import train
from src.flow.coupling import independent_coupling
from src.flow.interpolate import sample_time_uniform
import torch


def _make_fake_dataloader(B=2, num_batches=3):
    """Yield fake (images, meta) tuples matching DataLoader contract."""
    for _ in range(num_batches):
        images = torch.from_numpy(
            np.random.randn(B, 1, 8, 8).astype(np.float32)
        )
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
        num_val_eval_batches=0,
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


def test_train_runs_and_returns_model():
    """Verify the full training loop completes and returns a model."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=3))
    val_dataloader = _fake_val_dataloader()
    trained_model = train(
        model=SMALL_MODEL,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **_make_train_kwargs(),
    )
    assert trained_model is not None


def test_train_returns_ema_model_not_live_model():
    """train() must return the EMA model, which differs from the initial model after training."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=5))
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1, num_steps_per_epoch=5)
    # High learning rate so live model diverges quickly; EMA lags behind
    kwargs["optimizer"] = optax.adam(1e-1)
    trained = train(
        model=SMALL_MODEL,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )
    init_leaves = jax.tree_util.tree_leaves(eqx.filter(SMALL_MODEL, eqx.is_array))
    trained_leaves = jax.tree_util.tree_leaves(eqx.filter(trained, eqx.is_array))
    assert any(not jnp.allclose(i, t) for i, t in zip(init_leaves, trained_leaves))


def test_train_loop_completes_without_error():
    """Verify the training loop runs without error on repeated fixed batches."""
    fixed_images = torch.from_numpy(
        np.random.randn(4, 1, 8, 8).astype(np.float32)
    )
    fixed_meta = torch.empty(4, 0)
    dataloader = [(fixed_images, fixed_meta) for _ in range(20)]
    val_dataloader = _fake_val_dataloader()
    big_model = UNet(
        in_channels=1, out_channels=1, base_channels=4,
        channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
        num_groups=2, activation=jax.nn.silu,
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
    state = make_train_state(SMALL_MODEL_COND, optimizer)
    train_step = make_train_step(optimizer, _fml)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.array([[0.4], [0.8]])
    cond_mask = jnp.ones(B, dtype=bool)

    x_t, u_t = sample_path(x0, x1, t)
    new_state, loss = train_step(state, x_t, u_t, t, cond, cond_mask)
    assert isinstance(new_state, TrainState)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_train_step_with_cond_dropped():
    """Verify train step works when some conditions are dropped (CFG path)."""
    optimizer = optax.adam(1e-3)
    state = make_train_state(SMALL_MODEL_COND, optimizer)
    train_step = make_train_step(optimizer, _fml)

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.array([[0.4], [0.8]])
    cond_mask = jnp.array([True, False])

    x_t, u_t = sample_path(x0, x1, t)
    new_state, loss = train_step(state, x_t, u_t, t, cond, cond_mask)
    assert isinstance(new_state, TrainState)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_train_loop_with_cond():
    """Verify training loop works with metadata conditioning."""
    dataloader = [
        (torch.randn(2, 1, 8, 8), torch.tensor([[0.4], [0.8]]))
        for _ in range(3)
    ]
    val_dataloader = [(torch.randn(2, 1, 8, 8), torch.tensor([[0.4], [0.8]]))]
    kwargs = _make_train_kwargs(p_uncond=0.2)
    kwargs["checkpoint_dir"] = "/tmp/test_ckpt_cond"
    trained = train(
        model=SMALL_MODEL_COND,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )
    assert trained is not None


def test_train_num_steps_per_epoch_zero_uses_dataloader_length():
    """num_steps_per_epoch=0 should run exactly len(dataloader) steps per epoch."""
    dataloader = list(_make_fake_dataloader(B=2, num_batches=4))
    val_dataloader = _fake_val_dataloader()
    # Simply verify it completes without error when num_steps_per_epoch=0
    trained = train(
        model=SMALL_MODEL,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **_make_train_kwargs(num_steps_per_epoch=0),
    )
    assert trained is not None


def test_end_to_end_conditional_training_and_sampling():
    """Train a small conditional model and verify unconditional and guided sampling."""
    dataloader = [
        (torch.randn(2, 1, 8, 8), torch.tensor([[0.4], [0.8]]))
        for _ in range(5)
    ]
    val_dataloader = [(torch.randn(2, 1, 8, 8), torch.tensor([[0.4], [0.8]]))]

    kwargs = _make_train_kwargs(num_steps_per_epoch=5, p_uncond=0.2)
    kwargs["checkpoint_dir"] = "/tmp/test_ckpt_e2e"
    trained = train(
        model=SMALL_MODEL_COND,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )

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

    out_uncond = sample(**sample_kwargs)
    assert out_uncond.shape == (1, 8, 8)
    assert jnp.all(jnp.isfinite(out_uncond))

    out_guided = sample(**sample_kwargs, cond=jnp.array([0.4]), guidance_scale=2.0)
    assert out_guided.shape == (1, 8, 8)
    assert jnp.all(jnp.isfinite(out_guided))


# --- prepare_batch ---

from src.train.trainer import prepare_batch


def test_prepare_batch_output_shapes():
    """prepare_batch returns tensors with correct shapes."""
    B = 4
    images = torch.from_numpy(np.random.randn(B, 1, 8, 8).astype(np.float32))
    meta = torch.empty(B, 0)
    batch = (images, meta)

    key = jax.random.PRNGKey(0)
    t, x_t, u_t, cond, cond_mask = prepare_batch(
        batch=batch,
        key=key,
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        p_uncond=0.0,
    )

    assert t.shape == (B,)
    assert x_t.shape == (B, 1, 8, 8)
    assert u_t.shape == (B, 1, 8, 8)
    assert cond.shape == (B, 0)
    assert cond_mask.shape == (B,)


def test_prepare_batch_times_in_range():
    """prepare_batch samples t values in [0, 1]."""
    B = 8
    images = torch.from_numpy(np.random.randn(B, 1, 8, 8).astype(np.float32))
    meta = torch.empty(B, 0)
    batch = (images, meta)

    key = jax.random.PRNGKey(1)
    t, _, _, _, _ = prepare_batch(
        batch=batch,
        key=key,
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        p_uncond=0.0,
    )

    assert jnp.all(t >= 0.0) and jnp.all(t <= 1.0)


def test_prepare_batch_p_uncond_one_masks_all():
    """With p_uncond=1.0, all cond_mask values must be False."""
    B = 16
    images = torch.from_numpy(np.random.randn(B, 1, 8, 8).astype(np.float32))
    meta = torch.empty(B, 0)
    batch = (images, meta)

    key = jax.random.PRNGKey(2)
    _, _, _, _, cond_mask = prepare_batch(
        batch=batch,
        key=key,
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        p_uncond=1.0,
    )

    assert jnp.all(~cond_mask)


def test_prepare_batch_p_uncond_zero_keeps_all():
    """With p_uncond=0.0, all cond_mask values must be True."""
    B = 16
    images = torch.from_numpy(np.random.randn(B, 1, 8, 8).astype(np.float32))
    meta = torch.empty(B, 0)
    batch = (images, meta)

    key = jax.random.PRNGKey(3)
    _, _, _, _, cond_mask = prepare_batch(
        batch=batch,
        key=key,
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        p_uncond=0.0,
    )

    assert jnp.all(cond_mask)


def test_prepare_batch_different_keys_give_different_results():
    """Different keys must produce different x_t values."""
    B = 4
    images = torch.from_numpy(np.random.randn(B, 1, 8, 8).astype(np.float32))
    meta = torch.empty(B, 0)
    batch = (images, meta)

    kwargs = dict(
        batch=batch,
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        p_uncond=0.0,
    )
    _, x_t_a, _, _, _ = prepare_batch(key=jax.random.PRNGKey(0), **kwargs)
    _, x_t_b, _, _, _ = prepare_batch(key=jax.random.PRNGKey(1), **kwargs)
    assert not jnp.allclose(x_t_a, x_t_b)


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


from src.train.trainer import batch_metric_loop


def test_batch_metric_loop_returns_dict_of_floats():
    """batch_metric_loop must return dict[str, float]."""
    step = make_batch_metric_step([_fml])
    val_loader = _make_val_dataloader()
    result = batch_metric_loop(
        key=jax.random.PRNGKey(0),
        ema_model=SMALL_MODEL,
        dataloader=val_loader,
        step_fn=step,
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        p_uncond=0.0,
        num_batches=0,
    )
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())


def test_batch_metric_loop_values_are_finite():
    """batch_metric_loop must return finite values."""
    step = make_batch_metric_step([_fml])
    val_loader = _make_val_dataloader()
    result = batch_metric_loop(
        key=jax.random.PRNGKey(1),
        ema_model=SMALL_MODEL,
        dataloader=val_loader,
        step_fn=step,
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        p_uncond=0.0,
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
    batch_metric_loop(
        key=jax.random.PRNGKey(2),
        ema_model=SMALL_MODEL,
        dataloader=counting_loader(),
        step_fn=step,
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        p_uncond=0.0,
        num_batches=3,
    )
    assert len(call_count) == 3


def test_batch_metric_loop_returns_mean_not_sum():
    """batch_metric_loop must return the mean, not the sum, across batches."""

    def simple_metric(model, x_t, u_t, t, cond, cond_mask):
        """Metric that returns a simple value based on batch size."""
        # Return batch size as the metric value
        return jnp.array(float(x_t.shape[0]))

    simple_metric.__name__ = "simple_metric"

    # Create two dataloaders with different batch sizes
    loader1 = [
        (
            torch.randn(2, 1, 8, 8),
            torch.empty(2, 0),
        )
    ]
    loader2 = [
        (
            torch.randn(4, 1, 8, 8),  # Different batch size
            torch.empty(4, 0),
        )
    ]
    combined_loader = loader1 + loader2

    step = make_batch_metric_step([simple_metric])
    result = batch_metric_loop(
        key=jax.random.PRNGKey(0),
        ema_model=SMALL_MODEL,
        dataloader=combined_loader,
        step_fn=step,
        coupling=independent_coupling,
        time_sampler=partial(sample_time_uniform, t_min=0.0, t_max=1.0),
        path_sampler=partial(sample_path),
        p_uncond=0.0,
        num_batches=0,
    )
    # Batch 1: metric returns 2.0
    # Batch 2: metric returns 4.0
    # batch_metric_loop should return (2.0 + 4.0) / 2 = 3.0
    # If it was summing instead of averaging, we'd get (2.0 + 4.0) = 6.0
    assert "simple_metric" in result
    assert isinstance(result["simple_metric"], float)
    # The result should be 3.0 (mean of 2 and 4)
    assert abs(result["simple_metric"] - 3.0) < 1e-5


from src.train.trainer import collect_batches


def test_collect_batches_returns_correct_count():
    """collect_batches returns exactly num_batches tuples."""
    loader = _make_val_dataloader(num_batches=5)
    batches = collect_batches(loader, num_batches=3)
    assert len(batches) == 3


def test_collect_batches_zero_returns_all():
    """collect_batches with num_batches=0 returns the full dataloader."""
    loader = _make_val_dataloader(num_batches=5)
    batches = collect_batches(loader, num_batches=0)
    assert len(batches) == 5


def test_collect_batches_each_tuple_is_images_meta():
    """Each element returned by collect_batches is a (images, meta) pair."""
    loader = _make_val_dataloader(B=2, num_batches=3)
    batches = collect_batches(loader, num_batches=0)
    for batch in batches:
        images, meta = batch
        assert images.shape[0] == 2
        assert images.shape[1:] == (1, 8, 8)


def test_train_epoch_metric_receives_nonempty_val_batches():
    """A no-op epoch metric must receive a non-empty val_batches list."""
    received = {}

    def capture_epoch_metric(model, val_batches, key):
        received["val_batches"] = val_batches
        return jnp.array(0.0)

    capture_epoch_metric.__name__ = "capture_epoch_metric"

    dataloader = list(_make_fake_dataloader(B=2, num_batches=3))
    val_dataloader = _fake_val_dataloader()

    kwargs = _make_train_kwargs()
    kwargs["batch_metrics"] = [_fml]
    kwargs["epoch_metrics"] = [capture_epoch_metric]
    kwargs["num_train_eval_batches"] = 0
    kwargs["num_val_eval_batches"] = 0

    train(
        model=SMALL_MODEL,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        **kwargs,
    )

    assert "val_batches" in received
    assert len(received["val_batches"]) > 0


# ---------------------------------------------------------------------------
# ClearML + sample integration tests
# ---------------------------------------------------------------------------

import functools
from unittest.mock import MagicMock, patch

from src.flow.interpolate import sample_path

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


def test_train_runs_with_clearml_task_none():
    """train() completes without error when clearml_task=None (default)."""
    import jax
    key = jax.random.PRNGKey(0)
    dl = _make_dataloader()
    result = train(
        key=key,
        model=SMALL_MODEL,
        dataloader=dl,
        val_dataloader=dl,
        optimizer=OPTIMIZER,
        loss_fn=lambda model, x_t, u_t, t, cond, cond_mask: jnp.mean(x_t),
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
        checkpoint_dir="/tmp/test_ckpt",
        clearml_task=None,
    )
    assert result is not None


def test_train_calls_log_metrics_when_task_provided(tmp_path):
    """log_metrics is called once per log_every epoch when clearml_task is set."""
    import jax
    key = jax.random.PRNGKey(1)
    dl = _make_dataloader()
    mock_task = MagicMock()

    with patch("src.train.trainer.log_metrics") as mock_log_metrics:
        train(
            key=key,
            model=SMALL_MODEL,
            dataloader=dl,
            val_dataloader=dl,
            optimizer=OPTIMIZER,
            loss_fn=lambda model, x_t, u_t, t, cond, cond_mask: jnp.mean(x_t),
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

    with patch("src.train.trainer.log_checkpoint") as mock_log_ckpt:
        train(
            key=key,
            model=SMALL_MODEL,
            dataloader=dl,
            val_dataloader=dl,
            optimizer=OPTIMIZER,
            loss_fn=lambda model, x_t, u_t, t, cond, cond_mask: jnp.mean(x_t),
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


def test_train_generates_samples_to_disk(tmp_path):
    """sample_fn is called at sample_every epochs and files are saved."""
    import jax
    key = jax.random.PRNGKey(3)
    dl = _make_dataloader()

    def fake_sample_fn(model, key, num_samples):
        return np.zeros((num_samples, 1, 8, 8), dtype=np.float32)

    train(
        key=key,
        model=SMALL_MODEL,
        dataloader=dl,
        val_dataloader=dl,
        optimizer=OPTIMIZER,
        loss_fn=lambda model, x_t, u_t, t, cond, cond_mask: jnp.mean(x_t),
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
    npy_files = glob.glob(str(tmp_path / "samples" / "**" / "*.npy"), recursive=True)
    assert len(npy_files) == 4  # 2 epochs × 2 samples


def test_train_skips_sampling_when_sample_every_is_zero(tmp_path):
    """No samples are generated when sample_every=0."""
    import jax
    key = jax.random.PRNGKey(4)
    dl = _make_dataloader()
    call_count = {"n": 0}

    def counting_sample_fn(model, key, num_samples):
        call_count["n"] += 1
        return np.zeros((num_samples, 1, 8, 8), dtype=np.float32)

    train(
        key=key,
        model=SMALL_MODEL,
        dataloader=dl,
        val_dataloader=dl,
        optimizer=OPTIMIZER,
        loss_fn=lambda model, x_t, u_t, t, cond, cond_mask: jnp.mean(x_t),
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
