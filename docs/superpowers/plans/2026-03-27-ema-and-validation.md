# EMA Weights and Validation Step — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `train()` in `trainer.py` with EMA weight tracking and a per-epoch validation pass using EMA weights, replacing the step-based loop with an epoch-based one.

**Architecture:** `ema_model` lives as a local variable in `train()` (not inside `TrainState`) to keep the JIT-compiled `train_step` kernel free of unused pass-through arrays. A new `ema_update()` function blends arrays via `eqx.partition`/`tree_map`/`eqx.combine`. A new `make_val_step()` returns a JIT-compiled validation step that calls `flow_matching_loss` without gradients.

**Tech Stack:** JAX, Equinox, Optax, OmegaConf (Hydra), PyTorch DataLoader, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/train/trainer.py` | Modify | Add `ema_update`, `make_val_step`, `_run_validation`; rewrite `train` loop |
| `configs/train/train.yaml` | Modify | Replace `num_steps` with epoch-based config keys |
| `tests/train/test_trainer.py` | Modify | Fix existing tests for new signature; add new unit/integration tests |

---

### Task 1: `ema_update` — tests then implementation

**Files:**
- Modify: `tests/train/test_trainer.py`
- Modify: `src/train/trainer.py`

- [ ] **Step 1: Write the failing tests**

Add the following tests to `tests/train/test_trainer.py`. Place them after the existing `make_train_step` tests (around line 116, before the `from src.train.trainer import train` import).

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/train/test_trainer.py::test_ema_update_decay_one_leaves_ema_unchanged \
       tests/train/test_trainer.py::test_ema_update_decay_zero_copies_new_model \
       tests/train/test_trainer.py::test_ema_update_blends_arrays_correctly -v
```

Expected: `ImportError` or `AttributeError` — `ema_update` does not exist yet.

- [ ] **Step 3: Implement `ema_update` in `src/train/trainer.py`**

Add after the `make_train_step` function (after line 74):

```python
def ema_update(ema_model, new_model, decay: float):
    """Update EMA model weights using exponential moving average.

    Non-array leaves (static model configuration) are carried over from
    ``ema_model`` unchanged.

    Args:
        ema_model: Current EMA model.
        new_model: Latest trained model whose weights are blended in.
        decay:     EMA decay rate. Typical value: 0.9999.

    Returns:
        Updated EMA model with blended array leaves.
    """
    ema_arrays, static = eqx.partition(ema_model, eqx.is_array)
    new_arrays, _ = eqx.partition(new_model, eqx.is_array)
    updated = jax.tree_util.tree_map(
        lambda e, m: decay * e + (1.0 - decay) * m,
        ema_arrays,
        new_arrays,
    )
    return eqx.combine(updated, static)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/train/test_trainer.py::test_ema_update_decay_one_leaves_ema_unchanged \
       tests/train/test_trainer.py::test_ema_update_decay_zero_copies_new_model \
       tests/train/test_trainer.py::test_ema_update_blends_arrays_correctly -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/train/trainer.py tests/train/test_trainer.py
git commit -m "feat: add ema_update function with tests"
```

---

### Task 2: `make_val_step` — tests then implementation

**Files:**
- Modify: `tests/train/test_trainer.py`
- Modify: `src/train/trainer.py`

- [ ] **Step 1: Write the failing tests**

Add after the `ema_update` tests (after `test_ema_update_blends_arrays_correctly`):

```python
from src.train.trainer import make_val_step


def test_val_step_returns_scalar_loss():
    """val_step must return a scalar JAX array."""
    val_step = make_val_step()

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)

    from src.flow.otfm import sample_path as _sp
    x_t, u_t = _sp(x0, x1, t)

    loss = val_step(SMALL_MODEL, x_t, u_t, t, cond, cond_mask)
    assert loss.shape == ()


def test_val_step_loss_is_finite():
    """val_step loss must be finite."""
    val_step = make_val_step()

    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)

    from src.flow.otfm import sample_path as _sp
    x_t, u_t = _sp(x0, x1, t)

    loss = val_step(SMALL_MODEL, x_t, u_t, t, cond, cond_mask)
    assert jnp.isfinite(loss)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/train/test_trainer.py::test_val_step_returns_scalar_loss \
       tests/train/test_trainer.py::test_val_step_loss_is_finite -v
```

Expected: `ImportError` — `make_val_step` does not exist yet.

- [ ] **Step 3: Implement `make_val_step` in `src/train/trainer.py`**

Add directly after `ema_update`:

```python
def make_val_step():
    """Return a JIT-compiled validation step.

    Computes flow matching loss without gradient. Mirrors the structure of
    ``make_train_step`` but does not update any state.

    Returns:
        A ``filter_jit``-compiled callable with signature
        ``(model, x_t, u_t, t, cond, cond_mask) -> scalar_loss``.
    """

    @eqx.filter_jit
    def val_step(
        model,
        x_t: jax.Array,
        u_t: jax.Array,
        t: jax.Array,
        cond: jax.Array,
        cond_mask: jax.Array,
    ) -> jax.Array:
        return flow_matching_loss(model, x_t, u_t, t, cond, cond_mask)

    return val_step
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/train/test_trainer.py::test_val_step_returns_scalar_loss \
       tests/train/test_trainer.py::test_val_step_loss_is_finite -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/train/trainer.py tests/train/test_trainer.py
git commit -m "feat: add make_val_step with tests"
```

---

### Task 3: Rewrite `train` loop and update all tests

**Files:**
- Modify: `src/train/trainer.py`
- Modify: `tests/train/test_trainer.py`
- Modify: `configs/train/train.yaml`

#### Step 1: Update the config

- [ ] **Step 1: Update `configs/train/train.yaml`**

Replace the entire file with:

```yaml
num_epochs: 100
num_steps_per_epoch: 0
batch_size: 16
checkpoint_dir: ${work_dir}/checkpoints
checkpoint_every: 5
log_every: 1
val_every: 1
ema_decay: 0.9999
optimizer:
  _target_: optax.adam
  learning_rate: 1.0e-4
p_uncond: 0.1
```

#### Step 2: Write the failing tests

- [ ] **Step 2: Replace all existing `train`-loop tests in `tests/train/test_trainer.py`**

The existing tests use the old `num_steps` config key and the old 4-argument `train(cfg, model, dataloader, optimizer)` signature. Replace everything from the `from src.train.trainer import train` import downward (line 118 to end of file) with the following:

```python
from src.train.trainer import train
from src.flow.coupling import ot_coupling
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


def _base_cfg(num_epochs=1, num_steps_per_epoch=3):
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "seed": 0,
        "train": {
            "num_epochs": num_epochs,
            "num_steps_per_epoch": num_steps_per_epoch,
            "log_every": 1,
            "checkpoint_every": 100,
            "checkpoint_dir": "/tmp/test_ckpt",
            "p_uncond": 0.0,
            "ema_decay": 0.9999,
            "val_every": 1,
        },
        "flow": {"otfm": {"t_min": 0.0, "t_max": 1.0}},
    })


def test_train_runs_and_returns_model():
    """Verify the full training loop completes and returns a model."""
    optimizer = optax.adam(1e-3)
    dataloader = list(_make_fake_dataloader(B=2, num_batches=3))
    val_dataloader = _fake_val_dataloader()
    trained_model = train(_base_cfg(), SMALL_MODEL, dataloader, val_dataloader, optimizer)
    assert trained_model is not None


def test_train_returns_ema_model_not_live_model():
    """train() must return the EMA model, which differs from the live model after training."""
    optimizer = optax.adam(1e-3)
    dataloader = list(_make_fake_dataloader(B=2, num_batches=5))
    val_dataloader = _fake_val_dataloader()

    # Use a high learning rate so the live model diverges quickly from EMA
    big_lr_optimizer = optax.adam(1e-1)
    trained = train(
        _base_cfg(num_epochs=1, num_steps_per_epoch=5),
        SMALL_MODEL,
        dataloader,
        val_dataloader,
        big_lr_optimizer,
    )
    # EMA weights are a blend — they must differ from initial model
    init_leaves = jax.tree_util.tree_leaves(eqx.filter(SMALL_MODEL, eqx.is_array))
    trained_leaves = jax.tree_util.tree_leaves(eqx.filter(trained, eqx.is_array))
    assert any(not jnp.allclose(i, t) for i, t in zip(init_leaves, trained_leaves))


def test_train_reduces_loss():
    """Verify the training loop runs without error on repeated fixed batches."""
    fixed_images = torch.from_numpy(
        np.random.randn(4, 1, 8, 8).astype(np.float32)
    )
    fixed_meta = torch.empty(4, 0)
    dataloader = [
        (fixed_images, fixed_meta) for _ in range(20)
    ]
    val_dataloader = _fake_val_dataloader()
    optimizer = optax.adam(1e-3)
    big_model = UNet(
        in_channels=1, out_channels=1, base_channels=4,
        channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
        num_groups=2, activation=jax.nn.silu,
        key=jax.random.PRNGKey(99),
    )
    train(_base_cfg(num_steps_per_epoch=20), big_model, dataloader, val_dataloader, optimizer)


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

    x_t, u_t = sample_path(x0, x1, t)
    new_state, loss = train_step(state, x_t, u_t, t, cond, cond_mask)
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
    cond_mask = jnp.array([True, False])

    x_t, u_t = sample_path(x0, x1, t)
    new_state, loss = train_step(state, x_t, u_t, t, cond, cond_mask)
    assert isinstance(new_state, TrainState)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_train_loop_with_cond():
    """Verify training loop works with metadata conditioning."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "seed": 0,
        "train": {
            "num_epochs": 1,
            "num_steps_per_epoch": 3,
            "log_every": 1,
            "checkpoint_every": 100,
            "checkpoint_dir": "/tmp/test_ckpt_cond",
            "p_uncond": 0.2,
            "ema_decay": 0.9999,
            "val_every": 1,
        },
        "flow": {"otfm": {"t_min": 0.0, "t_max": 1.0}},
    })

    dataloader = [
        (torch.randn(2, 1, 8, 8), torch.tensor([[0.4], [0.8]]))
        for _ in range(3)
    ]
    val_dataloader = [(torch.randn(2, 1, 8, 8), torch.tensor([[0.4], [0.8]]))]
    optimizer = optax.adam(1e-3)
    trained = train(cfg, SMALL_MODEL_COND, dataloader, val_dataloader, optimizer)
    assert trained is not None


def test_train_raises_on_unknown_time_sampling():
    """train() raises ValueError for an unrecognised time_sampling value."""
    from unittest.mock import MagicMock

    model = UNet(
        in_channels=1, out_channels=1, base_channels=4,
        channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
        num_groups=2, activation=jax.nn.silu, key=jax.random.PRNGKey(0),
    )
    optimizer = optax.adam(1e-3)

    images = torch.ones(2, 1, 4, 4)
    meta = torch.zeros(2, 0)
    dataloader = [(images, meta)]
    val_dataloader = [(images, meta)]

    cfg = MagicMock()
    cfg.seed = 0
    cfg.flow.otfm.t_min = 0.0
    cfg.flow.otfm.t_max = 1.0
    cfg.flow.otfm.get.side_effect = lambda key, default=None: {
        "sigma_0": 0.0,
        "sigma_1": 0.0,
        "time_sampling": "bad_value",
    }.get(key, default)
    cfg.train.num_epochs = 1
    cfg.train.num_steps_per_epoch = 1
    cfg.train.log_every = 100
    cfg.train.checkpoint_every = 100
    cfg.train.checkpoint_dir = "/tmp/ckpt_test"
    cfg.train.val_every = 100
    cfg.train.get.side_effect = lambda key, default=None: {
        "p_uncond": 0.0,
        "ema_decay": 0.9999,
    }.get(key, default)

    with pytest.raises(ValueError, match="time_sampling"):
        train(cfg, model, dataloader, val_dataloader, optimizer)


def test_end_to_end_conditional_training_and_sampling():
    """Train a small conditional model and verify unconditional and guided sampling."""
    import diffrax
    from omegaconf import OmegaConf
    from src.flow.sample import sample

    cfg = OmegaConf.create({
        "seed": 0,
        "train": {
            "num_epochs": 1,
            "num_steps_per_epoch": 5,
            "log_every": 1,
            "checkpoint_every": 100,
            "checkpoint_dir": "/tmp/test_ckpt_e2e",
            "p_uncond": 0.2,
            "ema_decay": 0.9999,
            "val_every": 1,
        },
        "flow": {"otfm": {"t_min": 0.0, "t_max": 1.0}},
    })

    dataloader = [
        (torch.randn(2, 1, 8, 8), torch.tensor([[0.4], [0.8]]))
        for _ in range(5)
    ]
    val_dataloader = [(torch.randn(2, 1, 8, 8), torch.tensor([[0.4], [0.8]]))]

    optimizer = optax.adam(1e-3)
    trained = train(cfg, SMALL_MODEL_COND, dataloader, val_dataloader, optimizer)

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
```

- [ ] **Step 3: Run the new tests to verify they fail**

```bash
pytest tests/train/test_trainer.py -v -k "train_runs or train_returns_ema or train_reduces or train_loop_with_cond or train_raises or end_to_end"
```

Expected: failures due to wrong `train` signature (extra `val_dataloader` arg) or `num_steps` KeyError.

#### Step 4: Implement the new `train` function

- [ ] **Step 4: Replace the `train` function in `src/train/trainer.py`**

Replace the entire `train` function (lines 77–163) with:

```python
def _run_validation(
    ema_model,
    val_dataloader,
    val_step,
    sample_path_fn,
    key: jax.Array,
    time_sampling: str,
    t_min: float,
    t_max: float,
    p_uncond: float,
) -> float:
    """Run a full pass over val_dataloader and return mean flow matching loss.

    Args:
        ema_model:       EMA model used for inference.
        val_dataloader:  Iterable of ``(images, meta)`` batches.
        val_step:        JIT-compiled val step from ``make_val_step()``.
        sample_path_fn:  JIT-compiled ``sample_path`` partial.
        key:             JAX PRNG key (consumed internally via splitting).
        time_sampling:   ``"uniform"`` or ``"logit_normal"``.
        t_min:           Lower time bound (uniform sampling only).
        t_max:           Upper time bound (uniform sampling only).
        p_uncond:        Probability of dropping the condition per sample.

    Returns:
        Mean validation loss over all batches.
    """
    total_loss = 0.0
    n_batches = 0

    for batch in val_dataloader:
        images, meta = batch
        x1_np = images.numpy()
        cond_np = meta.numpy()
        B = x1_np.shape[0]

        key, key_cpu, key_time, key_path = jax.random.split(key, 4)
        cpu_seed = int(jax.random.randint(key_cpu, shape=(), minval=0, maxval=2**31 - 1))
        rng = np.random.default_rng(cpu_seed)

        x0_np = rng.standard_normal(x1_np.shape).astype(np.float32)
        x0_paired = ot_coupling(x0_np, x1_np)

        if time_sampling == "uniform":
            t = sample_time_uniform(key_time, B, t_min, t_max)
        elif time_sampling == "logit_normal":
            t = sample_time_logit_normal(key_time, B)
        else:
            raise ValueError(
                f"Unknown time_sampling={time_sampling!r}; "
                "choose 'uniform' or 'logit_normal'."
            )

        cond_mask_np = (rng.random(B) >= p_uncond).astype(bool)

        x_t, u_t = sample_path_fn(
            jnp.array(x0_paired), jnp.array(x1_np), t, key=key_path
        )
        cond = jnp.array(cond_np)
        cond_mask = jnp.array(cond_mask_np)

        total_loss += float(val_step(ema_model, x_t, u_t, t, cond, cond_mask))
        n_batches += 1

    return total_loss / n_batches


def train(cfg, model, dataloader, val_dataloader, optimizer: optax.GradientTransformation):
    """Main training loop with EMA and periodic validation.

    Args:
        cfg:            Hydra DictConfig with cfg.seed, cfg.train.*, cfg.flow.otfm.*
        model:          Velocity-field network to train.
        dataloader:     PyTorch DataLoader yielding ``(images, meta)`` tuples
                        where images is ``(B, C, H, W)`` and meta is
                        ``(B, cond_dim)`` or ``(B, 0)`` if unconditional.
        val_dataloader: DataLoader for the validation split, same format as
                        ``dataloader``. Used for periodic EMA model evaluation.
        optimizer:      Optax GradientTransformation (construct via
                        hydra.utils.instantiate(cfg.train.optimizer) before calling).

    Returns:
        Trained EMA model.
    """
    state = make_train_state(model, optimizer)
    train_step = make_train_step(optimizer)
    val_step = make_val_step()
    key = jax.random.PRNGKey(cfg.seed)

    t_min = float(cfg.flow.otfm.t_min)
    t_max = float(cfg.flow.otfm.t_max)
    sigma_0 = float(cfg.flow.otfm.get("sigma_0", 0.0))
    sigma_1 = float(cfg.flow.otfm.get("sigma_1", 0.0))
    time_sampling = cfg.flow.otfm.get("time_sampling", "uniform")
    num_epochs = int(cfg.train.num_epochs)
    num_steps_per_epoch = int(cfg.train.num_steps_per_epoch)
    log_every = int(cfg.train.log_every)
    ckpt_every = int(cfg.train.checkpoint_every)
    ckpt_dir = cfg.train.checkpoint_dir
    p_uncond = float(cfg.train.get("p_uncond", 0.0))
    ema_decay = float(cfg.train.get("ema_decay", 0.9999))
    val_every = int(cfg.train.val_every)

    steps_per_epoch = (
        len(dataloader) if num_steps_per_epoch == 0 else num_steps_per_epoch
    )

    _sample_path = jax.jit(
        functools.partial(sample_path, sigma_0=sigma_0, sigma_1=sigma_1)
    )

    ema_model = model
    data_iter = iter(dataloader)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            images, meta = batch
            x1_np = images.numpy()
            cond_np = meta.numpy()
            B = x1_np.shape[0]

            key, key_cpu, key_time, key_path = jax.random.split(key, 4)
            cpu_seed = int(
                jax.random.randint(key_cpu, shape=(), minval=0, maxval=2**31 - 1)
            )
            rng = np.random.default_rng(cpu_seed)

            x0_np = rng.standard_normal(x1_np.shape).astype(np.float32)
            x0_paired = ot_coupling(x0_np, x1_np)

            if time_sampling == "uniform":
                t = sample_time_uniform(key_time, B, t_min, t_max)
            elif time_sampling == "logit_normal":
                t = sample_time_logit_normal(key_time, B)
            else:
                raise ValueError(
                    f"Unknown time_sampling={time_sampling!r}; "
                    "choose 'uniform' or 'logit_normal'."
                )

            cond_mask_np = (rng.random(B) >= p_uncond).astype(bool)

            x_t, u_t = _sample_path(
                jnp.array(x0_paired), jnp.array(x1_np), t, key=key_path
            )
            cond = jnp.array(cond_np)
            cond_mask = jnp.array(cond_mask_np)

            state, loss = train_step(state, x_t, u_t, t, cond, cond_mask)
            ema_model = ema_update(ema_model, state.model, ema_decay)
            epoch_loss += float(loss)

        if (epoch + 1) % log_every == 0:
            logger.info(
                f"epoch={epoch + 1}  loss={epoch_loss / steps_per_epoch:.6f}"
            )

        if (epoch + 1) % val_every == 0:
            key, key_val = jax.random.split(key)
            val_loss = _run_validation(
                ema_model,
                val_dataloader,
                val_step,
                _sample_path,
                key_val,
                time_sampling,
                t_min,
                t_max,
                p_uncond,
            )
            logger.info(f"epoch={epoch + 1}  val_loss={val_loss:.6f}")

        if (epoch + 1) % ckpt_every == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            path = os.path.join(ckpt_dir, f"model_epoch{epoch + 1}.eqx")
            eqx.tree_serialise_leaves(path, ema_model)
            logger.info(f"Saved checkpoint: {path}")

    return ema_model
```

- [ ] **Step 5: Run all trainer tests**

```bash
pytest tests/train/test_trainer.py -v
```

Expected: all tests PASSED.

- [ ] **Step 6: Run the full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all tests PASSED.

- [ ] **Step 7: Commit**

```bash
git add src/train/trainer.py tests/train/test_trainer.py configs/train/train.yaml
git commit -m "feat: epoch-based training loop with EMA and validation"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|-----------------|------|
| EMA over steps of model weights | Task 1 (`ema_update`) + Task 3 (loop) |
| EMA model separate from `TrainState` (Approach B) | Task 3 (`ema_model` local var) |
| Validation every `val_every` epochs using EMA weights | Task 3 (`_run_validation`) |
| `make_val_step` mirrors `make_train_step` | Task 2 |
| Epoch-based outer loop, step-based inner loop | Task 3 |
| `num_steps_per_epoch=0` → use `len(dataloader)` | Task 3 |
| `ema_decay` configurable, default 0.9999 | Task 3 + config |
| Checkpoint saves `ema_model` (not `state.model`) | Task 3 |
| Returns `ema_model` | Task 3 |
| Config: remove `num_steps`, add new keys | Task 3 |
| Unit test `ema_update` | Task 1 |
| Unit test `make_val_step` | Task 2 |
| Integration test `train` with val | Task 3 |

All requirements covered. No gaps found.
