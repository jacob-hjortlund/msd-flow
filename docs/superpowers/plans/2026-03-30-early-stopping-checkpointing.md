# Early Stopping and Best-Model Checkpointing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add early stopping and best-model checkpointing driven by a single validation metric, fix all existing test signature bugs found during audit, and keep docs fully current.

**Architecture:** Three flat parameters (`monitor`, `monitor_mode`, `early_stopping_patience`) are appended to `train()`. Pre-loop state (`best_metric_value`, `patience_counter`, `best_epoch`) is initialised before the epoch loop. A monitoring block runs at the end of each validation cycle: looks up the metric, saves best-model checkpoint files on improvement, and optionally breaks the epoch loop when patience is exceeded. All existing tests have a systematic bug (missing `key` arg on batch-metric and train-step calls, and `prepare_batch` unpack of 5 instead of 6 values) that are fixed as Task 1.

**Tech Stack:** JAX, Equinox, Optax, Hydra YAML, pytest

---

## File Structure

**Modified:**
- `msdflow/train/trainer.py` — new params, mode validation, pre-loop state, post-val monitoring block
- `configs/train/train.yaml` — 3 new keys
- `configs/data/dataset.yaml` — remove `skip_download`
- `docs/training.md` — new best-model and early-stopping subsections, remove `skip_download` override
- `docs/configuration.md` — new config rows, remove `skip_download` row
- `tests/train/test_trainer.py` — fix missing `key` args and `prepare_batch` unpack throughout, add new tests

---

### Task 1: Fix audit findings in existing tests

The following bugs exist throughout `tests/train/test_trainer.py`. They must be fixed before new tests are added so the baseline is clean.

**Findings:**
1. `train_step` requires 7 args `(state, x_t, u_t, t, cond, cond_mask, key)` — all test call-sites pass 6.
2. `batch_metric_step` requires 7 args — all test call-sites pass 6.
3. `flow_matching_loss` requires 7 args — test helper `constant_loss` and inline test metrics (`dummy_metric`, `my_metric`, `my_metric_copy`) are missing `key`.
4. `prepare_batch` returns 6 values — all test call-sites unpack 5.

**Files:** Modify `tests/train/test_trainer.py`

- [ ] **Step 1: Run existing test suite to confirm the failures**

```bash
pytest tests/train/ -v 2>&1 | head -80
```

Expected: failures in `test_make_train_step_*`, `test_train_step_*`, `test_make_batch_metric_step_*`, `test_prepare_batch_*`.

- [ ] **Step 2: Fix `constant_loss` (used in `test_make_train_step_dispatches_to_injected_loss_fn`)**

Find:
```python
    def constant_loss(model, x_t, u_t, t, cond, cond_mask):
        return jnp.array(42.0)
```
Replace with:
```python
    def constant_loss(model, x_t, u_t, t, cond, cond_mask, key):
        return jnp.array(42.0)
```

- [ ] **Step 3: Fix all `train_step(...)` calls missing `key`**

Every call-site of `train_step` in the test file passes 6 positional args. Add `jax.random.PRNGKey(0)` as the 7th argument. Affected functions:
- `test_make_train_step_dispatches_to_injected_loss_fn`
- `test_train_step_returns_updated_state_and_loss`
- `test_train_step_loss_is_finite`
- `test_train_step_updates_model_params`
- `test_train_step_with_cond`
- `test_train_step_with_cond_dropped`

For each, change:
```python
    new_state, loss = train_step(state, x_t, u_t, t, cond, cond_mask)
```
to:
```python
    new_state, loss = train_step(state, x_t, u_t, t, cond, cond_mask, jax.random.PRNGKey(0))
```

And in `test_make_train_step_dispatches_to_injected_loss_fn`:
```python
    _, loss = train_step(state, x_t, u_t, t, cond, cond_mask, jax.random.PRNGKey(0))
```

- [ ] **Step 4: Fix inline test metrics missing `key`**

Find and replace:
```python
    def dummy_metric(model, x_t, u_t, t, cond, cond_mask):
        return jnp.array(0.0)
```
```python
    def dummy_metric(model, x_t, u_t, t, cond, cond_mask, key):
        return jnp.array(0.0)
```

```python
    def my_metric(model, x_t, u_t, t, cond, cond_mask, key):
        return jnp.array(0.0)

    def my_metric_copy(model, x_t, u_t, t, cond, cond_mask, key):
        return jnp.array(1.0)
    my_metric_copy.__name__ = "my_metric"
```

- [ ] **Step 5: Fix all `batch_metric_step` call-sites missing `key`**

In `test_make_batch_metric_step_returns_dict_keyed_by_fn_name`, `test_make_batch_metric_step_values_are_scalar_jax_arrays`, and `test_make_batch_metric_step_multiple_metrics_all_keys_present`, change:
```python
    result = step(SMALL_MODEL, x_t, u_t, t, cond, cond_mask)
```
to:
```python
    result = step(SMALL_MODEL, x_t, u_t, t, cond, cond_mask, jax.random.PRNGKey(0))
```

- [ ] **Step 6: Fix all `prepare_batch` call-sites unpacking 5 values**

`prepare_batch` returns `(t, x_t, u_t, cond, cond_mask, dropout_keys)` (6 values). Fix all call-sites in `test_prepare_batch_output_shapes`, `test_prepare_batch_times_in_range`, `test_prepare_batch_p_uncond_one_masks_all`, `test_prepare_batch_p_uncond_zero_keeps_all`, and `test_prepare_batch_different_keys_give_different_results`.

Change:
```python
    t, x_t, u_t, cond, cond_mask = prepare_batch(...)
```
to:
```python
    t, x_t, u_t, cond, cond_mask, _ = prepare_batch(...)
```

For `test_prepare_batch_times_in_range` (uses only `t`):
```python
    t, _, _, _, _, _ = prepare_batch(...)
```

For `test_prepare_batch_p_uncond_one_masks_all` and `test_prepare_batch_p_uncond_zero_keeps_all` (uses only `cond_mask`):
```python
    _, _, _, _, cond_mask, _ = prepare_batch(...)
```

For `test_prepare_batch_different_keys_give_different_results` (uses only `x_t`):
```python
    _, x_t_a, _, _, _, _ = prepare_batch(key=jax.random.PRNGKey(0), **kwargs)
    _, x_t_b, _, _, _, _ = prepare_batch(key=jax.random.PRNGKey(1), **kwargs)
```

- [ ] **Step 7: Run full test suite to confirm all pass**

```bash
pytest tests/train/ -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add tests/train/test_trainer.py
git commit -m "fix: correct test signatures — add missing key args and prepare_batch 6-tuple unpack"
```

---

### Task 2: Remove `skip_download` from config and docs

**Files:** Modify `configs/data/dataset.yaml`, `docs/configuration.md`, `docs/training.md`

- [ ] **Step 1: Remove from `configs/data/dataset.yaml`**

Remove the line:
```yaml
skip_download: false
```

- [ ] **Step 2: Remove from `docs/configuration.md`**

In the `configs/data/dataset.yaml` section, remove:
```
skip_download: false    # Set to true to skip download and reuse existing data
```

- [ ] **Step 3: Remove from `docs/training.md` CLI overrides**

In the `## Common CLI Overrides` section, remove:
```bash
# Skip data download (reuse existing processed data)
python train_model.py data.dataset.skip_download=true
```

- [ ] **Step 4: Commit**

```bash
git add configs/data/dataset.yaml docs/configuration.md docs/training.md
git commit -m "remove: drop unused skip_download config key and references"
```

---

### Task 3: TDD — `monitor_mode` validation and new parameters

**Files:** Modify `tests/train/test_trainer.py`, `msdflow/train/trainer.py`

- [ ] **Step 1: Write failing test**

Append to `tests/train/test_trainer.py`:

```python
def test_train_invalid_monitor_mode_raises(tmp_path):
    """train() raises ValueError immediately if monitor_mode is not 'min' or 'max'."""
    dataloader = list(_make_fake_dataloader())
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    kwargs["monitor_mode"] = "diagonal"
    with pytest.raises(ValueError, match="monitor_mode"):
        train(
            model=SMALL_MODEL,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            **kwargs,
        )
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/train/test_trainer.py::test_train_invalid_monitor_mode_raises -v
```

Expected: FAIL with `TypeError: train() got an unexpected keyword argument 'monitor_mode'`.

- [ ] **Step 3: Add new parameters and mode validation to `train()`**

In `msdflow/train/trainer.py`, add three parameters at the end of the `train()` signature (after `samples_dir: str | None = None`):

```python
    monitor: str = "flow_matching_loss",
    monitor_mode: str = "min",
    early_stopping_patience: int | None = None,
```

Add the following validation block immediately after the existing `samples_dir` guard (after `raise ValueError("samples_dir must be provided...")`):

```python
    if monitor_mode not in ("min", "max"):
        raise ValueError(
            f"monitor_mode must be 'min' or 'max', got {monitor_mode!r}"
        )
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/train/test_trainer.py::test_train_invalid_monitor_mode_raises -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/train/ -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/train/test_trainer.py msdflow/train/trainer.py
git commit -m "feat: add monitor, monitor_mode, early_stopping_patience params to train()"
```

---

### Task 4: TDD — best-model checkpointing

**Files:** Modify `tests/train/test_trainer.py`, `msdflow/train/trainer.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/train/test_trainer.py`:

```python
def test_best_checkpoint_saved_on_first_val(tmp_path):
    """Best-model checkpoint (raw + ema) is created after the first validation epoch."""
    dataloader = list(_make_fake_dataloader())
    val_dataloader = _fake_val_dataloader()
    kwargs = _make_train_kwargs(num_epochs=1)
    kwargs["checkpoint_dir"] = str(tmp_path)
    train(
        model=SMALL_MODEL,
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
    train(
        model=SMALL_MODEL,
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
    train(
        model=SMALL_MODEL,
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
    with pytest.raises(ValueError, match="nonexistent_metric"):
        train(
            model=SMALL_MODEL,
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
    with caplog.at_level(logging.INFO, logger="msdflow.train.trainer"):
        train(
            model=SMALL_MODEL,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            **kwargs,
        )
    best_logs = [r.message for r in caplog.records if "New best model" in r.message]
    assert len(best_logs) == 1
    assert best_logs[0].startswith("New best model at epoch 1: flow_matching_loss =")
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/train/test_trainer.py -k "best_checkpoint or unknown_monitor" -v
```

Expected: all 5 tests FAIL.

- [ ] **Step 3: Add pre-loop state to `train()`**

In `msdflow/train/trainer.py`, add the following three lines immediately after the existing pre-loop initialisations (`val_metrics: dict = {}`, `train_metrics: dict = {}`, `epoch_metric_results: dict = {}`):

```python
    best_metric_value = float("inf") if monitor_mode == "min" else float("-inf")
    patience_counter = 0
    best_epoch = None
```

- [ ] **Step 4: Add monitoring block inside the validation block**

In `msdflow/train/trainer.py`, inside `if (epoch + 1) % val_every == 0:`, add the following block **after** `avg_val_time = total_val_time / val_runs`:

```python
        # --- Best-model checkpointing and early stopping ---
        current_monitor = val_metrics.get(monitor)
        if current_monitor is None:
            current_monitor = epoch_metric_results.get(monitor)
        if current_monitor is None:
            raise ValueError(
                f"monitor metric '{monitor}' not found in val_metrics "
                f"{list(val_metrics.keys())} or epoch_metric_results "
                f"{list(epoch_metric_results.keys())}"
            )
        current_monitor = float(current_monitor)

        is_improved = (
            current_monitor < best_metric_value
            if monitor_mode == "min"
            else current_monitor > best_metric_value
        )

        if is_improved:
            all_metrics = {monitor: current_monitor}
            all_metrics.update(
                {k: v for k, v in val_metrics.items() if k != monitor}
            )
            all_metrics.update(
                {k: v for k, v in epoch_metric_results.items() if k != monitor}
            )
            metric_str = " | ".join(
                f"{k} = {float(v):.4g}" for k, v in all_metrics.items()
            )
            logger.info(f"New best model at epoch {epoch + 1}: {metric_str}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_raw_path = os.path.join(
                checkpoint_dir, f"model_epoch{epoch + 1}_best_raw.eqx"
            )
            best_ema_path = os.path.join(
                checkpoint_dir, f"model_epoch{epoch + 1}_best_ema.eqx"
            )
            eqx.tree_serialise_leaves(best_raw_path, state.model)
            eqx.tree_serialise_leaves(best_ema_path, ema_model)
            log_checkpoint(clearml_task, best_ema_path, epoch + 1)
            best_metric_value = current_monitor
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
```

- [ ] **Step 5: Run to verify tests pass**

```bash
pytest tests/train/test_trainer.py -k "best_checkpoint or unknown_monitor" -v
```

Expected: all 5 tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/train/ -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/train/test_trainer.py msdflow/train/trainer.py
git commit -m "feat: add best-model checkpointing to training loop"
```

---

### Task 5: TDD — early stopping

**Files:** Modify `tests/train/test_trainer.py`, `msdflow/train/trainer.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/train/test_trainer.py`:

```python
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
    train(
        model=SMALL_MODEL,
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
    train(
        model=SMALL_MODEL,
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
    with caplog.at_level(logging.INFO, logger="msdflow.train.trainer"):
        train(
            model=SMALL_MODEL,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            **kwargs,
        )
    stop_logs = [r.message for r in caplog.records if "Early stopping" in r.message]
    assert len(stop_logs) == 1
    assert "constant_metric" in stop_logs[0]
    assert "1" in stop_logs[0]  # patience count in message
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/train/test_trainer.py -k "early_stopping" -v
```

Expected: `test_early_stopping_triggers_at_correct_cycle` FAIL (all 10 epochs run instead of 2).

- [ ] **Step 3: Add the early-stopping `break` to `train()`**

In `msdflow/train/trainer.py`, add the following immediately after `else: patience_counter += 1` (still inside `if (epoch + 1) % val_every == 0:`):

```python
        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch + 1}: '{monitor}' "
                f"did not improve for {early_stopping_patience} consecutive "
                f"validation cycles."
            )
            break
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/train/test_trainer.py -k "early_stopping" -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/train/ -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/train/test_trainer.py msdflow/train/trainer.py
git commit -m "feat: add early stopping to training loop"
```

---

### Task 6: Update training config

**Files:** Modify `configs/train/train.yaml`

- [ ] **Step 1: Add new keys**

Append to `configs/train/train.yaml`:

```yaml
monitor: flow_matching_loss       # Bare metric name to track (no val/ prefix)
monitor_mode: min                 # "min" (lower is better) or "max" (higher is better)
early_stopping_patience: null     # null = disabled; positive int enables early stopping
```

- [ ] **Step 2: Commit**

```bash
git add configs/train/train.yaml
git commit -m "config: add monitor, monitor_mode, early_stopping_patience to train config"
```

---

### Task 7: Update docs

**Files:** Modify `docs/training.md`, `docs/configuration.md`

- [ ] **Step 1: Update checkpointing section in `docs/training.md`**

Replace the existing `### Checkpointing (every \`checkpoint_every\` epochs)` block with:

```markdown
### Checkpointing (every `checkpoint_every` epochs)

Saves two `.eqx` files to `checkpoint_dir`:

```
checkpoints/
  model_epoch10_raw.eqx    # Instantaneous model weights
  model_epoch10_ema.eqx    # EMA model weights — use this for inference
```

### Best-Model Checkpointing (every `val_every` epochs)

The training loop always tracks the best observed value of `monitor`. When a new best is found, two additional files are saved alongside the periodic checkpoints:

```
checkpoints/
  model_epoch47_best_raw.eqx   # Best instantaneous weights
  model_epoch47_best_ema.eqx   # Best EMA weights — use this for inference
```

The epoch stamp makes it unambiguous which checkpoint is the current best. Old best-checkpoint files are not deleted when a new best is found.

A log line is emitted each time a new best is found, with the monitored metric listed first:

```
New best model at epoch 47: flow_matching_loss = 0.0312 | other_metric = 0.1234
```

### Early Stopping (optional)

Set `train.early_stopping_patience` to a positive integer to stop training when the monitored metric fails to improve for that many consecutive **validation cycles** (each validation cycle runs every `val_every` epochs).

Example: `val_every=5` and `early_stopping_patience=10` halts training after 50 epochs without improvement.

Disabled by default (`early_stopping_patience: null`).
```

- [ ] **Step 2: Add CLI override examples to `docs/training.md`**

In the `## Common CLI Overrides` section, add:

```bash
# Enable early stopping after 20 validation cycles without improvement
python train_model.py train.early_stopping_patience=20

# Monitor a custom metric in max mode (higher is better)
python train_model.py train.monitor=my_metric train.monitor_mode=max
```

- [ ] **Step 3: Update `configs/train/train.yaml` section in `docs/configuration.md`**

In the `### configs/train/train.yaml` code block, add after `ema_decay: 0.9999`:

```yaml
monitor: flow_matching_loss         # Bare metric name to monitor (looked up in val metrics first,
                                    # then epoch metrics)
monitor_mode: min                   # "min" (lower is better) or "max" (higher is better)
early_stopping_patience: null       # null = disabled; set to a positive int to enable early stopping
                                    # (counts validation cycles, not epochs)
```

- [ ] **Step 4: Verify full test suite still passes**

```bash
pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/training.md docs/configuration.md
git commit -m "docs: document early stopping and best-model checkpointing"
```
