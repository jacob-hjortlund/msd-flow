# Design: Early Stopping and Best-Model Checkpointing

**Date:** 2026-03-30
**Branch:** `update_docs`

---

## Summary

Add early stopping and best-model checkpointing to the training loop, both driven by a single configurable validation metric. Also includes a full docs/tests audit and removal of the unused `skip_download` config key.

---

## New Parameters

Three parameters are added to `msdflow.train.trainer.train()`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `monitor` | `str` | `"flow_matching_loss"` | Bare metric name (no `val/` prefix). Looked up in `val_metrics` first, then `epoch_metric_results`. |
| `monitor_mode` | `str` | `"min"` | `"min"` or `"max"` — whether lower or higher values are better. |
| `early_stopping_patience` | `int \| None` | `None` | Number of consecutive validation cycles without improvement before stopping. `None` disables early stopping. Best-model checkpointing is always active. |

**Metric lookup:** `val_metrics` is checked first; if not found, `epoch_metric_results` is checked. A `ValueError` is raised on the first validation run if the metric is absent from both dicts.

**`monitor_mode` validation:** A `ValueError` is raised at the start of `train()` (before the epoch loop) if `monitor_mode` is not `"min"` or `"max"`.

---

## In-Loop State

Three local variables are initialised before the epoch loop:

```python
best_metric_value = float("inf") if monitor_mode == "min" else float("-inf")
patience_counter = 0
best_epoch = None
```

---

## Per-Validation-Cycle Logic

Runs inside the existing `if (epoch + 1) % val_every == 0:` block, after all metrics are computed:

1. **Lookup** — retrieve `current` value of `monitor` from `val_metrics` or `epoch_metric_results`. Raise `ValueError` if not found.
2. **Improvement check** — `current < best_metric_value` (min mode) or `current > best_metric_value` (max mode).
3. **If improved:**
   - Capture `prev_value = best_metric_value` and `prev_epoch = best_epoch` (before updating state).
   - Log: `New best model at epoch {epoch+1}: {monitor} = {current:.4g} (previous best: {prev_value:.4g} at epoch {prev_epoch})` — or `(first checkpoint)` if `prev_epoch is None`.
   - Save `model_epoch{epoch+1}_best_raw.eqx` and `model_epoch{epoch+1}_best_ema.eqx` to `checkpoint_dir` using `eqx.tree_serialise_leaves`.
   - Call `log_checkpoint` for ClearML tracking.
   - Update `best_metric_value = current`, `best_epoch = epoch + 1`, reset `patience_counter = 0`.
4. **If not improved:** increment `patience_counter`.
5. **Early stopping check** — if `early_stopping_patience is not None` and `patience_counter >= early_stopping_patience`: log a message and `break`.

---

## Checkpoint Naming Convention

Best-model checkpoints are saved alongside periodic checkpoints in `checkpoint_dir`:

```
checkpoints/
  model_epoch10_raw.eqx          # Periodic checkpoint (instantaneous weights)
  model_epoch10_ema.eqx          # Periodic checkpoint (EMA weights)
  model_epoch47_best_raw.eqx     # Best model (instantaneous weights)
  model_epoch47_best_ema.eqx     # Best model (EMA weights) — use for inference
```

A new pair of `_best_` files is written each time a new best is found. Old `_best_` files are not deleted — the epoch stamp makes it unambiguous which is current.

---

## Config Changes

`configs/train/train.yaml` — three new keys:

```yaml
monitor: flow_matching_loss       # Bare metric name to track
monitor_mode: min                 # "min" or "max"
early_stopping_patience: null     # null = disabled; set to an int to enable
```

`configs/data/dataset.yaml` — remove `skip_download` (unused, will be removed from code in a future PR).

---

## Documentation Updates

- `docs/training.md`: add subsection under "The Training Loop" covering best-model checkpointing (always on) and early stopping (opt-in), the log format, and the checkpoint naming convention. Remove CLI override example for `skip_download`.
- `docs/configuration.md`: add rows for `monitor`, `monitor_mode`, and `early_stopping_patience` in the `configs/train/train.yaml` table. Remove `skip_download` from the `configs/data/dataset.yaml` section.

---

## Tests

New tests in `tests/train/test_trainer.py`:

- Best checkpoint files are created when the metric improves.
- Best checkpoint is **not** saved when the metric does not improve.
- `patience_counter` triggers early stopping at the correct validation cycle count.
- `monitor_mode="max"` correctly identifies improvement.
- Unknown `monitor` name raises `ValueError` at the first val run.
- Log output contains the expected "new best" message with correct values.

---

## Docs and Tests Audit

Before implementing the new features, audit all existing docs and tests:

- Read all files in `docs/` and cross-reference against the current `train()` signature, config keys, and public API. Fix any params that are undocumented or docs that describe stale behaviour.
- Check `tests/train/test_trainer.py` against all public functions in `trainer.py`.
- Check `tests/train/test_metrics.py` against `metrics.py`.
- Fix any gaps found alongside the new feature work.
