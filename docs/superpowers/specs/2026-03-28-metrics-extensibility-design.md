# Metrics Extensibility Design

**Date:** 2026-03-28
**Branch:** feature/main_entry_point
**Status:** Approved

---

## Overview

Refactor `src/train/trainer.py`, `configs/train/train.yaml`, and `src/train/metrics.py`
so that new metrics (e.g. FID) can be added by dropping in a callable and updating
config, with no changes to trainer internals.

The current code hardcodes `flow_matching_loss` in `make_train_step` and
`make_val_step`/`validation_loop`. This design makes all three components
configurable via Hydra `_target_` instantiation, consistent with the rest of
the codebase.

---

## Metric Tiers

Three tiers of metrics, each with a distinct signature and evaluation cadence:

| Tier | Config key | Signature | When evaluated |
|---|---|---|---|
| Training objective | `loss_fn` | `(model, x_t, u_t, t, cond, cond_mask) → scalar` | Every train step (gradient signal) |
| Batch metrics | `batch_metrics` | `(model, x_t, u_t, t, cond, cond_mask) → scalar` | Every `val_every` epochs, over val loader + N train batches |
| Epoch metrics | `epoch_metrics` | `(model, val_batches, key) → scalar` | Every `val_every` epochs, on M collected val batches |

- `loss_fn` is the only differentiable objective; it drives gradient computation
  and is separate from logged metrics even if they share the same function.
- `batch_metrics` stream through their dataloaders per-batch — no batches are
  held in memory.
- `epoch_metrics` receive a pre-collected list of raw `(images, meta)` tuples.
  Any extra dependencies (solver, n_samples, feature extractor path, etc.) are
  baked in via Hydra `_partial_: true`. Intended for generation-based metrics
  such as FID and for early stopping signals.
- Metric names for logging are derived from `fn.__name__`.

---

## Component Changes

### `src/train/metrics.py`

No changes to existing functions. A module-level comment documents the two
expected metric signatures to guide future development. New metrics (FID, etc.)
are added here as standalone callables when implemented.

### `src/train/trainer.py`

**`make_train_step(optimizer, loss_fn)`**
- Adds `loss_fn` parameter; closes over it instead of hardcoding
  `flow_matching_loss`. No other changes.

**`make_val_step` → `make_batch_metric_step(batch_metrics: list[callable])`**
- Renamed. Closes over a list of metric callables.
- Returns `dict[str, jax.Array]` keyed by `fn.__name__` instead of a scalar.
- Still `filter_jit`-compiled.

**`validation_loop` → `batch_metric_loop(..., num_batches: int)`**
- Renamed. Adds `num_batches` parameter (0 = full loader).
- Returns `dict[str, float]` (per-metric mean over batches) instead of `float`.

**`collect_batches(dataloader, num_batches: int) → list`**
- New helper. Collects at most `num_batches` raw `(images, meta)` tuples from a
  dataloader. `num_batches=0` collects all. Used to supply `val_batches` to
  epoch metrics without holding the full val set in memory.

**`train()` — new parameters**
- `loss_fn: callable`
- `batch_metrics: list[callable]`
- `epoch_metrics: list[callable]`
- `num_train_eval_batches: int` — batches drawn from train loader for batch
  metrics (0 = all).
- `num_val_eval_batches: int` — batches collected for epoch metrics (0 = all).

**`train()` — evaluation cadence at each `val_every` epoch**
1. Stream full val_dataloader → `batch_metric_loop` → `val_metrics: dict`
2. Draw `num_train_eval_batches` from train_dataloader → `batch_metric_loop` →
   `train_metrics: dict`
3. `collect_batches(val_dataloader, num_val_eval_batches)` → `val_batches`
   *(second pass over val_dataloader; intentional — batch_metrics stream without
   storing batches so they cannot be reused for epoch_metrics)*
4. For each fn in `epoch_metrics`: `fn(ema_model, val_batches, key)` →
   `epoch_metric_results: dict`
5. Log unified dict with `val/`, `train/`, and `epoch/` prefixed keys.

### `configs/train/train.yaml`

```yaml
_target_: src.train.trainer.train
_partial_: true
loss_fn:
  _target_: src.train.metrics.flow_matching_loss
batch_metrics:
  - _target_: src.train.metrics.flow_matching_loss
epoch_metrics: []
num_train_eval_batches: 0
num_val_eval_batches: 0
optimizer:
  _target_: optax.adamw
  learning_rate: 1.0e-4
coupling:
  _target_: src.flow.independent_coupling
  _partial_: true
time_sampler:
  _target_: src.flow.sample_time_uniform
  _partial_: true
  t_min: 0.0
  t_max: 1.0
path_sampler:
  _target_: src.flow.sample_path
  _partial_: true
  sigma_0: 0.0
  sigma_1: 0.0
num_epochs: 100
num_steps_per_epoch: 0
p_uncond: "${if_cond: ${data.dataset.metadata_columns}, 0.1, 1.0}"
checkpoint_dir: ${work_dir}/checkpoints
checkpoint_every: 5
log_every: 1
val_every: 1
ema_decay: 0.9999
```

`flow_matching_loss` appears in both `loss_fn` and `batch_metrics` by default:
`loss_fn` drives gradients, `batch_metrics` drives train/val logging.

### `train_model.py`

No changes. Hydra passes the new config keys through to `train()` automatically.

---

## Testing

**Update existing tests in `test_trainer.py`:**
- Rename `make_val_step` → `make_batch_metric_step` at all call sites.
- Rename `validation_loop` → `batch_metric_loop` at all call sites.
- Update return-type assertions: `float` → `dict[str, float]` for loop results,
  scalar assertions updated to index into dict by key.
- Update `_make_train_kwargs` to include `loss_fn`, `batch_metrics`,
  `epoch_metrics`, `num_train_eval_batches`, `num_val_eval_batches`.
- Update direct `make_train_step(optimizer)` call sites to
  `make_train_step(optimizer, loss_fn)` where `loss_fn=flow_matching_loss`.

**New tests:**
- `make_batch_metric_step`: dict keys match metric `__name__`, all values are
  scalar JAX arrays.
- `batch_metric_loop`: returns `dict[str, float]`, values are finite, `num_batches`
  limit is respected (loop stops early when limit is hit).
- `collect_batches`: returns correct number of tuples; `num_batches=0` returns
  all; each tuple is `(images, meta)`.
- `train()` integration: passing a no-op epoch metric callable receives
  `val_batches` as a non-empty list.
