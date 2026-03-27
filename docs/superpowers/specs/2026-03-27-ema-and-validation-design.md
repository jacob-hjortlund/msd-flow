# EMA Weights and Validation Step — Design Spec

**Date:** 2026-03-27
**Branch:** feature/EMA

---

## Overview

Extend the `train` function in `src/train/trainer.py` with:

1. **Exponential Moving Average (EMA)** of model weights, updated after every training step.
2. **Validation step** using EMA weights, run every `val_every` epochs.
3. **Epoch-based training loop** replacing the current step-based loop, with `num_steps_per_epoch` controlling inner loop length.

---

## Architecture

### `TrainState` — unchanged

No modifications. `ema_model` is kept as a separate local variable in `train()` to avoid passing unused arrays through the JIT-compiled `train_step` kernel.

### New: `ema_update(ema_model, new_model, decay)`

A plain Python function (not JIT-compiled). Uses `eqx.partition` to separate array leaves from static leaves, applies the EMA blend via `jax.tree_util.tree_map`, then recombines with `eqx.combine`. Non-array leaves (static model config) pass through unchanged.

```
new_ema_arrays = decay * ema_arrays + (1 - decay) * model_arrays
```

### New: `make_val_step()`

Returns a `filter_jit`-compiled function with signature:

```python
val_step(model, x_t, u_t, t, cond, cond_mask) -> loss
```

Calls `flow_matching_loss` directly with no gradient computation. Mirrors `make_train_step` in structure.

### Updated: `train(cfg, model, dataloader, val_dataloader, optimizer)`

New `val_dataloader` argument. The function:

- Initialises `ema_model = model` before the loop.
- Runs an epoch-based outer loop, step-based inner loop.
- Calls `ema_update` after every `train_step`.
- Runs a full validation pass at the end of every `val_every` epochs.
- Checkpoints `ema_model` (not `state.model`) so saved weights are always smoothed.
- Returns `ema_model` instead of `state.model`.

---

## Loop Structure

```
steps_per_epoch = len(dataloader) if num_steps_per_epoch == 0 else num_steps_per_epoch
ema_model = model

for epoch in range(num_epochs):

    # inner training loop
    for local_step in range(steps_per_epoch):
        batch = next(data_iter)        # restart iterator on StopIteration
        state, loss = train_step(state, ...)
        ema_model = ema_update(ema_model, state.model, ema_decay)

    # end-of-epoch hooks
    if (epoch + 1) % log_every == 0:
        log mean training loss over epoch

    if (epoch + 1) % val_every == 0:
        val_loss = average flow_matching_loss over all val batches using ema_model
        log val_loss

    if (epoch + 1) % checkpoint_every == 0:
        save ema_model checkpoint

return ema_model
```

**PRNG in validation:** a `key_val` is split from the main `key` at the start of each validation pass and used for time sampling and OT coupling, keeping validation deterministic per epoch.

---

## Config Changes (`configs/train/train.yaml`)

| Key | Change | Default |
|-----|--------|---------|
| `num_steps` | **removed** | — |
| `num_epochs` | **added** | `100` |
| `num_steps_per_epoch` | **added** | `0` (use `len(dataloader)`) |
| `ema_decay` | **added** | `0.9999` |
| `val_every` | **added** | `1` (every epoch) |
| `log_every` | unit change: steps → epochs | `1` |
| `checkpoint_every` | unit change: steps → epochs | `5` |

---

## Data Flow

```
dataloader (train split)
    └─► train_step → state.model
                          └─► ema_update → ema_model ──► val_step (val_dataloader)
                                                    └──► checkpoint
```

---

## Error Handling

- `val_dataloader` is a required argument; no `None` guard needed at this stage.
- `num_steps_per_epoch = 0` is the only special-cased config value.
- Time sampling mode (`uniform` / `logit_normal`) applies identically in train and val.

---

## Testing

- Unit test for `ema_update`: verify decay arithmetic on a trivial model.
- Unit test for `make_val_step`: verify it returns a scalar loss without error.
- Integration test for `train`: run 2 epochs on a tiny synthetic dataloader + val_dataloader, assert train and val losses are logged and EMA model is returned.
