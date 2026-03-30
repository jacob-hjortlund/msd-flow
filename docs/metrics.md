# Implementing Metrics

Metrics in msd-flow are plain Python callables registered in `configs/train/train.yaml`.
There are two types, differing in when and how they are called during training.

---

## Metric Types

### Batch Metrics

**Signature:** `(model, x_t, u_t, t, cond, cond_mask) -> scalar`

Called once per validation batch on **interpolated** data. Used to track training
progress and detect overfitting by comparing train vs. val values.

| Argument | Shape | Description |
|----------|-------|-------------|
| `model` | — | Current EMA model (Equinox module) |
| `x_t` | `(B, C, H, W)` | Interpolated images at time `t` |
| `u_t` | `(B, C, H, W)` | Target velocity field (`x_1 - x_0`) |
| `t` | `(B,)` | Per-sample times in `[0, 1)` |
| `cond` | `(B, cond_dim)` | Conditioning vectors; `(B, 0)` if unconditional |
| `cond_mask` | `(B,)` bool | `True` = use condition; `False` = drop (CFG) |

**Returns:** A scalar JAX array.

Batch metrics are JIT-compiled together into a single `filter_jit` step.
Each metric must have a **unique function name** — the trainer uses `fn.__name__`
as the logging key.

### Epoch Metrics

**Signature:** `(model, val_batches, key) -> scalar`

Called once per validation cycle on a fixed list of **raw** `(images, meta)` batches
collected from the val dataloader. Used for generation-based metrics (e.g. FID)
that need to sample from the model.

| Argument | Description |
|----------|-------------|
| `model` | Current EMA model |
| `val_batches` | `list[tuple[Tensor, Tensor]]` — raw `(images, meta)` PyTorch batches |
| `key` | JAX PRNG key |

**Returns:** A scalar float or JAX array.

Any additional dependencies (ODE solver, number of samples, etc.) must be baked in via
`functools.partial` or Hydra `_partial_: true` in the config.

---

## Example: Custom Batch Metric

The following adds a **mean absolute error** (MAE) metric alongside the default MSE loss.

**Step 1 — Add the function to `msdflow/train/metrics.py`:**

```python
import equinox as eqx
import jax.numpy as jnp
from msdflow.train.metrics import _to_velocity


def flow_matching_mae(
    model,
    x_t: jnp.ndarray,
    u_t: jnp.ndarray,
    t: jnp.ndarray,
    cond: jnp.ndarray,
    cond_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Mean absolute error between predicted and target velocity fields.

    Args:
        model:     Velocity-field network with a ``prediction_type`` attribute.
        x_t:       shape (B, C, H, W) — interpolated samples at time t.
        u_t:       shape (B, C, H, W) — target velocities.
        t:         shape (B,) — per-sample times in [0, 1).
        cond:      shape (B, cond_dim) — conditioning vectors.
        cond_mask: shape (B,) bool — CFG condition mask.

    Returns:
        Scalar mean absolute error.
    """
    pred = eqx.filter_vmap(model)(t, x_t, cond, cond_mask)
    v_t = _to_velocity(pred, x_t, t, model.prediction_type)
    return jnp.mean(jnp.abs(v_t - u_t))
```

**Step 2 — Register it in `configs/train/train.yaml`:**

```yaml
batch_metrics:
  - _target_: msdflow.train.metrics.flow_matching_loss
    _partial_: true
  - _target_: msdflow.train.metrics.flow_matching_mae
    _partial_: true
```

The metric will appear in the logs and ClearML as `val/flow_matching_mae` and
`train/flow_matching_mae`.

!!! warning "Unique function names required"
    Each entry in `batch_metrics` must resolve to a callable with a unique `__name__`.
    If two metrics share a name, the trainer raises a `ValueError` at startup.

---

## Example: Custom Epoch Metric

The following computes the **mean pixel-wise standard deviation** of generated samples —
a basic diversity measure that requires running the ODE sampler.

**Step 1 — Add the function to `msdflow/train/metrics.py`:**

```python
import jax
import jax.numpy as jnp
import numpy as np


def sample_diversity(
    model,
    val_batches: list,
    key: jax.Array,
    sample_fn: callable,
    num_samples: int = 16,
) -> float:
    """Mean pixel-wise standard deviation of generated samples.

    Measures how spread out the model's output distribution is.
    Higher values indicate more diverse samples.

    Args:
        model:       EMA velocity-field network.
        val_batches: Raw ``(images, meta)`` batches from the val dataloader.
                     Used to infer the image shape; values are not used.
        key:         JAX PRNG key for sampling.
        sample_fn:   Callable ``(model, key, image_shape) -> jnp.ndarray``
                     with shape ``(C, H, W)``. Bake in solver config via partial.
        num_samples: Number of images to generate.

    Returns:
        Mean pixel-wise standard deviation across all generated samples.
    """
    images, _ = val_batches[0]
    image_shape = tuple(images.shape[1:])  # (C, H, W)

    keys = jax.random.split(key, num_samples)
    samples = np.stack([np.array(sample_fn(model, k, image_shape)) for k in keys])
    return float(np.mean(np.std(samples, axis=0)))
```

**Step 2 — Register it in `configs/train/train.yaml`:**

```yaml
epoch_metrics:
  - _target_: msdflow.train.metrics.sample_diversity
    _partial_: true
    sample_fn:
      _target_: your.sampling.function   # replace with your ODE sampler callable
      _partial_: true
    num_samples: 16
```

!!! note
    `sample_fn` must be a callable with signature `(model, key, image_shape) -> array`.
    Implement your ODE sampler using `diffrax` and register it here via `_target_`.

The metric will appear in logs as `epoch/sample_diversity`.

---

## Metric Logging

All metrics are logged under fixed prefixes:

| Prefix | Source |
|--------|--------|
| `train/loss` | Per-epoch mean training loss |
| `val/<name>` | Batch metric evaluated on the val split |
| `train/<name>` | Batch metric evaluated on the train split |
| `epoch/<name>` | Epoch metric |

When ClearML is enabled (`clearml.enabled=true`), these scalars are tracked as
time series. Without ClearML they appear in the Python log output.
