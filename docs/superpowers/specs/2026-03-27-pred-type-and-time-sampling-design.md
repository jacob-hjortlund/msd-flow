# Design: Prediction-Type Abstraction & Time Sampling

**Date:** 2026-03-27
**Branch:** feature/pred-refactor

---

## Overview

Two related changes:

1. Allow flow-matching models to predict in **image space** (`x_t_pred`) instead of velocity space, while the loss and sampler always operate in velocity space.
2. Add reusable **time sampling functions** (`sample_time_uniform`, `sample_time_logit_normal`) to `src/flow/otfm.py` and wire them into the trainer.

Default behaviour is unchanged: velocity prediction, uniform time sampling.

---

## 1. Model Attribute: `prediction_type`

### Change
Add `prediction_type: str = eqx.field(static=True)` to both `UNet` and `NCSNpp`. The field defaults to `"velocity"`. Valid values are `"velocity"` and `"image"`; anything else raises `ValueError` in `__init__`.

### Rationale
The model is the natural owner of this information: both the loss function and the ODE sampler need to know how to interpret the model's output, and tying that knowledge to the model avoids threading an extra argument through every call site.

### What does NOT change
`__call__` is identical in both cases — the network outputs whatever it outputs. The attribute is purely a semantic declaration.

### Updated docstrings
- Class-level docstring: update "predicts v_t = …" to "predicts the velocity field or image, depending on `prediction_type`".
- `__init__` docstring: document the `prediction_type` parameter.
- `__call__` docstring: note that the return value is a velocity field when `prediction_type="velocity"` and a predicted image when `prediction_type="image"`.

---

## 2. Velocity Conversion Helper

A module-private helper is added to `src/flow/otfm.py`:

```python
def _to_velocity(
    pred: jnp.ndarray,
    x_t: jnp.ndarray,
    t: jnp.ndarray,
    prediction_type: str,
) -> jnp.ndarray:
    if prediction_type == "image":
        t_ = t[:, None, None, None]
        return (pred - x_t) / (1.0 - t_)
    return pred
```

This is the single point where image-space predictions are converted to velocity. It is used by both `flow_matching_loss` and `sample.py`.

---

## 3. Loss Function: `flow_matching_loss`

### Change
After vmapping the model, convert its output to velocity via `_to_velocity`, then compute MSE against `u_t` as before.

```python
pred = eqx.filter_vmap(model)(t, x_t, cond, cond_mask)
v_t = _to_velocity(pred, x_t, t, model.prediction_type)
return jnp.mean((v_t - u_t) ** 2)
```

The function signature is unchanged.

---

## 4. Sampler: `sample.py`

### Change
The `drift` function inside `sample` must convert the model output to velocity before Diffrax integrates it. `_to_velocity` is imported from `otfm` and applied inside `drift`.

For CFG (guidance_scale != 1.0), each branch (conditional, unconditional) is converted independently **before** the guidance blend — blending happens in velocity space:

```python
def drift(t, y, args):
    if guidance_scale == 1.0:
        pred = model(t, y, _cond, _mask)
        return _to_velocity(pred, y, t[None], model.prediction_type)
    pred_cond = model(t, y, _cond, mask_true)
    pred_uncond = model(t, y, _cond, mask_false)
    v_cond = _to_velocity(pred_cond, y, t[None], model.prediction_type)
    v_uncond = _to_velocity(pred_uncond, y, t[None], model.prediction_type)
    return v_uncond + guidance_scale * (v_cond - v_uncond)
```

Note: inside `drift`, `t` is a JAX scalar (shape `()`). `_to_velocity` uses `t[:, None, None, None]` for broadcasting, so inside `drift` we pass `jnp.reshape(t, (1,))` and the returned tensor has shape `(1, C, H, W)`, which is squeezed to `(C, H, W)` before returning from `drift`.

---

## 5. Time Sampling Functions (`src/flow/otfm.py`)

Two new public functions, JAX-native:

```python
def sample_time_uniform(
    key: jax.Array,
    batch_size: int,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> jnp.ndarray:
    """Sample times uniformly in [t_min, t_max]."""
    return jax.random.uniform(key, (batch_size,), minval=t_min, maxval=t_max)


def sample_time_logit_normal(
    key: jax.Array,
    batch_size: int,
    mu: float = -0.8,
    sigma: float = 0.8,
) -> jnp.ndarray:
    """Sample times via logit-normal distribution.

    Draws u ~ Normal(mu, sigma), then applies sigmoid to map to (0, 1).
    """
    u = jax.random.normal(key, (batch_size,)) * sigma + mu
    return jax.nn.sigmoid(u)
```

---

## 6. Trainer: `src/train/trainer.py`

### Change
Replace the hardcoded numpy time-sampling call:

```python
t_np = rng.uniform(t_min, t_max, size=(B,)).astype(np.float32)
```

with a JAX key split + dispatch based on `cfg.flow.otfm.time_sampling`:

```python
key, key_time, key_path = jax.random.split(key, 3)
time_sampling = cfg.flow.otfm.get("time_sampling", "uniform")
if time_sampling == "uniform":
    t = sample_time_uniform(key_time, B, t_min, t_max)
elif time_sampling == "logit_normal":
    t = sample_time_logit_normal(key_time, B)
else:
    raise ValueError(f"Unknown time_sampling: {time_sampling!r}")
```

`t` is now a `jax.Array` directly; no `.astype(np.float32)` conversion needed.

---

## 7. Tests

### New tests for `UNet` / `NCSNpp`
- `prediction_type` defaults to `"velocity"`.
- `prediction_type="image"` is accepted.
- Invalid `prediction_type` raises `ValueError`.

### New tests for `_to_velocity` / `flow_matching_loss`
- Image-prediction model: loss is computed in velocity space (verify the conversion formula numerically).
- Velocity-prediction model: behaviour is unchanged.

### New tests for time sampling
- `sample_time_uniform`: output in `[t_min, t_max]`, shape `(B,)`, deterministic given same key.
- `sample_time_logit_normal`: output in `(0, 1)`, shape `(B,)`, deterministic given same key.

### Updated tests
- `test_otfm.py`: existing tests are unaffected (models default to `"velocity"`).
- `test_sample.py`: ensure `drift` works for both prediction types.
