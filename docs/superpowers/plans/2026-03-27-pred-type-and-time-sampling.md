# Prediction-Type Abstraction & Time Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow flow-matching models to predict in image space while the loss and sampler always operate in velocity space; add reusable JAX time-sampling functions and wire them into the trainer.

**Architecture:** A `prediction_type` static field on each model declares its output semantics. A shared `_to_velocity` helper in `otfm.py` converts model output to velocity when needed, and is called by both `flow_matching_loss` and the ODE drift in `sample.py`. Time sampling functions are pure JAX functions in `otfm.py`; the trainer dispatches to them via a config key.

**Tech Stack:** JAX, Equinox, Diffrax, Hydra, pytest

---

## File Map

| File | Change |
|---|---|
| `src/flow/otfm.py` | Add `_to_velocity`, `sample_time_uniform`, `sample_time_logit_normal`; update `flow_matching_loss` |
| `src/model/unet.py` | Add `prediction_type` field + param + validation |
| `src/model/ncsnpp.py` | Add `prediction_type` field + param + validation |
| `src/flow/sample.py` | Import `_to_velocity`; update `drift` closure |
| `src/train/trainer.py` | Import new samplers; replace numpy time sampling with JAX dispatch |
| `tests/flow/test_otfm.py` | Add tests for `_to_velocity`, time samplers, image-mode loss |
| `tests/model/test_unet.py` | Add `prediction_type` tests |
| `tests/model/test_ncsnpp.py` | Add `prediction_type` tests |
| `tests/flow/test_sample.py` | Add image-prediction sampling tests |
| `tests/train/test_trainer.py` | Add invalid time_sampling error test |

---

## Task 1: Add `_to_velocity` and time-sampling functions to `otfm.py`

**Files:**
- Modify: `src/flow/otfm.py`
- Test: `tests/flow/test_otfm.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/flow/test_otfm.py`:

```python
from src.flow.otfm import _to_velocity, sample_time_uniform, sample_time_logit_normal


# --- _to_velocity ---

def test_to_velocity_velocity_mode_returns_pred():
    """In velocity mode, _to_velocity is an identity on pred."""
    pred = jnp.ones((2, 1, 4, 4)) * 3.0
    x_t = jnp.ones((2, 1, 4, 4)) * 1.0
    t = jnp.array([0.4, 0.6])
    v = _to_velocity(pred, x_t, t, "velocity")
    assert jnp.allclose(v, pred)


def test_to_velocity_image_mode_formula():
    """In image mode, _to_velocity applies (pred - x_t) / (1 - t)."""
    pred = jnp.ones((2, 1, 4, 4)) * 2.0
    x_t = jnp.ones((2, 1, 4, 4)) * 0.5
    t = jnp.array([0.5, 0.5])
    v = _to_velocity(pred, x_t, t, "image")
    expected = (pred - x_t) / (1.0 - t[:, None, None, None])
    assert jnp.allclose(v, expected)


def test_to_velocity_image_mode_shape():
    """Output shape matches input shape."""
    B, C, H, W = 3, 2, 8, 8
    pred = jax.random.normal(KEY, (B, C, H, W))
    x_t = jax.random.normal(KEY, (B, C, H, W))
    t = jnp.array([0.1, 0.5, 0.9])
    v = _to_velocity(pred, x_t, t, "image")
    assert v.shape == (B, C, H, W)


# --- sample_time_uniform ---

def test_sample_time_uniform_shape():
    """Output has shape (batch_size,)."""
    t = sample_time_uniform(KEY, 64)
    assert t.shape == (64,)


def test_sample_time_uniform_range_defaults():
    """All samples lie in [0, 1] with default t_min/t_max."""
    t = sample_time_uniform(KEY, 1000)
    assert jnp.all(t >= 0.0) and jnp.all(t <= 1.0)


def test_sample_time_uniform_custom_range():
    """All samples lie in [t_min, t_max] when overridden."""
    t = sample_time_uniform(KEY, 1000, t_min=0.2, t_max=0.7)
    assert jnp.all(t >= 0.2) and jnp.all(t <= 0.7)


def test_sample_time_uniform_deterministic():
    """Same key gives identical output."""
    t1 = sample_time_uniform(jax.random.PRNGKey(7), 32)
    t2 = sample_time_uniform(jax.random.PRNGKey(7), 32)
    assert jnp.allclose(t1, t2)


def test_sample_time_uniform_different_keys_differ():
    """Different keys give different samples."""
    t1 = sample_time_uniform(jax.random.PRNGKey(0), 32)
    t2 = sample_time_uniform(jax.random.PRNGKey(1), 32)
    assert not jnp.allclose(t1, t2)


# --- sample_time_logit_normal ---

def test_sample_time_logit_normal_shape():
    """Output has shape (batch_size,)."""
    t = sample_time_logit_normal(KEY, 64)
    assert t.shape == (64,)


def test_sample_time_logit_normal_range():
    """All samples are strictly in (0, 1) — sigmoid always outputs in that range."""
    t = sample_time_logit_normal(KEY, 1000)
    assert jnp.all(t > 0.0) and jnp.all(t < 1.0)


def test_sample_time_logit_normal_deterministic():
    """Same key gives identical output."""
    t1 = sample_time_logit_normal(jax.random.PRNGKey(7), 32)
    t2 = sample_time_logit_normal(jax.random.PRNGKey(7), 32)
    assert jnp.allclose(t1, t2)


def test_sample_time_logit_normal_different_keys_differ():
    """Different keys give different samples."""
    t1 = sample_time_logit_normal(jax.random.PRNGKey(0), 32)
    t2 = sample_time_logit_normal(jax.random.PRNGKey(1), 32)
    assert not jnp.allclose(t1, t2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/flow/test_otfm.py::test_to_velocity_velocity_mode_returns_pred \
       tests/flow/test_otfm.py::test_sample_time_uniform_shape \
       tests/flow/test_otfm.py::test_sample_time_logit_normal_shape -v
```

Expected: `ImportError` or `FAILED` — `_to_velocity`, `sample_time_uniform`, `sample_time_logit_normal` do not exist yet.

- [ ] **Step 3: Add the three functions to `src/flow/otfm.py`**

Add after the existing imports, before `sample_path`:

```python
def _to_velocity(
    pred: jnp.ndarray,
    x_t: jnp.ndarray,
    t: jnp.ndarray,
    prediction_type: str,
) -> jnp.ndarray:
    """Convert a model prediction to a velocity field.

    Args:
        pred:            shape (B, C, H, W) — raw model output.
        x_t:             shape (B, C, H, W) — interpolated samples at time t.
        t:               shape (B,) — per-sample times in [0, 1).
        prediction_type: ``"velocity"`` returns ``pred`` unchanged;
            ``"image"`` applies ``(pred - x_t) / (1 - t)``.

    Returns:
        Velocity field of shape (B, C, H, W).
    """
    if prediction_type == "image":
        t_ = t[:, None, None, None]
        return (pred - x_t) / (1.0 - t_)
    return pred


def sample_time_uniform(
    key: jax.Array,
    batch_size: int,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> jnp.ndarray:
    """Sample times uniformly from [t_min, t_max].

    Args:
        key:        JAX PRNG key.
        batch_size: Number of time samples to draw.
        t_min:      Lower bound of the uniform distribution. Default 0.0.
        t_max:      Upper bound of the uniform distribution. Default 1.0.

    Returns:
        Array of shape (batch_size,) with values in [t_min, t_max].
    """
    return jax.random.uniform(key, (batch_size,), minval=t_min, maxval=t_max)


def sample_time_logit_normal(
    key: jax.Array,
    batch_size: int,
    mu: float = -0.8,
    sigma: float = 0.8,
) -> jnp.ndarray:
    """Sample times via a logit-normal distribution.

    Draws ``u ~ Normal(mu, sigma)`` then applies sigmoid to map to (0, 1).
    The default ``mu=-0.8, sigma=0.8`` biases samples toward the middle of the
    interval, following Esser et al. 2024 (Stable Diffusion 3).

    Args:
        key:        JAX PRNG key.
        batch_size: Number of time samples to draw.
        mu:         Mean of the underlying normal. Default -0.8.
        sigma:      Std-dev of the underlying normal. Default 0.8.

    Returns:
        Array of shape (batch_size,) with values in (0, 1).
    """
    u = jax.random.normal(key, (batch_size,)) * sigma + mu
    return jax.nn.sigmoid(u)
```

- [ ] **Step 4: Run all new tests to verify they pass**

```bash
pytest tests/flow/test_otfm.py -k "to_velocity or time_uniform or time_logit" -v
```

Expected: all 11 new tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flow/otfm.py tests/flow/test_otfm.py
git commit -m "feat: add _to_velocity helper and time sampling functions to otfm"
```

---

## Task 2: Add `prediction_type` to `UNet`

**Files:**
- Modify: `src/model/unet.py`
- Test: `tests/model/test_unet.py`

- [ ] **Step 1: Write failing tests**

Read `tests/model/test_unet.py` first, then append:

```python
from src.model.unet import UNet
import pytest

_KEY = jax.random.PRNGKey(0)
_SMALL = dict(
    in_channels=1, out_channels=1, base_channels=4,
    channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
    num_groups=2, activation=jax.nn.silu,
)


def test_unet_prediction_type_default():
    """UNet defaults to velocity prediction."""
    model = UNet(**_SMALL, key=_KEY)
    assert model.prediction_type == "velocity"


def test_unet_prediction_type_image():
    """UNet accepts prediction_type='image'."""
    model = UNet(**_SMALL, key=_KEY, prediction_type="image")
    assert model.prediction_type == "image"


def test_unet_prediction_type_invalid():
    """UNet raises ValueError for unknown prediction_type."""
    with pytest.raises(ValueError, match="prediction_type"):
        UNet(**_SMALL, key=_KEY, prediction_type="score")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/model/test_unet.py::test_unet_prediction_type_default \
       tests/model/test_unet.py::test_unet_prediction_type_image \
       tests/model/test_unet.py::test_unet_prediction_type_invalid -v
```

Expected: `FAILED` or `AttributeError` — `prediction_type` does not exist yet.

- [ ] **Step 3: Add the field and parameter to `src/model/unet.py`**

**3a.** In the class body, add the field after `activation`:

```python
    activation: Callable = eqx.field(static=True)
    prediction_type: str = eqx.field(static=True)
```

**3b.** In `__init__`, add `prediction_type: str = "velocity"` after `cond_dim`:

```python
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        channel_multipliers: List[int],
        num_res_blocks: int,
        num_heads: int,
        num_groups: int,
        activation: Callable,
        key: jax.Array,
        cond_dim: int = 0,
        prediction_type: str = "velocity",
    ):
```

**3c.** Add validation and assignment in `__init__`, right after the `cond_dim` validation block:

```python
        if prediction_type not in ("velocity", "image"):
            raise ValueError(
                f"prediction_type={prediction_type!r} is not supported; "
                "choose 'velocity' or 'image'."
            )
        self.prediction_type = prediction_type
```

**3d.** Update the `__init__` docstring — add under the `cond_dim` line:

```
            prediction_type: Output semantics of the network. ``"velocity"``
                (default) means the network predicts the velocity field
                ``v_t`` directly. ``"image"`` means it predicts the target
                image ``x_t_pred``; the caller converts to velocity via
                ``(x_t_pred - x_t) / (1 - t)``.
```

**3e.** Update the class-level docstring attribute list — replace:

```
        activation: Activation function used throughout.
```

with:

```
        activation: Activation function used throughout.
        prediction_type: Output semantics — ``"velocity"`` or ``"image"``.
```

**3f.** Update the `__call__` docstring — replace the `Returns:` line:

```
        Returns:
            Predicted velocity field of shape ``(C, H, W)`` when
            ``prediction_type="velocity"``, or predicted image of shape
            ``(C, H, W)`` when ``prediction_type="image"``.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/model/test_unet.py -v
```

Expected: all tests (existing + 3 new) PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model/unet.py tests/model/test_unet.py
git commit -m "feat: add prediction_type field to UNet"
```

---

## Task 3: Add `prediction_type` to `NCSNpp`

**Files:**
- Modify: `src/model/ncsnpp.py`
- Test: `tests/model/test_ncsnpp.py`

- [ ] **Step 1: Write failing tests**

Read the full `tests/model/test_ncsnpp.py`, then append (the file already defines `SMALL_CFG`):

```python
def test_ncsnpp_prediction_type_default():
    """NCSNpp defaults to velocity prediction."""
    model = NCSNpp(**SMALL_CFG, key=KEY)
    assert model.prediction_type == "velocity"


def test_ncsnpp_prediction_type_image():
    """NCSNpp accepts prediction_type='image'."""
    model = NCSNpp(**SMALL_CFG, key=KEY, prediction_type="image")
    assert model.prediction_type == "image"


def test_ncsnpp_prediction_type_invalid():
    """NCSNpp raises ValueError for unknown prediction_type."""
    with pytest.raises(ValueError, match="prediction_type"):
        NCSNpp(**SMALL_CFG, key=KEY, prediction_type="score")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/model/test_ncsnpp.py::test_ncsnpp_prediction_type_default \
       tests/model/test_ncsnpp.py::test_ncsnpp_prediction_type_image \
       tests/model/test_ncsnpp.py::test_ncsnpp_prediction_type_invalid -v
```

Expected: `FAILED` or `AttributeError`.

- [ ] **Step 3: Add the field and parameter to `src/model/ncsnpp.py`**

**3a.** In the class body, add after `activation`:

```python
    activation: Callable = eqx.field(static=True)
    channel_multipliers: List[int] = eqx.field(static=True)
    num_res_blocks: int = eqx.field(static=True)
    attn_resolutions: List[int] = eqx.field(static=True)
    image_size: int = eqx.field(static=True)
    prediction_type: str = eqx.field(static=True)
```

(The last four lines already exist — just add `prediction_type` after `image_size`.)

**3b.** In `__init__`, add `prediction_type: str = "velocity"` after `cond_dim`:

```python
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        channel_multipliers: List[int],
        num_res_blocks: int,
        attn_resolutions: List[int],
        dropout: float,
        num_groups: int,
        num_heads: int,
        activation: Callable,
        fourier_scale: float,
        skip_rescale: bool,
        image_size: int,
        key: jax.Array,
        cond_dim: int = 0,
        prediction_type: str = "velocity",
    ):
```

**3c.** Add validation and assignment right after the `cond_dim` validation block (around line 126):

```python
        if prediction_type not in ("velocity", "image"):
            raise ValueError(
                f"prediction_type={prediction_type!r} is not supported; "
                "choose 'velocity' or 'image'."
            )
        self.prediction_type = prediction_type
```

**3d.** Update the `__init__` docstring — add under the `cond_dim` line:

```
            prediction_type: Output semantics of the network. ``"velocity"``
                (default) means the network predicts the velocity field
                ``v_t`` directly. ``"image"`` means it predicts the target
                image ``x_t_pred``; the caller converts to velocity via
                ``(x_t_pred - x_t) / (1 - t)``.
```

**3e.** Update the class-level docstring — add after `activation`:

```
        prediction_type: Output semantics — ``"velocity"`` or ``"image"``.
```

**3f.** Update `__call__` docstring `Returns:` line:

```
        Returns:
            Predicted velocity field of shape ``(C, H, W)`` when
            ``prediction_type="velocity"``, or predicted image of shape
            ``(C, H, W)`` when ``prediction_type="image"``.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/model/test_ncsnpp.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model/ncsnpp.py tests/model/test_ncsnpp.py
git commit -m "feat: add prediction_type field to NCSNpp"
```

---

## Task 4: Update `flow_matching_loss` to use `_to_velocity`

**Files:**
- Modify: `src/flow/otfm.py`
- Test: `tests/flow/test_otfm.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/flow/test_otfm.py`:

```python
from src.model.unet import UNet as _UNet

_KEY2 = jax.random.PRNGKey(99)
_SMALL_IMG_MODEL = _UNet(
    in_channels=1, out_channels=1, base_channels=4,
    channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
    num_groups=2, activation=jax.nn.silu, key=_KEY2,
    prediction_type="image",
)


def test_flow_matching_loss_image_mode_is_scalar():
    """Loss with an image-prediction model is a scalar."""
    B = 2
    k1, k2 = jax.random.split(_KEY2)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss = flow_matching_loss(_SMALL_IMG_MODEL, x_t, u_t, t, cond, cond_mask)
    assert loss.shape == ()


def test_flow_matching_loss_image_mode_is_finite():
    """Loss with an image-prediction model is finite."""
    B = 2
    k1, k2 = jax.random.split(_KEY2)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss = flow_matching_loss(_SMALL_IMG_MODEL, x_t, u_t, t, cond, cond_mask)
    assert jnp.isfinite(loss)


def test_flow_matching_loss_image_mode_has_gradient():
    """Gradients flow through image-mode loss."""
    B = 2
    k1, k2 = jax.random.split(_KEY2)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    _, grads = eqx.filter_value_and_grad(flow_matching_loss)(
        _SMALL_IMG_MODEL, x_t, u_t, t, cond, cond_mask
    )
    grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(g != 0.0) for g in grad_leaves)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/flow/test_otfm.py::test_flow_matching_loss_image_mode_is_scalar \
       tests/flow/test_otfm.py::test_flow_matching_loss_image_mode_is_finite \
       tests/flow/test_otfm.py::test_flow_matching_loss_image_mode_has_gradient -v
```

Expected: tests fail because `flow_matching_loss` ignores `model.prediction_type` and the loss is wrong.

- [ ] **Step 3: Update `flow_matching_loss` in `src/flow/otfm.py`**

Replace the body of `flow_matching_loss`:

```python
def flow_matching_loss(
    model,
    x_t: jnp.ndarray,
    u_t: jnp.ndarray,
    t: jnp.ndarray,
    cond: jnp.ndarray,
    cond_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the flow matching MSE loss.

    Supports velocity-predicting and image-predicting models. The loss is
    always computed in velocity space; image-space predictions are converted
    via ``v_t = (x_t_pred - x_t) / (1 - t)`` before the MSE is evaluated.

    Args:
        model: Network accepting ``(t, x_t, cond, cond_mask)``. Must have a
            ``prediction_type`` attribute of ``"velocity"`` (default) or
            ``"image"``.
        x_t:   shape (B, C, H, W) — interpolated samples at time t.
        u_t:   shape (B, C, H, W) — target velocities (x1 - x0).
        t:     shape (B,) — per-sample times in [0, 1).
        cond:  shape (B, cond_dim) — conditioning vectors. Pass
            ``jnp.empty((B, 0))`` when the model is unconditional.
        cond_mask: shape (B,) bool — per-sample mask. ``True`` = use
            the real condition; ``False`` = use the null embedding.

    Returns:
        Scalar mean squared error between predicted and target velocities.
    """
    pred = eqx.filter_vmap(model)(t, x_t, cond, cond_mask)
    v_t = _to_velocity(pred, x_t, t, model.prediction_type)
    return jnp.mean((v_t - u_t) ** 2)
```

- [ ] **Step 4: Run the full `test_otfm.py` to verify all tests pass**

```bash
pytest tests/flow/test_otfm.py -v
```

Expected: all tests (existing + new) PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flow/otfm.py tests/flow/test_otfm.py
git commit -m "feat: update flow_matching_loss to support image-space prediction"
```

---

## Task 5: Update `sample.py` drift to use `_to_velocity`

**Files:**
- Modify: `src/flow/sample.py`
- Test: `tests/flow/test_sample.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/flow/test_sample.py`:

```python
_SMALL_IMG = UNet(
    in_channels=1, out_channels=1, base_channels=4,
    channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
    num_groups=2, activation=jax.nn.silu, key=KEY,
    prediction_type="image",
)

_SMALL_IMG_COND = UNet(
    in_channels=1, out_channels=1, base_channels=4,
    channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
    num_groups=2, activation=jax.nn.silu, cond_dim=1, key=KEY,
    prediction_type="image",
)


def test_sample_image_prediction_shape():
    """sample() with an image-prediction model returns correct shape."""
    out = sample(
        model=_SMALL_IMG,
        shape=(1, 8, 8),
        key=KEY,
        solver=diffrax.Euler,
        dt0=0.1,
        t0=0.0,
        t1=1.0,
        stepsize_controller=diffrax.ConstantStepSize,
        stepsize_controller_cfg={},
    )
    assert out.shape == (1, 8, 8)


def test_sample_image_prediction_finite():
    """sample() with an image-prediction model returns finite values."""
    out = sample(
        model=_SMALL_IMG,
        shape=(1, 8, 8),
        key=KEY,
        solver=diffrax.Euler,
        dt0=0.1,
        t0=0.0,
        t1=1.0,
        stepsize_controller=diffrax.ConstantStepSize,
        stepsize_controller_cfg={},
    )
    assert jnp.all(jnp.isfinite(out))


def test_sample_image_prediction_guided_shape():
    """Guided sampling with an image-prediction model returns correct shape."""
    out = sample(
        model=_SMALL_IMG_COND,
        shape=(1, 8, 8),
        key=KEY,
        solver=diffrax.Euler,
        dt0=0.1,
        t0=0.0,
        t1=1.0,
        stepsize_controller=diffrax.ConstantStepSize,
        stepsize_controller_cfg={},
        cond=jnp.array([0.4]),
        guidance_scale=2.0,
    )
    assert out.shape == (1, 8, 8)
    assert jnp.all(jnp.isfinite(out))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/flow/test_sample.py::test_sample_image_prediction_shape \
       tests/flow/test_sample.py::test_sample_image_prediction_finite \
       tests/flow/test_sample.py::test_sample_image_prediction_guided_shape -v
```

Expected: `FAILED` — the current `drift` returns the model output directly (treating it as velocity), which is wrong for `prediction_type="image"`.

- [ ] **Step 3: Update `src/flow/sample.py`**

**3a.** Add `_to_velocity` to the import line at the top:

```python
from src.flow.otfm import _to_velocity
```

**3b.** Replace the `drift` closure inside `sample`:

```python
    def drift(t, y, args):
        # t is a JAX scalar; _to_velocity expects shape (B,), so we
        # temporarily add/remove a batch dimension.
        t_batch = jnp.reshape(t, (1,))
        y_batch = y[None]  # (1, C, H, W)

        if guidance_scale == 1.0:
            pred = model(t, y, _cond, _mask)
            return _to_velocity(pred[None], y_batch, t_batch, model.prediction_type)[0]

        pred_cond = model(t, y, _cond, mask_true)
        pred_uncond = model(t, y, _cond, mask_false)
        v_cond = _to_velocity(pred_cond[None], y_batch, t_batch, model.prediction_type)[0]
        v_uncond = _to_velocity(pred_uncond[None], y_batch, t_batch, model.prediction_type)[0]
        return v_uncond + guidance_scale * (v_cond - v_uncond)
```

- [ ] **Step 4: Run the full `test_sample.py` to verify all tests pass**

```bash
pytest tests/flow/test_sample.py -v
```

Expected: all tests (existing + 3 new) PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flow/sample.py tests/flow/test_sample.py
git commit -m "feat: update sample drift to support image-space prediction via _to_velocity"
```

---

## Task 6: Update trainer to use JAX time-sampling functions

**Files:**
- Modify: `src/train/trainer.py`
- Test: `tests/train/test_trainer.py`

- [ ] **Step 1: Write failing test**

Read `tests/train/test_trainer.py` first, then append:

```python
import pytest
from unittest.mock import MagicMock


def test_train_raises_on_unknown_time_sampling():
    """train() raises ValueError for an unrecognised time_sampling value."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax
    from src.model.unet import UNet
    from src.train.trainer import train

    key = jax.random.PRNGKey(0)
    model = UNet(
        in_channels=1, out_channels=1, base_channels=4,
        channel_multipliers=[1, 2], num_res_blocks=1, num_heads=1,
        num_groups=2, activation=jax.nn.silu, key=key,
    )
    optimizer = optax.adam(1e-3)

    # Minimal fake dataloader: one batch of ones
    import torch
    images = torch.ones(2, 1, 4, 4)
    meta = torch.zeros(2, 0)
    dataloader = [(images, meta)]

    cfg = MagicMock()
    cfg.seed = 0
    cfg.flow.otfm.t_min = 0.0
    cfg.flow.otfm.t_max = 1.0
    cfg.flow.otfm.get.side_effect = lambda key, default=None: {
        "sigma_0": 0.0,
        "sigma_1": 0.0,
        "time_sampling": "bad_value",
    }.get(key, default)
    cfg.train.num_steps = 1
    cfg.train.log_every = 1
    cfg.train.checkpoint_every = 100
    cfg.train.checkpoint_dir = "/tmp/ckpt_test"
    cfg.train.get.return_value = 0.0  # p_uncond

    with pytest.raises(ValueError, match="time_sampling"):
        train(cfg, model, dataloader, optimizer)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/train/test_trainer.py::test_train_raises_on_unknown_time_sampling -v
```

Expected: `FAILED` — the trainer does not yet raise on unknown `time_sampling`.

- [ ] **Step 3: Update `src/train/trainer.py`**

**3a.** Update the import from `otfm`:

```python
from src.flow.otfm import (
    flow_matching_loss,
    sample_path,
    sample_time_uniform,
    sample_time_logit_normal,
)
```

**3b.** Inside `train`, replace the key split line and the numpy time-sampling line.

Replace:

```python
        key, key_cpu, key_path = jax.random.split(key, 3)
```

with:

```python
        key, key_cpu, key_time, key_path = jax.random.split(key, 4)
```

Replace:

```python
        t_np = rng.uniform(t_min, t_max, size=(B,)).astype(np.float32)
```

with:

```python
        if time_sampling == "uniform":
            t = sample_time_uniform(key_time, B, t_min, t_max)
        elif time_sampling == "logit_normal":
            t = sample_time_logit_normal(key_time, B)
        else:
            raise ValueError(
                f"Unknown time_sampling={time_sampling!r}; "
                "choose 'uniform' or 'logit_normal'."
            )
```

**3c.** Read `time_sampling` from config before the loop (alongside `t_min`, `t_max`):

```python
    time_sampling = cfg.flow.otfm.get("time_sampling", "uniform")
```

**3d.** Replace the two lines that convert numpy arrays to JAX:

```python
        x_t, u_t = _sample_path(
            jnp.array(x0_paired), jnp.array(x1_np), jnp.array(t_np), key=key_path
        )
        t = jnp.array(t_np)
```

with:

```python
        x_t, u_t = _sample_path(
            jnp.array(x0_paired), jnp.array(x1_np), t, key=key_path
        )
```

(`t` is now already a `jax.Array` from the sampling functions above.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/train/test_trainer.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/train/trainer.py tests/train/test_trainer.py
git commit -m "feat: replace numpy time sampling in trainer with JAX sample_time_* functions"
```
