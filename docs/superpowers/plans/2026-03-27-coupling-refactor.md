# Coupling Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract coupling logic from `otfm.py` into `coupling.py`, and extend `sample_ot_path` (renamed `sample_path`) with optional Gaussian noise injection on the interpolant.

**Architecture:** A new `src/flow/coupling.py` module owns both coupling strategies behind a uniform `(x0, x1) -> x0_paired` interface. `otfm.py` is trimmed to path construction and loss only. All existing callers are updated to import from the new locations.

**Tech Stack:** JAX, NumPy, SciPy (`linear_sum_assignment`), pytest

---

### Task 1: Create `src/flow/coupling.py` with `independent_coupling` and `ot_coupling`

**Files:**
- Create: `src/flow/coupling.py`
- Create: `tests/flow/test_coupling.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/flow/test_coupling.py`:

```python
"""Tests for src.flow.coupling."""

import numpy as np
import pytest
from src.flow.coupling import independent_coupling, ot_coupling


def test_independent_coupling_returns_x0_unchanged():
    """independent_coupling must return the exact x0 array."""
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    x1 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    result = independent_coupling(x0, x1)
    np.testing.assert_array_equal(result, x0)


def test_independent_coupling_output_shape():
    """independent_coupling output shape matches input."""
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    x1 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    assert independent_coupling(x0, x1).shape == x0.shape


def test_ot_coupling_output_shape():
    """ot_coupling output shape matches input shape."""
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    x1 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    x0_paired = ot_coupling(x0, x1)
    assert x0_paired.shape == x0.shape


def test_ot_coupling_is_permutation():
    """ot_coupling returns a permutation of the source rows."""
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    x1 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    x0_paired = ot_coupling(x0, x1)
    x0_flat = x0.reshape(4, -1)
    x0p_flat = x0_paired.reshape(4, -1)
    for row in x0p_flat:
        assert any(np.allclose(row, x0_row) for x0_row in x0_flat)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/flow/test_coupling.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `coupling.py` doesn't exist yet.

- [ ] **Step 3: Create `src/flow/coupling.py`**

```python
"""Minibatch coupling strategies for flow matching.

Both functions share the interface (x0, x1) -> x0_paired, where x0_paired
is the permuted (or identity) source array to be paired with x1 during training.
All operations run on NumPy/CPU, outside JAX JIT.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def independent_coupling(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Return x0 unchanged — the independent (no-pairing) coupling.

    x0 and x1 are treated as independently drawn samples with no attempt to
    match them. This is the baseline coupling for vanilla flow matching.

    Args:
        x0: shape (B, C, H, W) — noise samples.
        x1: shape (B, C, H, W) — data samples (unused).

    Returns:
        x0 unchanged, shape (B, C, H, W).
    """
    return x0


def ot_coupling(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Pair x0 ~ N(0,I) with x1 ~ p_data via minibatch optimal transport.

    Runs the Hungarian algorithm on the pairwise squared L2 cost matrix to
    return a permutation of x0 that minimises total transport cost to x1.
    Both inputs shape (B, C, H, W).

    Args:
        x0: shape (B, C, H, W) — noise samples.
        x1: shape (B, C, H, W) — data samples.

    Returns:
        Permutation of x0 that minimises squared L2 cost to x1,
        shape (B, C, H, W).
    """
    B = x0.shape[0]
    x0_flat = x0.reshape(B, -1)
    x1_flat = x1.reshape(B, -1)
    cost = np.sum((x0_flat[:, None, :] - x1_flat[None, :, :]) ** 2, axis=-1)
    _, col_ind = linear_sum_assignment(cost)
    return x0[col_ind]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/flow/test_coupling.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flow/coupling.py tests/flow/test_coupling.py
git commit -m "feat: add coupling.py with independent_coupling and ot_coupling"
```

---

### Task 2: Remove `minibatch_ot_coupling` from `otfm.py` and update its tests

**Files:**
- Modify: `src/flow/otfm.py` — delete `minibatch_ot_coupling` and its imports
- Modify: `tests/flow/test_otfm.py` — update imports; existing OT coupling tests move to `test_coupling.py` (already done in Task 1)

- [ ] **Step 1: Delete `minibatch_ot_coupling` from `otfm.py`**

Remove the `scipy` import line and the `minibatch_ot_coupling` function. The top of `src/flow/otfm.py` should become:

```python
"""Optimal-transport flow matching loss and utilities.

Implements the linear interpolant path and the MSE flow matching objective.
"""

import jax
import numpy as np
import equinox as eqx
import jax.numpy as jnp
```

Remove lines 12–27 (the `from scipy.optimize import linear_sum_assignment` import and the entire `minibatch_ot_coupling` function body).

- [ ] **Step 2: Update imports in `tests/flow/test_otfm.py`**

Change line 12 from:

```python
from src.flow.otfm import minibatch_ot_coupling, sample_ot_path
```

to:

```python
from src.flow.otfm import sample_ot_path
```

(The OT coupling tests live in `test_coupling.py` from Task 1; no OT coupling tests remain in `test_otfm.py`.)

- [ ] **Step 3: Run the full test suite to verify nothing broke**

```bash
pytest tests/flow/ -v
```

Expected: all existing `test_otfm.py` tests PASS, `test_coupling.py` tests PASS. No import errors.

- [ ] **Step 4: Commit**

```bash
git add src/flow/otfm.py tests/flow/test_otfm.py
git commit -m "refactor: remove minibatch_ot_coupling from otfm.py"
```

---

### Task 3: Rename `sample_ot_path` → `sample_path` and add optional noise

**Files:**
- Modify: `src/flow/otfm.py` — rename function, add sigma/key params, add noise branch
- Modify: `tests/flow/test_otfm.py` — update all references; add stochastic tests

- [ ] **Step 1: Write the new failing tests in `tests/flow/test_otfm.py`**

Replace all occurrences of `sample_ot_path` with `sample_path` in the import and existing test names. Then add these new tests at the end of the file:

Update the import line (currently `from src.flow.otfm import sample_ot_path`) to:

```python
from src.flow.otfm import flow_matching_loss, sample_path
```

Rename existing test functions (these already pass conceptually — just the name changes):

- `test_sample_ot_path_at_t0_gives_x0` → `test_sample_path_at_t0_gives_x0`
- `test_sample_ot_path_at_t1_gives_x1` → `test_sample_path_at_t1_gives_x1`
- `test_sample_ot_path_velocity_is_x1_minus_x0` → `test_sample_path_velocity_is_x1_minus_x0`
- `test_sample_ot_path_shapes` → `test_sample_path_shapes`

Update each renamed test's body to call `sample_path(...)` instead of `sample_ot_path(...)`.

Add these new tests at the end of the file:

```python
def test_sample_path_stochastic_x_t_differs_from_deterministic():
    """With nonzero sigma, x_t must differ from the noiseless interpolant."""
    import jax
    key = jax.random.PRNGKey(42)
    x0 = jnp.ones((2, 1, 4, 4)) * 2.0
    x1 = jnp.ones((2, 1, 4, 4)) * 5.0
    t = jnp.array([0.3, 0.7])
    x_t_det, _ = sample_path(x0, x1, t)
    x_t_stoch, _ = sample_path(x0, x1, t, sigma_0=0.1, sigma_1=0.1, key=key)
    assert not jnp.allclose(x_t_det, x_t_stoch)


def test_sample_path_stochastic_velocity_unchanged():
    """Velocity u_t must equal x1 - x0 regardless of sigma values."""
    import jax
    key = jax.random.PRNGKey(7)
    x0 = jnp.ones((2, 1, 4, 4)) * 2.0
    x1 = jnp.ones((2, 1, 4, 4)) * 5.0
    t = jnp.array([0.3, 0.7])
    _, u_t = sample_path(x0, x1, t, sigma_0=0.5, sigma_1=0.2, key=key)
    assert jnp.allclose(u_t, x1 - x0)


def test_sample_path_zero_sigma_matches_deterministic():
    """sigma_0=0, sigma_1=0 with a key provided must give the same result as no key."""
    import jax
    key = jax.random.PRNGKey(0)
    x0 = jnp.ones((2, 1, 4, 4)) * 2.0
    x1 = jnp.ones((2, 1, 4, 4)) * 5.0
    t = jnp.array([0.3, 0.7])
    x_t_a, u_t_a = sample_path(x0, x1, t)
    x_t_b, u_t_b = sample_path(x0, x1, t, sigma_0=0.0, sigma_1=0.0, key=key)
    assert jnp.allclose(x_t_a, x_t_b)
    assert jnp.allclose(u_t_a, u_t_b)
```

- [ ] **Step 2: Run tests to verify the new ones fail**

```bash
pytest tests/flow/test_otfm.py -v -k "stochastic or zero_sigma"
```

Expected: `ImportError` or `AttributeError` — `sample_path` doesn't exist yet.

- [ ] **Step 3: Rename and update `sample_ot_path` in `otfm.py`**

Replace the `sample_ot_path` function entirely with:

```python
def sample_path(
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    t: jnp.ndarray,
    sigma_0: float = 0.0,
    sigma_1: float = 0.0,
    key: jax.Array | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Construct the linear interpolant and target velocity, with optional noise.

    Args:
        x0: shape (B, C, H, W) — noise samples (already coupled to x1).
        x1: shape (B, C, H, W) — data samples.
        t:  shape (B,) — per-sample time values in [0, 1].
        sigma_0: Noise std at t=0. Default 0 (deterministic).
        sigma_1: Noise std at t=1. Default 0 (deterministic).
        key: JAX PRNG key required when sigma_0 or sigma_1 is nonzero.

    Returns:
        x_t: Interpolant at time t, optionally perturbed by Gaussian noise.
        u_t: Target velocity (x1 - x0), unchanged by noise.
    """
    t_ = t[:, None, None, None]  # broadcast over (C, H, W)
    x_t = (1.0 - t_) * x0 + t_ * x1
    u_t = x1 - x0
    if sigma_0 != 0.0 or sigma_1 != 0.0:
        sigma_t = (1.0 - t_) * sigma_0 + t_ * sigma_1
        eps = jax.random.normal(key, x0.shape)
        x_t = x_t + sigma_t * eps
    return x_t, u_t
```

Also update the call inside `flow_matching_loss` — change `sample_ot_path(x0, x1, t)` to `sample_path(x0, x1, t)` (no sigma args needed; defaults apply).

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/flow/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flow/otfm.py tests/flow/test_otfm.py
git commit -m "feat: rename sample_ot_path to sample_path, add optional stochastic interpolation"
```

---

### Task 4: Update `flow_matching_loss` signature to expose sigma/key params

**Files:**
- Modify: `src/flow/otfm.py` — update `flow_matching_loss` signature

- [ ] **Step 1: Write the failing test**

Add to `tests/flow/test_otfm.py`:

```python
def test_flow_matching_loss_stochastic_runs():
    """flow_matching_loss must accept sigma_0, sigma_1, and key without error."""
    import jax
    key = jax.random.PRNGKey(99)
    B = 2
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    loss = flow_matching_loss(
        SMALL_MODEL, x0, x1, t, cond, cond_mask,
        sigma_0=0.1, sigma_1=0.1, key=key,
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/flow/test_otfm.py::test_flow_matching_loss_stochastic_runs -v
```

Expected: `TypeError` — unexpected keyword arguments.

- [ ] **Step 3: Update `flow_matching_loss` in `otfm.py`**

Replace the existing `flow_matching_loss` signature and body with:

```python
def flow_matching_loss(
    model,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    t: jnp.ndarray,
    cond: jnp.ndarray,
    cond_mask: jnp.ndarray,
    sigma_0: float = 0.0,
    sigma_1: float = 0.0,
    key: jax.Array | None = None,
) -> jnp.ndarray:
    """Compute the flow matching MSE loss.

    Args:
        model: Velocity-field network accepting ``(t, x_t, cond, cond_mask)``.
        x0:    shape (B, C, H, W) — noise samples, coupled to x1.
        x1:    shape (B, C, H, W) — data samples.
        t:     shape (B,) — per-sample times in [0, 1].
        cond:  shape (B, cond_dim) — conditioning vectors. Pass
            ``jnp.empty((B, 0))`` when the model is unconditional.
        cond_mask: shape (B,) bool — per-sample mask. ``True`` = use
            the real condition; ``False`` = use the null embedding.
        sigma_0: Noise std at t=0 for the stochastic interpolant. Default 0.
        sigma_1: Noise std at t=1 for the stochastic interpolant. Default 0.
        key: JAX PRNG key required when sigma_0 or sigma_1 is nonzero.

    Returns:
        Scalar mean squared error between predicted and target velocities.
    """
    x_t, u_t = sample_path(x0, x1, t, sigma_0=sigma_0, sigma_1=sigma_1, key=key)
    v_t = eqx.filter_vmap(model)(t, x_t, cond, cond_mask)
    return jnp.mean((v_t - u_t) ** 2)
```

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/flow/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flow/otfm.py tests/flow/test_otfm.py
git commit -m "feat: expose sigma_0, sigma_1, key in flow_matching_loss"
```

---

### Task 5: Update all callers — `trainer.py` and `tests/train/test_trainer.py`

**Files:**
- Modify: `src/train/trainer.py` — swap import; rename call site
- Modify: `tests/train/test_trainer.py` — swap import

- [ ] **Step 1: Update `src/train/trainer.py`**

Change line 18 from:

```python
from src.flow.otfm import flow_matching_loss, minibatch_ot_coupling
```

to:

```python
from src.flow.otfm import flow_matching_loss
from src.flow.coupling import ot_coupling
```

Change line 115 (the call site) from:

```python
x0_paired = minibatch_ot_coupling(x0_np, x1_np)
```

to:

```python
x0_paired = ot_coupling(x0_np, x1_np)
```

- [ ] **Step 2: Update `tests/train/test_trainer.py`**

Change line 114 from:

```python
from src.flow.otfm import minibatch_ot_coupling
```

to:

```python
from src.flow.coupling import ot_coupling
```

Then find any usage of `minibatch_ot_coupling` in that file and replace with `ot_coupling`.

- [ ] **Step 3: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS, no import errors anywhere.

- [ ] **Step 4: Commit**

```bash
git add src/train/trainer.py tests/train/test_trainer.py
git commit -m "refactor: update trainer and tests to import from coupling.py"
```

---

## Self-Review

**Spec coverage:**
- `coupling.py` with `independent_coupling` and `ot_coupling` — Task 1 ✓
- Remove `minibatch_ot_coupling` from `otfm.py` — Task 2 ✓
- Rename `sample_ot_path` → `sample_path` with sigma/key params — Task 3 ✓
- Update `flow_matching_loss` signature — Task 4 ✓
- Import updates in `trainer.py` and both test files — Task 5 ✓
- Tests for `independent_coupling` (shape, identity) — Task 1 ✓
- Tests for `sample_path` with nonzero sigmas — Task 3 ✓
- Deterministic path unchanged — Task 3 (`test_sample_path_zero_sigma_matches_deterministic`) ✓
- Velocity left unchanged in stochastic case — Task 3 (`test_sample_path_stochastic_velocity_unchanged`) ✓

**No placeholders found.**

**Type consistency:** `sample_path` defined in Task 3 with `sigma_0`, `sigma_1`, `key` params; `flow_matching_loss` in Task 4 forwards them with matching names. `ot_coupling` defined in Task 1, used in Task 5. All consistent.
