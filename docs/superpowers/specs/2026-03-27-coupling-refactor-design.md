# Coupling Refactor Design

**Date:** 2026-03-27
**Branch:** feature/CFG

## Summary

Extract coupling logic from `otfm.py` into a dedicated `coupling.py` module, and extend `sample_ot_path` (renamed `sample_path`) to support optional Gaussian noise injection during interpolation.

## Scope

- New file: `src/flow/coupling.py`
- Modified file: `src/flow/otfm.py`
- Import updates: `src/train/trainer.py`, `tests/flow/test_otfm.py`, `tests/train/test_trainer.py`

## `src/flow/coupling.py`

Two functions with a uniform interface `(x0: np.ndarray, x1: np.ndarray) -> np.ndarray`:

### `independent_coupling(x0, x1)`

Returns `x0` unchanged. x0 and x1 are treated as independently drawn samples with no pairing. This is the baseline coupling used in vanilla flow matching.

### `ot_coupling(x0, x1)`

Moves the logic from `minibatch_ot_coupling` verbatim. Runs Hungarian algorithm (via `scipy.optimize.linear_sum_assignment`) on the pairwise squared L2 cost matrix to return a permutation of `x0` that minimises total transport cost to `x1`. Runs on NumPy/CPU, outside JAX JIT.

Both functions accept shape `(B, C, H, W)` arrays and return shape `(B, C, H, W)`.

## `src/flow/otfm.py` changes

### Remove `minibatch_ot_coupling`

The function is deleted from `otfm.py`. All callers import from `src.flow.coupling` instead.

### Rename `sample_ot_path` → `sample_path`

Signature:

```python
def sample_path(
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    t: jnp.ndarray,
    sigma_0: float = 0.0,
    sigma_1: float = 0.0,
    key: jax.Array | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
```

**Behaviour:**

- Deterministic path (default): when `sigma_0 == sigma_1 == 0.0`, computes the linear interpolant `x_t = (1 - t) * x0 + t * x1` and returns `(x_t, u_t)` with `u_t = x1 - x0`. Identical to the old `sample_ot_path`; `key` is unused.
- Stochastic path: when either sigma is nonzero, computes `sigma_t = (1 - t) * sigma_0 + t * sigma_1` and perturbs the interpolant: `x_t = (1 - t) * x0 + t * x1 + sigma_t * ε`, where `ε ~ N(0, I)` drawn with `key`. The velocity target is left unchanged: `u_t = x1 - x0`.

**Key handling:** `key` is a required runtime argument when either sigma is nonzero. Callers are responsible for providing a valid JAX PRNG key. When both sigmas are zero the argument is ignored and may be omitted.

### Update `flow_matching_loss`

Adds matching `sigma_0=0.0`, `sigma_1=0.0`, `key=None` params and forwards them to `sample_path`. Default behaviour is unchanged.

## Import updates

| File | Old import | New import |
|------|-----------|------------|
| `src/train/trainer.py` | `from src.flow.otfm import flow_matching_loss, minibatch_ot_coupling` | `from src.flow.otfm import flow_matching_loss` + `from src.flow.coupling import ot_coupling` |
| `tests/flow/test_otfm.py` | `from src.flow.otfm import minibatch_ot_coupling, sample_ot_path` | `from src.flow.coupling import ot_coupling` + `from src.flow.otfm import sample_path` |
| `tests/train/test_trainer.py` | `from src.flow.otfm import minibatch_ot_coupling` | `from src.flow.coupling import ot_coupling` |

## Testing

- Existing OT coupling tests in `test_otfm.py` are updated to call `ot_coupling` from the new module.
- New tests for `independent_coupling`: output shape matches input; returned array is identical to input `x0`.
- New tests for `sample_path` with nonzero sigmas: `x_t` differs from the deterministic interpolant; `u_t` is still `x1 - x0`.
- All existing `sample_ot_path` tests updated to call `sample_path`; deterministic behaviour is verified to be unchanged.

## Non-goals

- The velocity target `u_t` is intentionally left as `x1 - x0` even in the stochastic case (no `(sigma_1 - sigma_0) * ε` correction).
- No changes to `sample.py` or the Hydra config.
