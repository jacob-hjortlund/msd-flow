# tqdm Progress Bars Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tqdm progress bars to `batch_metric_loop` and `compute_fid_metrics` so slow evaluation passes show real-time progress.

**Architecture:** Wrap existing loops in both functions with tqdm. No signature changes. Consistent style (`leave=False`, `dynamic_ncols=True`) with the existing training pbar.

**Tech Stack:** tqdm (already a dependency)

---

### Task 1: Add tqdm to `batch_metric_loop`

**Files:**
- Modify: `msdflow/train/trainer.py:226-253`
- Test: `tests/train/test_trainer.py` (existing tests — no new tests needed, this is a display-only change)

- [ ] **Step 1: Modify `batch_metric_loop` to use tqdm**

Replace the `while True` loop (lines 226–249) with a tqdm-wrapped `for` loop. Compute `total` as `num_batches` if > 0, else `len(dataloader)`.

```python
def batch_metric_loop(
    key: jax.Array,
    ema_model,
    dataloader,
    step_fn: callable,
    coupling: callable,
    time_sampler: callable,
    path_sampler: callable,
    p_uncond: float,
    num_batches: int = 0,
) -> dict:
    # ... docstring unchanged ...
    totals: dict = {}
    n_batches = 0
    total = num_batches if num_batches > 0 else len(dataloader)
    data_iter = iter(dataloader)

    for _ in tqdm(range(total), desc="Batch metrics", leave=False, dynamic_ncols=True):
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        batch_key, key = jax.random.split(key, 2)
        t, x_t, u_t, cond, cond_mask, dropout_keys = prepare_batch(
            batch=batch,
            key=batch_key,
            coupling=coupling,
            time_sampler=time_sampler,
            path_sampler=path_sampler,
            p_uncond=p_uncond,
        )
        results = step_fn(ema_model, x_t, u_t, t, cond, cond_mask, dropout_keys)
        for k, v in results.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n_batches += 1

    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in totals.items()}
```

- [ ] **Step 2: Run existing tests to verify no regression**

Run: `pytest tests/train/test_trainer.py::test_batch_metric_loop_returns_dict_of_floats tests/train/test_trainer.py::test_batch_metric_loop_values_are_finite tests/train/test_trainer.py::test_batch_metric_loop_num_batches_limit tests/train/test_trainer.py::test_batch_metric_loop_returns_mean_not_sum -v`

Expected: All 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add msdflow/train/trainer.py
git commit -m "feat: add tqdm progress bar to batch_metric_loop"
```

---

### Task 2: Add tqdm to `compute_fid_metrics`

**Files:**
- Modify: `msdflow/train/metrics.py:1-7` (add import)
- Modify: `msdflow/train/metrics.py:220-266` (wrap both loops)
- Test: `tests/train/test_fid.py` (existing tests — no new tests needed)

- [ ] **Step 1: Add tqdm import to metrics.py**

Add `from tqdm import tqdm` after the existing imports at the top of the file (after line 6, the `map_coordinates` import).

```python
from tqdm import tqdm
```

- [ ] **Step 2: Wrap the real-image pass with tqdm**

Replace line 226 (`for images, _meta in val_dataloader:`) with a tqdm-wrapped version:

```python
        for images, _meta in tqdm(val_dataloader, desc="FID real", leave=False, dynamic_ncols=True):
```

The rest of the loop body (lines 227–237) stays identical — the `n_real` early-break logic is unchanged.

- [ ] **Step 3: Wrap the fake-image pass with tqdm**

Replace the `while n_generated < n_samples:` loop (lines 258–266) with a tqdm-based pattern:

```python
    pbar = tqdm(total=n_samples, desc="FID fake", leave=False, dynamic_ncols=True)
    while n_generated < n_samples:
        chunk_size = min(gen_batch_size, n_samples - n_generated)
        all_keys = jax.random.split(key, chunk_size + 1)
        key = all_keys[0]
        sub_keys = all_keys[1:]
        fake_images = _generate_fn(sub_keys)
        for acc in accumulators.values():
            acc.update(fake_images)
        n_generated += chunk_size
        pbar.update(chunk_size)
    pbar.close()
```

- [ ] **Step 4: Run existing FID tests to verify no regression**

Run: `pytest tests/train/test_fid.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add msdflow/train/metrics.py
git commit -m "feat: add tqdm progress bars to compute_fid_metrics"
```
