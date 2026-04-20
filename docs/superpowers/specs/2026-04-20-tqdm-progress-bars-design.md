# tqdm Progress Bars for Metric Loops

**Date:** 2026-04-20
**Status:** Approved

## Goal

Add tqdm progress bars to `batch_metric_loop` (trainer.py) and `compute_fid_metrics` (metrics.py) so slow evaluation passes give real-time feedback.

## Design

### `batch_metric_loop` (msdflow/train/trainer.py)

- Compute `total` upfront: `num_batches` if > 0, else `len(dataloader)`.
- Replace the `while True` / manual counter pattern with `for _ in tqdm(range(total), ...)`.
- Loop body unchanged.
- Bar config: `desc="Batch metrics"`, `leave=False`, `dynamic_ncols=True`.

### `compute_fid_metrics` (msdflow/train/metrics.py)

- Add `from tqdm import tqdm` import.
- **Real-image pass:** Wrap `val_dataloader` iteration with `tqdm(val_dataloader, desc="FID real", leave=False, dynamic_ncols=True)`. Early-break logic for `n_real` stays inside the loop.
- **Fake-image pass:** Use `tqdm(total=n_samples, desc="FID fake", leave=False, dynamic_ncols=True)` around the generation while-loop, updating by `chunk_size` each iteration.

### Constraints

- No signature changes to either function.
- No new parameters.
- All bars use `leave=False` and `dynamic_ncols=True`, consistent with the existing training loop.
- `tqdm` is already a dependency (used in trainer.py).
