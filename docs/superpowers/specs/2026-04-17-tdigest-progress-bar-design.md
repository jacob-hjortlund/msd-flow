# TDigest Progress Bar Design

**Date:** 2026-04-17
**Status:** Approved

## Problem

`build_tdigest` in `msdflow/data/preprocess.py` processes potentially thousands of `.npy` files to build a TDigest, but provides no progress feedback. This affects both the sequential (`n_workers=0`) and parallel (`n_workers>0`) code paths.

## Solution

Add per-file `tqdm` progress bars to both code paths in `build_tdigest` by refactoring the worker function from chunk-based to single-file-based processing.

## Architecture

### Worker Function

Replace `_worker_build_tdigest` (processes a chunk of files, returns one TDigest) with `_worker_single_file` (processes one file, returns one TDigest).

**Signature:**
```python
def _worker_single_file(args: tuple) -> TDigest:
    """Build a TDigest from a single .npy file.

    Args:
        args: Tuple of (data_dir, filename, transforms, pixel_filter).

    Returns:
        Fitted TDigest for this file.
    """
```

The old `_worker_build_tdigest` is removed entirely.

### `build_tdigest` Function

The public signature is unchanged:

```python
def build_tdigest(
    data_dir: str,
    filenames: list[str],
    transforms,
    pixel_filter,
    n_workers: int = 0,
) -> TDigest:
```

**Sequential path** (`n_workers <= 0`):
- Loop over files, call `_worker_single_file` per file.
- Merge each result incrementally into a running TDigest.
- Wrap the loop with `tqdm(desc="Building TDigest", total=len(filenames), unit="file")`.

**Parallel path** (`n_workers > 0`):
- Build args list: one `(data_dir, filename, transforms, pixel_filter)` tuple per file.
- Use `pool.imap_unordered(_worker_single_file, args)`.
- Wrap the iterator with `tqdm(total=len(filenames), desc="Building TDigest", unit="file")`.
- Merge results incrementally as they arrive from workers.

### tqdm Configuration

| Parameter | Value |
|-----------|-------|
| `desc` | `"Building TDigest"` |
| `unit` | `"file"` |
| `total` | `len(filenames)` |

### Data Flow

```
files → _worker_single_file(file) → TDigest per file
     → tqdm tracks each completion
     → incremental merge into single result TDigest
```

## Callers

Both `ArcsinhStretch._build_tdigest` and `GlobalNorm._build_tdigest` call `build_tdigest`. Neither requires changes since the public interface is unchanged.

## Dependencies

- `tqdm` — added as an import in `preprocess.py`. Already present in the project environment.

## Trade-offs

- **More TDigest objects**: One per file instead of one per chunk. TDigest merge is O(compression), which is negligible compared to `.npy` file I/O.
- **`imap_unordered` vs `pool.map`**: Slightly different semantics (results arrive out of order), but order doesn't matter for merging TDigests.
- **Always-on progress bar**: The bar is always displayed — no opt-out parameter. This keeps the interface simple and matches the use case (long-running dataset preprocessing).

## Testing

- Existing tests for `build_tdigest` should pass without modification (the output TDigest is identical).
- Visual verification: run with a real dataset to confirm tqdm bar renders correctly in both sequential and parallel modes.
