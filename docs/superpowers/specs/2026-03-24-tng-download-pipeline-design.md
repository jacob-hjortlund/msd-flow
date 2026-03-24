# TNG50 Download and Processing Pipeline

## Overview

Refactor `download_tng.py` into a batched download-extract-cleanup pipeline that downloads TNG50 galaxy FITS files, extracts multi-band images and full FITS header metadata, saves them as individual `.npy` files with a `metadata.csv` index, and deletes FITS files after extraction. Supports resumption and configurable batch sizes. A companion `TNG50Dataset` provides PyTorch DataLoader integration.

## Motivation

The full TNG50 dataset will not fit on disk as raw FITS files. We need a pipeline that processes data in batches — downloading a fixed number of galaxies, extracting the useful data into a compact format, and freeing disk space before the next batch. The output format must support efficient random access for PyTorch DataLoader with prefetching (`num_workers > 0`).

## Output Format

Individual `.npy` files were chosen over HDF5 for PyTorch DataLoader compatibility. HDF5 file handles are not fork-safe, requiring lazy-open workarounds that limit parallel prefetching. Individual files allow fully independent reads across DataLoader workers with no shared state.

### Disk layout

```
data/processed/<run_name>/
  ├── metadata.csv
  ├── galaxy_00000.npy    # shape: (C, H, W)
  ├── galaxy_00001.npy
  └── ...
```

- Each `.npy` file contains a single galaxy image as a `(C, H, W)` float array, where `C = len(bands)`.
- `metadata.csv` contains one row per galaxy with columns:
  - `filename` — the `.npy` filename (e.g., `galaxy_00000.npy`)
  - `fits_name` — original FITS filename without extension (used for resumption)
  - `band_map` — comma-separated band names in channel order (e.g., `g,r,i`)
  - All FITS header fields from the first band's extension (flattened into columns)

## Module Changes

### `src/data/download_tng.py` (refactored)

#### Moved from `preprocess.py`

- **`load_fits(filename, bands)`** — generalized to accept a list of band names. Returns `(np.ndarray, dict)` where the array is `(C, H, W)` (bands stacked in the order given) and the dict is the full FITS header from the first band's extension. All bands are assumed to have identical spatial dimensions. Raises `ValueError` if any band is not found.

The function is removed from `preprocess.py` entirely (no downstream code depends on it there).

#### Existing functions

- **`extract_tng_urls(version_ids, snap_ids, headers, N=0)`** — unchanged. Returns list of FITS download URLs.
- **`download_tng_fits_file(url, save_dir, headers, max_retries=4, timeout_base=3)`** — **return type changed**: returns the file path (str) on success, or `None` on failure (previously returned a status string). Internal retry/backoff logic unchanged.

#### New functions

- **`fits_name_from_url(url) -> str`** — extracts the FITS identifier from a URL by parsing the URL path segments (snapshot ID, subhalo ID, filename). Returns the filename without extension. This is the single source of truth for URL-to-filename mapping; `download_tng_fits_file` should call this function internally rather than duplicating the parsing logic.

- **`get_existing_ids(processed_dir) -> set[str]`** — reads `metadata.csv` in `processed_dir` (if it exists, using `on_bad_lines='skip'` to handle truncated CSVs from crashes) and returns the set of `fits_name` values. Used for resumption filtering.

- **`download_batch(urls, raw_dir, headers, max_workers) -> list[str]`** — downloads a list of URLs in parallel via `ThreadPoolExecutor`. Filters results from `download_tng_fits_file`, collecting non-`None` return values. Returns list of successfully downloaded file paths.

- **`extract_batch(fits_paths, bands, processed_dir, start_idx) -> list[dict]`** — for each FITS file:
  1. Call `load_fits(path, bands)` to get `(C, H, W)` array and header dict.
  2. On failure (e.g., missing band), log warning and skip. The success counter does not increment.
  3. Save array as `galaxy_{start_idx + success_count:05d}.npy`.
  4. Build metadata dict: `filename`, `fits_name`, `band_map`, plus all header fields.
  5. Increment success counter.
  6. Returns list of metadata dicts for successfully extracted galaxies.

Galaxy filenames are numbered by a running success counter (not by position in `fits_paths`), so there are never numbering gaps.

- **`cleanup_batch(fits_paths) -> None`** — deletes all FITS files in the list unconditionally.

- **`save_metadata(records, processed_dir) -> None`** — appends metadata dicts as rows to `metadata.csv`. Reads existing CSV (if any), concatenates new rows, writes the full result to a temporary file, then atomically renames to replace the original. This ensures the CSV is never left in a partially-written state.

- **`main(cfg)`** — Hydra entry point. Orchestrates:
  1. Fetch all URLs via `extract_tng_urls`.
  2. Load existing IDs via `get_existing_ids` for resumption.
  3. Filter URLs: skip any whose FITS filename (derived from URL) is already in the existing IDs set.
  4. Chunk remaining URLs into batches of `cfg.data.download.batch_size`.
  5. For each batch: `download_batch` -> `extract_batch` -> `save_metadata` -> `cleanup_batch`.

### `src/data/preprocess.py` (modified)

- Remove `load_fits` function entirely.
- No other changes. All other preprocessing functions remain.

### `src/data/dataset.py` (new)

- **`TNG50Dataset(torch.utils.data.Dataset)`**
  - `__init__(self, processed_dir, transform=None)` — reads `metadata.csv` via pandas, builds list of `.npy` file paths. Stores the metadata DataFrame for optional access.
  - `__getitem__(self, idx)` — loads `np.load(path)`, converts to `torch.Tensor`. Applies `transform` if provided. Returns the tensor only (metadata is accessible separately via `self.metadata`).
  - `__len__(self)` — number of rows in metadata.

No preprocessing is done here — transforms are passed in at construction time.

## Configuration

Updated `configs/data/download.yaml`:

```yaml
# Existing
api_key: ${oc.env:TNG_API_KEY}
version_ids: [0, 1, 2, 3]
snapshots: ${generate_snapshot_ids:72,20}
num_files_per_view: 50
max_workers: 5
raw_dir: "${hydra:runtime.cwd}/data/raw"

# New
bands: ["g"]
batch_size: 100
processed_dir: "${hydra:runtime.cwd}/data/processed/g_band"
```

## Resumption

On startup, `main()` determines which galaxies have already been processed:

1. `get_existing_ids(processed_dir)` reads the `fits_name` column from `metadata.csv`.
2. Each URL is mapped to a FITS identifier via `fits_name_from_url(url)`, which uses the same URL-parsing logic as `download_tng_fits_file` (parsing snapshot ID, subhalo ID, and original filename from URL path segments).
3. URLs whose FITS identifier is already in the existing set are skipped.
4. New `.npy` files continue numbering from `start_idx = len(existing_ids)`.

This means: if the process crashes mid-batch, the partial batch's un-recorded files will be re-downloaded and re-extracted on the next run. Successfully extracted files (with metadata rows) are never re-processed.

## Error Handling

- **Download failures:** existing exponential-backoff retry in `download_tng_fits_file`. After max retries, the URL is logged and skipped. The batch continues with successful downloads.
- **Extraction failures:** if `load_fits` fails (e.g., missing band), the file is logged and skipped. No `.npy` or metadata row is created.
- **Cleanup:** all FITS files in a batch are deleted unconditionally after extraction, regardless of whether extraction succeeded. Failed files will be re-downloaded on resumption since they won't appear in `metadata.csv`.
- **Metadata:** written once per batch after extraction but before cleanup. This ordering is critical: metadata must be persisted before FITS files are deleted, so that a crash at any point leaves the system in a resumable state. A crash before the metadata write means the batch is retried on resumption; a crash after the write but before cleanup just leaves harmless FITS files on disk.

## Test Plan

- Unit tests for `load_fits` with multi-band extraction (moved from `test_preprocess.py`).
- Unit tests for `get_existing_ids`, `extract_batch`, `cleanup_batch`, `save_metadata`.
- Integration test for the batch loop with mocked downloads.
- Unit tests for `TNG50Dataset` — length, `__getitem__`, transform application.
- Update existing `test_preprocess.py` to remove `load_fits` tests.
- Update existing `test_download_tng.py` to reflect refactored `main()`.
