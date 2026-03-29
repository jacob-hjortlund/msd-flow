# Dataset Resolution Design

**Date:** 2026-03-29
**Branch:** feature/clearml
**Status:** Approved

## Overview

Extend the training pipeline to avoid unnecessary re-downloads when only split configuration (seed/ratios) has changed. A new `resolve_dataset()` coordinator encapsulates a three-way decision: reuse an exact existing dataset, re-split existing data, or perform a full download. Both ClearML and local (no-tracking) paths are supported. The resolved local data path is injected into the dataloader config at runtime.

## Goals

- Skip re-download when only `seed` or `ratios` have changed
- ClearML path: create a child dataset version with updated `metadata.csv` only (inherits `.npy` files from parent — no re-upload)
- Local path: overwrite `metadata.csv` in-place with new split assignments; track current splits via a `.splits_hash` marker file
- Inject the resolved dataset path into `cfg.data.dataloader.data_dir` so all three dataloader splits use it
- Remove the `call(cfg.dataset.download)` bug from `train_model.py` and replace with clean coordinator call

## Non-Goals

- Preserving split history in the local (no-ClearML) case
- Supporting multiple simultaneous dataset sources
- Changes to metric/checkpoint logging or the training loop

---

## Architecture

### New Files

| File | Purpose |
|------|---------|
| `src/data/utils.py` | `compute_download_hash` and `compute_full_hash` |
| `src/data/pipeline.py` | `resolve_dataset()` coordinator |

### Modified Files

| File | Change |
|------|--------|
| `configs/data/dataset.yaml` | Add `data_dir`; remove `download` reference |
| `configs/data/download_tng50.yaml` | Flatten to top-level `_target_` + `_partial_: true`; derive `raw_dir` from `data.dataset.data_dir`; remove redundant `dataset_name`/`seed`/`ratios` |
| `configs/data/dataloader.yaml` | Add top-level `data_dir: null`; replace all `processed_dir: null` and transform `data_dir: null` with `${data.dataloader.data_dir}` |
| `src/tracking.py` | Remove `_compute_dataset_hash`; update `get_dataset_id`; add `get_base_dataset_id` and `create_dataset_version`; update `register_dataset` |
| `train_model.py` | Replace data section with `resolve_dataset()` call + `open_dict` injection |

---

## Config

### `configs/data/dataset.yaml`

```yaml
dataset_name: "TNG50"
data_dir: "${hydra:runtime.cwd}/data"
seed: ${seed}
ratios:
  train: 0.9
  val: 0.05
  test: 0.05
skip_download: false
```

### `configs/data/download_tng50.yaml`

```yaml
_target_: src.data.download_tng.download_tng_data
_partial_: true
api_key: ${oc.env:TNG_API_KEY}
version_ids: [0,1,2,3]
snapshots: ${generate_snapshot_ids:72,20}
num_files_per_view: 50
max_workers: 5
raw_dir: "${data.dataset.data_dir}/raw"
bands: ["SUBARU_HSC.I"]
batch_size: 100
```

`processed_dir` is absent — injected at runtime by `resolve_dataset()`.

### `configs/data/dataloader.yaml` (additions/changes)

Add at top level:
```yaml
data_dir: null
```

Replace all occurrences of `processed_dir: null` and transform-level `data_dir: null` with:
```yaml
processed_dir: ${data.dataloader.data_dir}
data_dir: ${data.dataloader.data_dir}
```

`train_model.py` sets `cfg.data.dataloader.data_dir` after `resolve_dataset()` returns.

---

## `src/data/utils.py`

```python
def compute_download_hash(
    version_ids, snapshots, bands, num_files_per_view, **kwargs
) -> str:
    """16-char SHA-256 hash of download-determining config fields.

    Ignores seed, ratios, max_workers, batch_size, raw_dir, api_key.
    """
    data = {
        "version_ids": sorted(version_ids),
        "snapshots": sorted(snapshots),
        "bands": sorted(bands),
        "num_files_per_view": num_files_per_view,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def compute_full_hash(download_hash: str, seed: int, ratios: dict) -> str:
    """16-char SHA-256 hash of download hash + split config."""
    data = {"download_hash": download_hash, "seed": seed, "ratios": ratios}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
```

---

## `src/tracking.py` changes

### Removed

- `_compute_dataset_hash` — superseded by `compute_download_hash` / `compute_full_hash` in `utils.py`

### Updated: `get_dataset_id`

```python
def get_dataset_id(task, dataset_name: str, full_hash: str) -> str | None:
    """Find a ClearML dataset tagged with splits:<full_hash>."""
    # Dataset.get(dataset_name=..., dataset_project=..., dataset_tags=["splits:<full_hash>"])
```

### New: `get_base_dataset_id`

```python
def get_base_dataset_id(task, dataset_name: str, download_hash: str) -> str | None:
    """Find the most recent ClearML dataset tagged with download:<download_hash>."""
    # Dataset.list_datasets(dataset_name=..., tags=["download:<download_hash>"])
    # return latest by creation time, or None
```

### Updated: `register_dataset`

```python
def register_dataset(
    task, dataset_name: str, processed_dir: str,
    download_hash: str, full_hash: str
) -> str | None:
    """Create a new ClearML dataset tagged with both hashes."""
    # tags=["download:<download_hash>", "splits:<full_hash>"]
    # dataset.add_files(processed_dir); dataset.finalize()
```

### New: `create_dataset_version`

```python
def create_dataset_version(
    task, dataset_name: str, base_id: str,
    metadata_csv_path: str,
    download_hash: str, full_hash: str,
) -> str | None:
    """Create a child dataset inheriting .npy files from base_id, overriding metadata.csv.

    Only metadata.csv is uploaded. All .npy files are inherited from the parent.

    Args:
        metadata_csv_path: Path to the updated metadata.csv file (in a temp dir).
    """
    # Dataset.create(parent_datasets=[base_id], tags=["download:<download_hash>", "splits:<full_hash>"])
    # dataset.add_files(metadata_csv_path, local_base_folder=temp_dir)
    # dataset.finalize()
```

---

## `src/data/pipeline.py`

### `resolve_dataset()`

```python
def resolve_dataset(
    task,
    dataset_name: str,
    data_dir: str,
    seed: int,
    ratios: dict,
    download_cfg,        # Hydra DictConfig with _target_ + _partial_: true
    skip_download: bool = False,
) -> str:
    """Resolve the local path to a processed dataset.

    Three cases (checked in order):
    - Case A (exact match): dataset with current download + split config already exists.
      Return path immediately, no work done.
    - Case B (re-split only): dataset with current download config exists but splits differ.
      Re-assign splits. ClearML: create child dataset with updated metadata.csv.
      Local: overwrite metadata.csv in-place, update .splits_hash.
    - Case C (full download): no matching data found.
      Download, extract, assign splits, register.

    Args:
        task: Active ClearML Task, or None for local mode.
        dataset_name: ClearML dataset name.
        data_dir: Base data directory. processed_dir = data_dir/<download_hash>.
        seed: Split seed.
        ratios: Train/val/test ratios dict.
        download_cfg: Hydra partial config for download_tng_data (missing processed_dir).
        skip_download: If True, raise instead of downloading in Case C.

    Returns:
        Local path to the processed dataset directory.
    """
```

#### Local path (task=None)

```
download_hash = compute_download_hash(**resolved_download_cfg)
processed_dir = data_dir / download_hash
full_hash = compute_full_hash(download_hash, seed, ratios)

if metadata.csv exists in processed_dir:
    if .splits_hash == full_hash:               # Case A
        return processed_dir
    assign_splits(processed_dir, seed, ratios)  # Case B
    write full_hash → .splits_hash
    return processed_dir
else:
    if skip_download: raise FileNotFoundError
    call(download_cfg)(processed_dir=processed_dir)   # Case C
    assign_splits(processed_dir, seed, ratios)
    write full_hash → .splits_hash
    return processed_dir
```

#### ClearML path (task≠None)

```
exact_id = get_dataset_id(task, dataset_name, full_hash)
if exact_id:                                            # Case A
    return Dataset.get(exact_id).get_local_copy()

base_id = get_base_dataset_id(task, dataset_name, download_hash)
if base_id:                                             # Case B
    base_path = Dataset.get(base_id).get_local_copy()
    copy base_path/metadata.csv → temp_dir/metadata.csv
    assign_splits(temp_dir, seed, ratios)
    new_id = create_dataset_version(task, dataset_name, base_id,
                                     temp_dir/metadata.csv, download_hash, full_hash)
    return Dataset.get(new_id).get_local_copy()

if skip_download: raise FileNotFoundError               # Case C
call(download_cfg)(processed_dir=processed_dir)
assign_splits(processed_dir, seed, ratios)
new_id = register_dataset(task, dataset_name, processed_dir, download_hash, full_hash)
return Dataset.get(new_id).get_local_copy()
```

---

## `train_model.py` changes

```python
from omegaconf import open_dict
from src.data.pipeline import resolve_dataset

@hydra.main(...)
def main(cfg: DictConfig):

    # 0. ClearML setup
    task = setup_task(cfg.clearml)

    # 1. Dataset resolution
    log.info("--- Step 1: Dataset Resolution ---")
    dataset_cfg = cfg.data.dataset
    dataset_path = resolve_dataset(
        task=task,
        dataset_name=dataset_cfg.dataset_name,
        data_dir=dataset_cfg.data_dir,
        seed=dataset_cfg.seed,
        ratios=OmegaConf.to_container(dataset_cfg.ratios, resolve=True),
        download_cfg=cfg.data.download,
        skip_download=dataset_cfg.skip_download,
    )

    # 2. Inject resolved path into dataloader config
    with open_dict(cfg):
        cfg.data.dataloader.data_dir = dataset_path

    # 3. Build dataloaders
    log.info("--- Step 3: Dataloader Initialization ---")
    train_loader = instantiate(cfg.data.dataloader.train)
    val_loader   = instantiate(cfg.data.dataloader.val)
    test_loader  = instantiate(cfg.data.dataloader.test)
    ...
```

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| `skip_download=true` and no local data | `resolve_dataset` raises `FileNotFoundError` with a clear message |
| ClearML server unreachable | `setup_task` falls back to offline mode (existing behaviour); dataset functions log warnings and return `None` |
| ClearML dataset functions fail | Log warning, return `None`; pipeline falls through to Case C |
| `metadata.csv` missing from ClearML local copy | Treat as Case C (full download) |
| `.splits_hash` absent (local) | Treat as Case B (re-split); file is written after split assignment |

---

## Data Flow Summary

```
config.yaml
  └─ data.dataset  ──► dataset_name, data_dir, seed, ratios, skip_download
  └─ data.download ──► _partial_ download_tng_data (no processed_dir)

train_model.py
  1. resolve_dataset(task, dataset_cfg, download_cfg) → dataset_path
  2. cfg.data.dataloader.data_dir = dataset_path
  3. instantiate(cfg.data.dataloader.train/val/test)
```
