# Dataset Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add smart dataset resolution to `train_model.py` — reuse existing data when only seed/ratios changed, avoiding unnecessary re-downloads.

**Architecture:** A `resolve_dataset()` coordinator in `src/data/pipeline.py` encapsulates three-way logic (exact match / re-split only / full download) for both local and ClearML paths. Hash functions in `src/data/utils.py` determine `processed_dir` and ClearML tags. The resolved path is injected into the dataloader config via `open_dict`.

**Tech Stack:** Python, ClearML, Hydra/OmegaConf, pandas, pytest, unittest.mock

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/data/utils.py` | `compute_download_hash`, `compute_full_hash` |
| Create | `src/data/pipeline.py` | `resolve_dataset()` coordinator (local + ClearML) |
| Create | `tests/data/test_utils.py` | Tests for hash functions |
| Create | `tests/data/test_pipeline.py` | Tests for `resolve_dataset` |
| Modify | `src/tracking.py` | Update `get_dataset_id`; add `get_base_dataset_id`, `create_dataset_version`; update `register_dataset`; remove `_compute_dataset_hash` |
| Modify | `tests/tracking/test_tracking.py` | Replace old dataset API tests with new ones |
| Modify | `configs/data/dataset.yaml` | Add `data_dir`; remove `download` interpolation |
| Modify | `configs/data/download_tng50.yaml` | Flatten to top-level `_target_` + `_partial_: true`; derive `raw_dir` |
| Modify | `configs/data/dataloader.yaml` | Add `data_dir: null`; replace all `processed_dir: null` + transform `data_dir: null` |
| Modify | `configs/config.yaml` | Remove deleted `data@data.split: split` default |
| Modify | `train_model.py` | Use `resolve_dataset`; inject path via `open_dict`; clean up old imports |
| Modify | `src/data/__init__.py` | Export `resolve_dataset` |

---

## Task 1: Hash utilities (`src/data/utils.py`)

**Files:**
- Create: `src/data/utils.py`
- Create: `tests/data/test_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_utils.py
"""Tests for src.data.utils hash functions."""

import pytest
from src.data.utils import compute_download_hash, compute_full_hash


BASE = dict(
    version_ids=[0, 1],
    snapshots=[72, 73],
    bands=["SUBARU_HSC.I"],
    num_files_per_view=50,
)


class TestComputeDownloadHash:

    def test_is_deterministic(self):
        assert compute_download_hash(**BASE) == compute_download_hash(**BASE)

    def test_returns_16_char_hex(self):
        h = compute_download_hash(**BASE)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_order_independent(self):
        h1 = compute_download_hash(
            version_ids=[0, 1], snapshots=[72, 73],
            bands=["B", "A"], num_files_per_view=50,
        )
        h2 = compute_download_hash(
            version_ids=[1, 0], snapshots=[73, 72],
            bands=["A", "B"], num_files_per_view=50,
        )
        assert h1 == h2

    def test_differs_for_different_bands(self):
        h1 = compute_download_hash(**BASE)
        h2 = compute_download_hash(**{**BASE, "bands": ["SUBARU_HSC.R"]})
        assert h1 != h2

    def test_differs_for_different_snapshots(self):
        h1 = compute_download_hash(**BASE)
        h2 = compute_download_hash(**{**BASE, "snapshots": [99]})
        assert h1 != h2

    def test_differs_for_different_num_files(self):
        h1 = compute_download_hash(**BASE)
        h2 = compute_download_hash(**{**BASE, "num_files_per_view": 100})
        assert h1 != h2

    def test_ignores_extra_kwargs(self):
        h1 = compute_download_hash(**BASE)
        h2 = compute_download_hash(**BASE, max_workers=99, batch_size=1000, api_key="secret")
        assert h1 == h2


class TestComputeFullHash:

    RATIOS = {"train": 0.9, "val": 0.05, "test": 0.05}

    def test_is_deterministic(self):
        h1 = compute_full_hash("abc123", seed=42, ratios=self.RATIOS)
        h2 = compute_full_hash("abc123", seed=42, ratios=self.RATIOS)
        assert h1 == h2

    def test_returns_16_char_hex(self):
        h = compute_full_hash("abc123", seed=42, ratios=self.RATIOS)
        assert len(h) == 16

    def test_differs_for_different_seed(self):
        h1 = compute_full_hash("abc123", seed=42, ratios=self.RATIOS)
        h2 = compute_full_hash("abc123", seed=99, ratios=self.RATIOS)
        assert h1 != h2

    def test_differs_for_different_ratios(self):
        h1 = compute_full_hash("abc123", seed=42, ratios=self.RATIOS)
        h2 = compute_full_hash("abc123", seed=42, ratios={"train": 0.8, "val": 0.1, "test": 0.1})
        assert h1 != h2

    def test_differs_for_different_download_hash(self):
        h1 = compute_full_hash("abc123", seed=42, ratios=self.RATIOS)
        h2 = compute_full_hash("def456", seed=42, ratios=self.RATIOS)
        assert h1 != h2
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/data/test_utils.py -v
```

Expected: `ImportError: cannot import name 'compute_download_hash'`

- [ ] **Step 3: Implement `src/data/utils.py`**

```python
"""Dataset configuration hashing utilities."""

import json
import hashlib


def compute_download_hash(
    version_ids,
    snapshots,
    bands,
    num_files_per_view,
    **kwargs,
) -> str:
    """Compute a 16-char SHA-256 hash of download-determining config fields.

    Ignores seed, ratios, max_workers, batch_size, raw_dir, api_key, and
    any other fields not listed above.

    Args:
        version_ids: TNG version integers (order-independent).
        snapshots: Snapshot integers (order-independent).
        bands: Band name strings (order-independent).
        num_files_per_view: Max FITS files per version/snapshot combination.
        **kwargs: Ignored.

    Returns:
        16-character lowercase hex string.
    """
    data = {
        "version_ids": sorted(version_ids),
        "snapshots": sorted(snapshots),
        "bands": sorted(bands),
        "num_files_per_view": num_files_per_view,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def compute_full_hash(download_hash: str, seed: int, ratios: dict) -> str:
    """Compute a 16-char SHA-256 hash of download hash + split configuration.

    Args:
        download_hash: Output of :func:`compute_download_hash`.
        seed: Random seed used for split assignment.
        ratios: Dict mapping split name to fraction (e.g. ``{"train": 0.9}``).

    Returns:
        16-character lowercase hex string.
    """
    data = {"download_hash": download_hash, "seed": seed, "ratios": ratios}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
pytest tests/data/test_utils.py -v
```

Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/utils.py tests/data/test_utils.py
git commit -m "feat: add compute_download_hash and compute_full_hash to src/data/utils.py"
```

---

## Task 2: Update `src/tracking.py` dataset functions

The current `src/tracking.py` has `get_dataset_id` and `register_dataset` with old signatures. The test file references `register_or_get_dataset` and `_compute_dataset_hash` which no longer exist in the implementation — those tests are already broken. This task replaces them all.

**Files:**
- Modify: `src/tracking.py`
- Modify: `tests/tracking/test_tracking.py`

- [ ] **Step 1: Replace dataset-related tests in `tests/tracking/test_tracking.py`**

Delete everything from line 132 onward (the `register_or_get_dataset` and hash tests) and replace with:

```python
# ---------------------------------------------------------------------------
# get_dataset_id
# ---------------------------------------------------------------------------

def test_get_dataset_id_returns_none_when_task_is_none():
    from src.tracking import get_dataset_id
    assert get_dataset_id(None, "TNG50", "abc123") is None


def test_get_dataset_id_queries_with_splits_tag():
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    mock_dataset = MagicMock()
    mock_dataset.id = "found-id"
    with patch("src.tracking.Dataset") as MockDataset:
        MockDataset.get.return_value = mock_dataset
        from src.tracking import get_dataset_id
        result = get_dataset_id(mock_task, "TNG50", "abc123")
    assert result == "found-id"
    MockDataset.get.assert_called_once_with(
        dataset_name="TNG50",
        dataset_project="msd-flow",
        dataset_tags=["splits:abc123"],
    )


def test_get_dataset_id_returns_none_when_not_found():
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    with patch("src.tracking.Dataset") as MockDataset:
        MockDataset.get.side_effect = ValueError("not found")
        from src.tracking import get_dataset_id
        result = get_dataset_id(mock_task, "TNG50", "abc123")
    assert result is None


# ---------------------------------------------------------------------------
# get_base_dataset_id
# ---------------------------------------------------------------------------

def test_get_base_dataset_id_returns_none_when_task_is_none():
    from src.tracking import get_base_dataset_id
    assert get_base_dataset_id(None, "TNG50", "dl_hash") is None


def test_get_base_dataset_id_returns_latest_by_created():
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    with patch("src.tracking.Dataset") as MockDataset:
        MockDataset.list_datasets.return_value = [
            {"id": "old-id", "created": "2026-01-01T00:00:00"},
            {"id": "new-id", "created": "2026-03-01T00:00:00"},
        ]
        from src.tracking import get_base_dataset_id
        result = get_base_dataset_id(mock_task, "TNG50", "dl_hash")
    assert result == "new-id"
    MockDataset.list_datasets.assert_called_once_with(
        dataset_name="TNG50",
        dataset_project="msd-flow",
        tags=["download:dl_hash"],
    )


def test_get_base_dataset_id_returns_none_when_empty():
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    with patch("src.tracking.Dataset") as MockDataset:
        MockDataset.list_datasets.return_value = []
        from src.tracking import get_base_dataset_id
        result = get_base_dataset_id(mock_task, "TNG50", "dl_hash")
    assert result is None


# ---------------------------------------------------------------------------
# register_dataset
# ---------------------------------------------------------------------------

def test_register_dataset_returns_none_when_task_is_none():
    from src.tracking import register_dataset
    assert register_dataset(None, "TNG50", "/data", "dl_hash", "full_hash") is None


def test_register_dataset_creates_with_both_tags(tmp_path):
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    mock_dataset = MagicMock()
    mock_dataset.id = "new-id"
    with patch("src.tracking.Dataset") as MockDataset:
        MockDataset.create.return_value = mock_dataset
        from src.tracking import register_dataset
        result = register_dataset(mock_task, "TNG50", str(tmp_path), "dl_hash", "full_hash")
    assert result == "new-id"
    MockDataset.create.assert_called_once_with(
        dataset_name="TNG50",
        dataset_project="msd-flow",
        dataset_tags=["download:dl_hash", "splits:full_hash"],
    )
    mock_dataset.add_files.assert_called_once_with(str(tmp_path))
    mock_dataset.finalize.assert_called_once()


def test_register_dataset_returns_none_on_exception():
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    with patch("src.tracking.Dataset") as MockDataset:
        MockDataset.create.side_effect = RuntimeError("server error")
        from src.tracking import register_dataset
        result = register_dataset(mock_task, "TNG50", "/data", "dl_hash", "full_hash")
    assert result is None


# ---------------------------------------------------------------------------
# create_dataset_version
# ---------------------------------------------------------------------------

def test_create_dataset_version_returns_none_when_task_is_none():
    from src.tracking import create_dataset_version
    assert create_dataset_version(None, "TNG50", "base-id", "/tmp/meta.csv", "dl", "full") is None


def test_create_dataset_version_creates_child_with_parent_and_tags(tmp_path):
    import pandas as pd
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    mock_dataset = MagicMock()
    mock_dataset.id = "child-id"

    metadata_path = str(tmp_path / "metadata.csv")
    pd.DataFrame([{"filename": "galaxy_00000.npy", "split": "train"}]).to_csv(
        metadata_path, index=False
    )

    with patch("src.tracking.Dataset") as MockDataset:
        MockDataset.create.return_value = mock_dataset
        from src.tracking import create_dataset_version
        result = create_dataset_version(
            mock_task, "TNG50", "base-id", metadata_path, "dl_hash", "full_hash"
        )
    assert result == "child-id"
    MockDataset.create.assert_called_once_with(
        dataset_name="TNG50",
        dataset_project="msd-flow",
        parent_datasets=["base-id"],
        dataset_tags=["download:dl_hash", "splits:full_hash"],
    )
    mock_dataset.add_files.assert_called_once_with(
        metadata_path, local_base_folder=str(tmp_path)
    )
    mock_dataset.finalize.assert_called_once()


def test_create_dataset_version_returns_none_on_exception():
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    with patch("src.tracking.Dataset") as MockDataset:
        MockDataset.create.side_effect = RuntimeError("server error")
        from src.tracking import create_dataset_version
        result = create_dataset_version(
            mock_task, "TNG50", "base-id", "/tmp/meta.csv", "dl", "full"
        )
    assert result is None
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/tracking/test_tracking.py -v -k "dataset"
```

Expected: multiple failures — `get_dataset_id` wrong signature, `get_base_dataset_id` not found, etc.

- [ ] **Step 3: Update `src/tracking.py` — replace dataset functions**

Remove `_compute_dataset_hash`, `get_dataset_id`, `register_dataset` and add the new versions plus `get_base_dataset_id` and `create_dataset_version`. Keep `setup_task`, `get_dataset_path`, `log_metrics`, `log_checkpoint`, `log_samples` unchanged.

Replace the four dataset functions (lines 64–177 in the current file) with:

```python
def get_dataset_id(
    task: Any,
    dataset_name: str,
    full_hash: str,
) -> str | None:
    """Find a ClearML dataset tagged with ``splits:<full_hash>``.

    Args:
        task: Active ClearML Task, or None (no-op).
        dataset_name: Name of the ClearML dataset.
        full_hash: Output of :func:`src.data.utils.compute_full_hash`.

    Returns:
        ClearML dataset ID string, or None if not found.
    """
    if task is None:
        return None
    try:
        dataset_project = task.get_project_name()
        dataset = Dataset.get(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            dataset_tags=[f"splits:{full_hash}"],
        )
        logger.info("Found existing ClearML dataset: %s", dataset.id)
        return dataset.id
    except ValueError:
        logger.info("No ClearML dataset found with splits tag %s", full_hash)
        return None
    except Exception as exc:
        logger.warning("ClearML dataset retrieval failed (%s). Skipping.", exc)
        return None


def get_base_dataset_id(
    task: Any,
    dataset_name: str,
    download_hash: str,
) -> str | None:
    """Find the most recent ClearML dataset tagged with ``download:<download_hash>``.

    Used to locate a base dataset when only split config (seed/ratios) has changed.

    Args:
        task: Active ClearML Task, or None (no-op).
        dataset_name: Name of the ClearML dataset.
        download_hash: Output of :func:`src.data.utils.compute_download_hash`.

    Returns:
        ClearML dataset ID string of the latest matching dataset, or None.
    """
    if task is None:
        return None
    try:
        dataset_project = task.get_project_name()
        datasets = Dataset.list_datasets(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            tags=[f"download:{download_hash}"],
        )
        if not datasets:
            logger.info("No ClearML base dataset found with download tag %s", download_hash)
            return None
        latest = max(datasets, key=lambda d: d.get("created", ""))
        logger.info("Found base ClearML dataset: %s", latest["id"])
        return latest["id"]
    except Exception as exc:
        logger.warning("ClearML base dataset retrieval failed (%s). Skipping.", exc)
        return None


def register_dataset(
    task: Any,
    dataset_name: str,
    processed_dir: str,
    download_hash: str,
    full_hash: str,
) -> str | None:
    """Register a new ClearML dataset from a local processed directory.

    Tags the dataset with both ``download:<download_hash>`` and
    ``splits:<full_hash>`` so it can be found by either hash later.

    Args:
        task: Active ClearML Task, or None (no-op).
        dataset_name: Name for the new ClearML dataset.
        processed_dir: Local directory containing ``.npy`` files and ``metadata.csv``.
        download_hash: Output of :func:`src.data.utils.compute_download_hash`.
        full_hash: Output of :func:`src.data.utils.compute_full_hash`.

    Returns:
        ClearML dataset ID string, or None if registration failed.
    """
    if task is None:
        return None
    try:
        dataset_project = task.get_project_name()
        dataset = Dataset.create(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            dataset_tags=[f"download:{download_hash}", f"splits:{full_hash}"],
        )
        dataset.add_files(processed_dir)
        dataset.finalize()
        logger.info("Registered new dataset: %s", dataset.id)
        return dataset.id
    except Exception as exc:
        logger.warning("ClearML dataset registration failed (%s). Skipping.", exc)
        return None


def create_dataset_version(
    task: Any,
    dataset_name: str,
    base_id: str,
    metadata_csv_path: str,
    download_hash: str,
    full_hash: str,
) -> str | None:
    """Create a child ClearML dataset that overrides only ``metadata.csv``.

    All ``.npy`` files are inherited from ``base_id`` (no re-upload).
    Only the updated ``metadata.csv`` is added to the child.

    Args:
        task: Active ClearML Task, or None (no-op).
        dataset_name: Name for the new ClearML dataset.
        base_id: ClearML dataset ID of the parent dataset.
        metadata_csv_path: Absolute path to the updated ``metadata.csv`` file.
            The file must be in a temporary directory that serves as the
            ``local_base_folder`` so the path inside the dataset is ``metadata.csv``.
        download_hash: Output of :func:`src.data.utils.compute_download_hash`.
        full_hash: Output of :func:`src.data.utils.compute_full_hash`.

    Returns:
        ClearML dataset ID string of the new version, or None if creation failed.
    """
    if task is None:
        return None
    try:
        dataset_project = task.get_project_name()
        dataset = Dataset.create(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            parent_datasets=[base_id],
            dataset_tags=[f"download:{download_hash}", f"splits:{full_hash}"],
        )
        dataset.add_files(
            metadata_csv_path,
            local_base_folder=os.path.dirname(metadata_csv_path),
        )
        dataset.finalize()
        logger.info("Created dataset version: %s (parent: %s)", dataset.id, base_id)
        return dataset.id
    except Exception as exc:
        logger.warning("ClearML dataset version creation failed (%s). Skipping.", exc)
        return None
```

- [ ] **Step 4: Run tests — expect all dataset tests pass**

```bash
pytest tests/tracking/test_tracking.py -v
```

Expected: all pass (setup_task, log_* tests unchanged; new dataset tests pass)

- [ ] **Step 5: Commit**

```bash
git add src/tracking.py tests/tracking/test_tracking.py
git commit -m "refactor: update tracking.py dataset functions for two-hash design"
```

---

## Task 3: Config changes

**Files:**
- Modify: `configs/data/dataset.yaml`
- Modify: `configs/data/download_tng50.yaml`
- Modify: `configs/data/dataloader.yaml`
- Modify: `configs/config.yaml`

No tests — config correctness is validated at runtime by Hydra.

- [ ] **Step 1: Update `configs/data/dataset.yaml`**

Replace the entire file with:

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

- [ ] **Step 2: Update `configs/data/download_tng50.yaml`**

Replace the entire file with:

```yaml
_target_: src.data.download_tng.download_tng_data
_partial_: true
api_key: ${oc.env:TNG_API_KEY,null}
version_ids: [0,1,2,3]
snapshots: ${generate_snapshot_ids:72,20}
num_files_per_view: 50
max_workers: 5
raw_dir: "${data.dataset.data_dir}/raw"
bands: ["SUBARU_HSC.I"]
batch_size: 100
```

Note: `processed_dir` is intentionally absent — it is injected at runtime in `resolve_dataset`.

- [ ] **Step 3: Update `configs/data/dataloader.yaml`**

Add `data_dir: null` as the very first line of the file:

```yaml
data_dir: null
```

Then replace all three `processed_dir: null` entries (under `train_dataset`, `val_dataset`, `test_dataset`) with:

```yaml
processed_dir: ${data.dataloader.data_dir}
```

Also replace the two transform-level `data_dir: null` entries (under `arcsinh_transform.transforms.0` and `post_arcsinh_image_transforms.transforms.0`) with:

```yaml
data_dir: ${data.dataloader.data_dir}
```

- [ ] **Step 4: Remove deleted split config from `configs/config.yaml`**

Remove the line:

```yaml
  - data@data.split: split
```

`configs/data/split.yaml` was deleted; keeping this line causes a Hydra config error at startup.

- [ ] **Step 5: Commit**

```bash
git add configs/data/dataset.yaml configs/data/download_tng50.yaml \
        configs/data/dataloader.yaml configs/config.yaml
git commit -m "feat: restructure data configs — add data_dir, flatten download config, wire dataloader.data_dir"
```

---

## Task 4: `pipeline.py` — local path

**Files:**
- Create: `src/data/pipeline.py`
- Create: `tests/data/test_pipeline.py`

- [ ] **Step 1: Write failing tests for the local path**

```python
# tests/data/test_pipeline.py
"""Tests for src.data.pipeline.resolve_dataset."""

import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, call as mock_call
from omegaconf import OmegaConf

from src.data.utils import compute_download_hash, compute_full_hash


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DOWNLOAD_KWARGS = dict(
    version_ids=[0],
    snapshots=[72],
    bands=["SUBARU_HSC.I"],
    num_files_per_view=50,
    _target_="src.data.download_tng.download_tng_data",
    _partial_=True,
    max_workers=5,
    batch_size=100,
    raw_dir="/data/raw",
    api_key="key",
)

_RATIOS = {"train": 0.9, "val": 0.05, "test": 0.05}


def _cfg():
    return OmegaConf.create(_DOWNLOAD_KWARGS)


def _dl_hash():
    return compute_download_hash(**_DOWNLOAD_KWARGS)


def _full_hash():
    return compute_full_hash(_dl_hash(), seed=42, ratios=_RATIOS)


def _make_metadata(directory):
    """Write a minimal metadata.csv with 10 rows into directory (str or Path)."""
    pd.DataFrame([
        {"filename": f"galaxy_{i:05d}.npy", "fits_name": f"snap_{i}"}
        for i in range(10)
    ]).to_csv(os.path.join(str(directory), "metadata.csv"), index=False)


# ---------------------------------------------------------------------------
# Local path
# ---------------------------------------------------------------------------

class TestResolveDatasetLocal:

    def test_case_a_returns_processed_dir_without_splitting(self, tmp_path):
        """Case A: metadata.csv + matching .splits_hash → return immediately, no split."""
        processed_dir = tmp_path / _dl_hash()
        processed_dir.mkdir()
        _make_metadata(processed_dir)
        (processed_dir / ".splits_hash").write_text(_full_hash())

        with patch("src.data.pipeline.assign_splits") as mock_split:
            from src.data.pipeline import resolve_dataset
            result = resolve_dataset(
                task=None,
                dataset_name="TNG50",
                data_dir=str(tmp_path),
                seed=42,
                ratios=_RATIOS,
                download_cfg=_cfg(),
            )

        mock_split.assert_not_called()
        assert result == str(processed_dir)

    def test_case_b_resplits_when_hash_mismatch(self, tmp_path):
        """Case B: metadata.csv exists but .splits_hash differs → assign_splits called."""
        processed_dir = tmp_path / _dl_hash()
        processed_dir.mkdir()
        _make_metadata(processed_dir)
        (processed_dir / ".splits_hash").write_text("stale_hash")

        from src.data.pipeline import resolve_dataset
        result = resolve_dataset(
            task=None,
            dataset_name="TNG50",
            data_dir=str(tmp_path),
            seed=42,
            ratios=_RATIOS,
            download_cfg=_cfg(),
        )

        df = pd.read_csv(os.path.join(str(processed_dir), "metadata.csv"))
        assert "split" in df.columns
        assert result == str(processed_dir)

    def test_case_b_resplits_when_no_splits_hash_file(self, tmp_path):
        """Case B: metadata.csv exists but no .splits_hash → assign_splits called."""
        processed_dir = tmp_path / _dl_hash()
        processed_dir.mkdir()
        _make_metadata(processed_dir)

        from src.data.pipeline import resolve_dataset
        result = resolve_dataset(
            task=None,
            dataset_name="TNG50",
            data_dir=str(tmp_path),
            seed=42,
            ratios=_RATIOS,
            download_cfg=_cfg(),
        )

        df = pd.read_csv(os.path.join(str(processed_dir), "metadata.csv"))
        assert "split" in df.columns

    def test_case_b_writes_updated_splits_hash(self, tmp_path):
        """Case B: .splits_hash is updated with the new full_hash after re-split."""
        processed_dir = tmp_path / _dl_hash()
        processed_dir.mkdir()
        _make_metadata(processed_dir)
        (processed_dir / ".splits_hash").write_text("stale_hash")

        from src.data.pipeline import resolve_dataset
        resolve_dataset(
            task=None,
            dataset_name="TNG50",
            data_dir=str(tmp_path),
            seed=42,
            ratios=_RATIOS,
            download_cfg=_cfg(),
        )

        stored = (processed_dir / ".splits_hash").read_text().strip()
        assert stored == _full_hash()

    def test_case_c_calls_download_when_no_metadata(self, tmp_path):
        """Case C: no metadata.csv → call(download_cfg)(processed_dir=...) is called."""
        mock_partial = MagicMock()

        def fake_download(processed_dir):
            os.makedirs(processed_dir, exist_ok=True)
            _make_metadata(processed_dir)

        mock_partial.side_effect = fake_download

        with patch("src.data.pipeline.call", return_value=mock_partial) as mock_call_fn:
            from src.data.pipeline import resolve_dataset
            resolve_dataset(
                task=None,
                dataset_name="TNG50",
                data_dir=str(tmp_path),
                seed=42,
                ratios=_RATIOS,
                download_cfg=_cfg(),
            )

        mock_call_fn.assert_called_once()
        mock_partial.assert_called_once()
        _, kwargs = mock_partial.call_args
        assert "processed_dir" in kwargs

    def test_case_c_assigns_splits_after_download(self, tmp_path):
        """Case C: splits are assigned after download."""
        mock_partial = MagicMock()

        def fake_download(processed_dir):
            os.makedirs(processed_dir, exist_ok=True)
            _make_metadata(processed_dir)

        mock_partial.side_effect = fake_download

        with patch("src.data.pipeline.call", return_value=mock_partial):
            from src.data.pipeline import resolve_dataset
            result = resolve_dataset(
                task=None,
                dataset_name="TNG50",
                data_dir=str(tmp_path),
                seed=42,
                ratios=_RATIOS,
                download_cfg=_cfg(),
            )

        df = pd.read_csv(os.path.join(result, "metadata.csv"))
        assert "split" in df.columns

    def test_case_c_raises_when_skip_download(self, tmp_path):
        """Case C: skip_download=True with no data → FileNotFoundError."""
        from src.data.pipeline import resolve_dataset
        with pytest.raises(FileNotFoundError, match="skip_download"):
            resolve_dataset(
                task=None,
                dataset_name="TNG50",
                data_dir=str(tmp_path),
                seed=42,
                ratios=_RATIOS,
                download_cfg=_cfg(),
                skip_download=True,
            )

    def test_processed_dir_is_derived_from_download_hash(self, tmp_path):
        """The returned path is data_dir/<download_hash>."""
        processed_dir = tmp_path / _dl_hash()
        processed_dir.mkdir()
        _make_metadata(processed_dir)
        (processed_dir / ".splits_hash").write_text(_full_hash())

        from src.data.pipeline import resolve_dataset
        result = resolve_dataset(
            task=None,
            dataset_name="TNG50",
            data_dir=str(tmp_path),
            seed=42,
            ratios=_RATIOS,
            download_cfg=_cfg(),
        )
        assert os.path.basename(result) == _dl_hash()
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/data/test_pipeline.py -v -k "Local"
```

Expected: `ImportError: cannot import name 'resolve_dataset'`

- [ ] **Step 3: Implement local path in `src/data/pipeline.py`**

```python
"""Dataset resolution coordinator.

Determines whether to download, re-split, or reuse an existing dataset.
"""

import os
import shutil
import logging
import tempfile
from typing import Any

from omegaconf import OmegaConf
from hydra.utils import call

from src.data.utils import compute_download_hash, compute_full_hash
from src.data.split import assign_splits
from src.tracking import (
    get_dataset_id,
    get_base_dataset_id,
    register_dataset,
    create_dataset_version,
)

logger = logging.getLogger(__name__)

_SPLITS_HASH_FILE = ".splits_hash"


def resolve_dataset(
    task: Any,
    dataset_name: str,
    data_dir: str,
    seed: int,
    ratios: dict,
    download_cfg,
    skip_download: bool = False,
) -> str:
    """Resolve the local path to a processed dataset.

    Checks whether a dataset matching the current config already exists and
    acts accordingly:

    - **Case A (exact match):** Dataset with current download *and* split
      config already exists. Returns path immediately with no work done.
    - **Case B (re-split only):** Dataset with current download config exists
      but splits differ. Re-assigns splits without re-downloading.
    - **Case C (full download):** No matching data found. Downloads, extracts,
      assigns splits, and registers.

    Args:
        task: Active ClearML Task, or ``None`` for local (no-tracking) mode.
        dataset_name: ClearML dataset name (unused in local mode).
        data_dir: Base data directory. ``processed_dir = data_dir/<download_hash>``.
        seed: Random seed for split assignment.
        ratios: Dict mapping split name to fraction (must sum to 1.0).
        download_cfg: Hydra DictConfig with ``_target_`` and ``_partial_: true``.
            Must *not* contain ``processed_dir`` — it is injected at call time.
        skip_download: If ``True``, raise ``FileNotFoundError`` in Case C instead
            of downloading.

    Returns:
        Absolute local path to the resolved ``processed_dir``.
    """
    resolved = OmegaConf.to_container(download_cfg, resolve=True)
    download_hash = compute_download_hash(**resolved)
    full_hash = compute_full_hash(download_hash, seed, ratios)
    processed_dir = os.path.join(data_dir, download_hash)

    if task is None:
        return _resolve_local(
            processed_dir, full_hash, seed, ratios, download_cfg, skip_download
        )
    return _resolve_clearml(
        task, dataset_name, processed_dir, download_hash, full_hash,
        seed, ratios, download_cfg, skip_download,
    )


def _resolve_local(
    processed_dir: str,
    full_hash: str,
    seed: int,
    ratios: dict,
    download_cfg,
    skip_download: bool,
) -> str:
    metadata_path = os.path.join(processed_dir, "metadata.csv")
    splits_hash_path = os.path.join(processed_dir, _SPLITS_HASH_FILE)

    if os.path.exists(metadata_path):
        if os.path.exists(splits_hash_path):
            with open(splits_hash_path) as f:
                stored = f.read().strip()
            if stored == full_hash:
                logger.info("Case A: exact dataset match. Using %s", processed_dir)
                return processed_dir
        # Case B: re-split only
        logger.info("Case B: re-assigning splits in %s", processed_dir)
        assign_splits(processed_dir, seed=seed, ratios=ratios)
        with open(splits_hash_path, "w") as f:
            f.write(full_hash)
        return processed_dir

    # Case C: full download
    if skip_download:
        raise FileNotFoundError(
            f"skip_download=True but no dataset found at {processed_dir}"
        )
    logger.info("Case C: downloading dataset to %s", processed_dir)
    call(download_cfg)(processed_dir=processed_dir)
    assign_splits(processed_dir, seed=seed, ratios=ratios)
    with open(splits_hash_path, "w") as f:
        f.write(full_hash)
    return processed_dir


def _resolve_clearml(
    task: Any,
    dataset_name: str,
    processed_dir: str,
    download_hash: str,
    full_hash: str,
    seed: int,
    ratios: dict,
    download_cfg,
    skip_download: bool,
) -> str:
    from clearml import Dataset

    # Case A: exact match
    exact_id = get_dataset_id(task, dataset_name, full_hash)
    if exact_id:
        logger.info("Case A: exact ClearML dataset match (%s)", exact_id)
        return Dataset.get(dataset_id=exact_id).get_local_copy()

    # Case B: re-split from base
    base_id = get_base_dataset_id(task, dataset_name, download_hash)
    if base_id:
        logger.info("Case B: re-splitting ClearML dataset (base: %s)", base_id)
        base_path = Dataset.get(dataset_id=base_id).get_local_copy()
        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy(os.path.join(base_path, "metadata.csv"), tmp_dir)
            assign_splits(tmp_dir, seed=seed, ratios=ratios)
            new_id = create_dataset_version(
                task, dataset_name, base_id,
                os.path.join(tmp_dir, "metadata.csv"),
                download_hash, full_hash,
            )
        if new_id:
            return Dataset.get(dataset_id=new_id).get_local_copy()
        logger.warning(
            "Dataset versioning failed; using base path with local splits applied"
        )
        assign_splits(base_path, seed=seed, ratios=ratios)
        return base_path

    # Case C: full download
    if skip_download:
        raise FileNotFoundError(
            f"skip_download=True but no ClearML dataset found for "
            f"download_hash={download_hash}"
        )
    logger.info("Case C: downloading and registering new ClearML dataset")
    call(download_cfg)(processed_dir=processed_dir)
    assign_splits(processed_dir, seed=seed, ratios=ratios)
    new_id = register_dataset(
        task, dataset_name, processed_dir, download_hash, full_hash
    )
    if new_id:
        return Dataset.get(dataset_id=new_id).get_local_copy()
    return processed_dir
```

- [ ] **Step 4: Run local tests — expect all pass**

```bash
pytest tests/data/test_pipeline.py -v -k "Local"
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/pipeline.py tests/data/test_pipeline.py
git commit -m "feat: add resolve_dataset local path in src/data/pipeline.py"
```

---

## Task 5: `pipeline.py` — ClearML path tests

**Files:**
- Modify: `tests/data/test_pipeline.py`

- [ ] **Step 1: Append ClearML tests to `tests/data/test_pipeline.py`**

```python
# ---------------------------------------------------------------------------
# ClearML path
# ---------------------------------------------------------------------------

class TestResolveDatasetClearML:

    def test_case_a_returns_clearml_local_copy(self, tmp_path):
        """Case A: exact ClearML dataset found → return get_local_copy()."""
        mock_task = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.get_local_copy.return_value = "/clearml_cache/exact"

        with patch("src.data.pipeline.get_dataset_id", return_value="exact-id"), \
             patch("src.data.pipeline.Dataset") as MockDataset:
            MockDataset.get.return_value = mock_dataset
            from src.data.pipeline import resolve_dataset
            result = resolve_dataset(
                task=mock_task,
                dataset_name="TNG50",
                data_dir=str(tmp_path),
                seed=42,
                ratios=_RATIOS,
                download_cfg=_cfg(),
            )

        assert result == "/clearml_cache/exact"

    def test_case_a_does_not_call_download(self, tmp_path):
        """Case A: no download is triggered."""
        mock_task = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.get_local_copy.return_value = "/clearml_cache/exact"

        with patch("src.data.pipeline.get_dataset_id", return_value="exact-id"), \
             patch("src.data.pipeline.Dataset") as MockDataset, \
             patch("src.data.pipeline.call") as mock_call_fn:
            MockDataset.get.return_value = mock_dataset
            from src.data.pipeline import resolve_dataset
            resolve_dataset(
                task=mock_task,
                dataset_name="TNG50",
                data_dir=str(tmp_path),
                seed=42,
                ratios=_RATIOS,
                download_cfg=_cfg(),
            )

        mock_call_fn.assert_not_called()

    def test_case_b_creates_child_dataset(self, tmp_path):
        """Case B: base ClearML dataset found → create child dataset version."""
        mock_task = MagicMock()

        base_cache = tmp_path / "base_cache"
        base_cache.mkdir()
        _make_metadata(base_cache)

        mock_base = MagicMock()
        mock_base.get_local_copy.return_value = str(base_cache)

        mock_child = MagicMock()
        mock_child.get_local_copy.return_value = "/clearml_cache/child"

        def dataset_get(dataset_id=None, **kwargs):
            return mock_child if dataset_id == "child-id" else mock_base

        with patch("src.data.pipeline.get_dataset_id", return_value=None), \
             patch("src.data.pipeline.get_base_dataset_id", return_value="base-id"), \
             patch("src.data.pipeline.create_dataset_version", return_value="child-id") as mock_version, \
             patch("src.data.pipeline.Dataset") as MockDataset:
            MockDataset.get.side_effect = dataset_get
            from src.data.pipeline import resolve_dataset
            result = resolve_dataset(
                task=mock_task,
                dataset_name="TNG50",
                data_dir=str(tmp_path),
                seed=42,
                ratios=_RATIOS,
                download_cfg=_cfg(),
            )

        mock_version.assert_called_once()
        assert result == "/clearml_cache/child"

    def test_case_b_passes_updated_metadata_to_create_version(self, tmp_path):
        """Case B: the metadata.csv passed to create_dataset_version has split column."""
        mock_task = MagicMock()

        base_cache = tmp_path / "base_cache"
        base_cache.mkdir()
        _make_metadata(base_cache)

        captured = {}

        def fake_create_version(task, name, base_id, metadata_csv_path, dl_hash, full_hash):
            captured["path"] = metadata_csv_path
            return "child-id"

        mock_base = MagicMock()
        mock_base.get_local_copy.return_value = str(base_cache)
        mock_child = MagicMock()
        mock_child.get_local_copy.return_value = "/clearml_cache/child"

        with patch("src.data.pipeline.get_dataset_id", return_value=None), \
             patch("src.data.pipeline.get_base_dataset_id", return_value="base-id"), \
             patch("src.data.pipeline.create_dataset_version", side_effect=fake_create_version), \
             patch("src.data.pipeline.Dataset") as MockDataset:
            MockDataset.get.return_value = mock_child
            MockDataset.get.side_effect = lambda dataset_id=None, **kw: (
                mock_base if dataset_id == "base-id" else mock_child
            )
            from src.data.pipeline import resolve_dataset
            resolve_dataset(
                task=mock_task,
                dataset_name="TNG50",
                data_dir=str(tmp_path),
                seed=42,
                ratios=_RATIOS,
                download_cfg=_cfg(),
            )

        df = pd.read_csv(captured["path"])
        assert "split" in df.columns

    def test_case_c_downloads_and_registers(self, tmp_path):
        """Case C: no ClearML dataset found → download, split, register."""
        mock_task = MagicMock()
        mock_partial = MagicMock()
        mock_new = MagicMock()
        mock_new.get_local_copy.return_value = "/clearml_cache/new"

        expected_processed_dir = os.path.join(str(tmp_path), _dl_hash())

        def fake_download(processed_dir):
            os.makedirs(processed_dir, exist_ok=True)
            _make_metadata(processed_dir)

        mock_partial.side_effect = fake_download

        with patch("src.data.pipeline.get_dataset_id", return_value=None), \
             patch("src.data.pipeline.get_base_dataset_id", return_value=None), \
             patch("src.data.pipeline.register_dataset", return_value="new-id"), \
             patch("src.data.pipeline.call", return_value=mock_partial), \
             patch("src.data.pipeline.Dataset") as MockDataset:
            MockDataset.get.return_value = mock_new
            from src.data.pipeline import resolve_dataset
            result = resolve_dataset(
                task=mock_task,
                dataset_name="TNG50",
                data_dir=str(tmp_path),
                seed=42,
                ratios=_RATIOS,
                download_cfg=_cfg(),
            )

        assert result == "/clearml_cache/new"

    def test_case_c_raises_when_skip_download(self, tmp_path):
        """Case C: skip_download=True → FileNotFoundError."""
        mock_task = MagicMock()
        with patch("src.data.pipeline.get_dataset_id", return_value=None), \
             patch("src.data.pipeline.get_base_dataset_id", return_value=None):
            from src.data.pipeline import resolve_dataset
            with pytest.raises(FileNotFoundError):
                resolve_dataset(
                    task=mock_task,
                    dataset_name="TNG50",
                    data_dir=str(tmp_path),
                    seed=42,
                    ratios=_RATIOS,
                    download_cfg=_cfg(),
                    skip_download=True,
                )
```

- [ ] **Step 2: Run ClearML tests — expect all pass**

```bash
pytest tests/data/test_pipeline.py -v -k "ClearML"
```

Expected: 6 passed

- [ ] **Step 3: Run full pipeline test suite**

```bash
pytest tests/data/test_pipeline.py -v
```

Expected: 15 passed

- [ ] **Step 4: Commit**

```bash
git add tests/data/test_pipeline.py
git commit -m "test: add ClearML path tests for resolve_dataset"
```

---

## Task 6: Wire `resolve_dataset` into `train_model.py`

**Files:**
- Modify: `train_model.py`
- Modify: `src/data/__init__.py`

- [ ] **Step 1: Export `resolve_dataset` from `src/data/__init__.py`**

```python
# src/data/__init__.py

from .split import assign_splits
from .download_tng import download_tng_data
from .pipeline import resolve_dataset

__all__ = [
    "download_tng_data",
    "assign_splits",
    "resolve_dataset",
]
```

- [ ] **Step 2: Rewrite `train_model.py`**

```python
import logging

import hydra
import jax.random as jr

from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf, open_dict

from src.utils import register_all_resolvers
from src.tracking import setup_task
from src.data.pipeline import resolve_dataset


register_all_resolvers()
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):

    # 0. ClearML setup
    task = setup_task(cfg.clearml)

    # 1. Dataset resolution — download / re-split / reuse as needed
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
    val_loader = instantiate(cfg.data.dataloader.val)
    test_loader = instantiate(cfg.data.dataloader.test)

    log.info(f"Initialized train loader with {len(train_loader)} batches.")

    # 4. Seed
    seed = cfg.seed
    rng_key = jr.PRNGKey(seed)

    # 5. Build model
    log.info("--- Step 5: Model Initialization ---")
    model_key, rng_key = jr.split(rng_key)
    model = instantiate(cfg.model)(key=model_key)

    # 6. Train model
    log.info("--- Step 6: Model Training ---")
    train_key, rng_key = jr.split(rng_key)
    trained_model = call(cfg.train)(
        key=train_key,
        model=model,
        dataloader=train_loader,
        val_dataloader=val_loader,
        clearml_task=task,
        sample_fn=instantiate(cfg.train.sample_fn) if cfg.train.sample_fn else None,
        sample_every=cfg.train.sample_every,
        num_samples=cfg.train.num_samples,
        samples_dir=cfg.train.samples_dir,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/train
```

Expected: all pass (train tests may have unrelated dependencies — skip for now)

- [ ] **Step 4: Commit**

```bash
git add train_model.py src/data/__init__.py
git commit -m "feat: wire resolve_dataset into train_model.py; inject data path into dataloader config"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Skip re-download when only seed/ratios changed — Cases A and B in `pipeline.py`
- [x] ClearML: child dataset via `parent_datasets` — `create_dataset_version` + Case B ClearML
- [x] Local: overwrite metadata.csv in-place — Case B local
- [x] `.splits_hash` marker file — Task 4 implementation
- [x] `data_dir` config key — Task 3 `dataset.yaml`
- [x] `processed_dir = data_dir/<download_hash>` — `_resolve_local` and `_resolve_clearml`
- [x] `raw_dir` derived from `data_dir` — Task 3 `download_tng50.yaml`
- [x] `dataloader.data_dir` injection — Task 6 `open_dict`
- [x] Two hashes (download vs full) — Task 1 `utils.py`
- [x] `skip_download` raises FileNotFoundError — tested in Tasks 4 and 5
- [x] Remove `call(cfg.dataset.download)` bug — replaced in Task 6
- [x] Remove deleted `data@data.split: split` from config — Task 3
