# Preprocessing Pipeline Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add train/val/test splitting, split-aware transform statistics, PDFNorm fixes, and updated Hydra configs.

**Architecture:** A new `split.py` module assigns splits to `metadata.csv`. `TNG50Dataset` gains a `split` filter. `ArcsinhStretch` and `GlobalNorm` compute TDigests from train-only data. Hydra configs define separate train/val/test datasets.

**Tech Stack:** NumPy, pandas, Hydra/OmegaConf, PyTorch DataLoader, fastdigest TDigest, pytest.

**Spec:** `docs/superpowers/specs/2026-03-25-preprocessing-improvements-design.md`

**Working directory for all commands:** `msd-flow/`

**Branch:** `feature/data`

---

### Task 1: PDFNorm Docstring and Zero-Guard

**Files:**
- Modify: `src/data/preprocess.py:77-84`
- Test: `tests/data/test_preprocess.py`

- [ ] **Step 1: Write failing tests for PDFNorm**

Add a `TestPDFNorm` class to `tests/data/test_preprocess.py`. Add `PDFNorm` to the imports.

```python
from src.data.preprocess import (
    ArcsinhStretch,
    ClipAndPad,
    GlobalNorm,
    LinearNormalize,
    PDFNorm,
    PercentileClip,
    RandomHorizontalFlip,
    RandomRotation90,
    RandomVerticalFlip,
    SurfaceBrightnessToNanomaggies,
)


class TestPDFNorm:
    """Tests for PDFNorm transform."""

    def test_output_sums_to_one(self):
        """Verify output pixel values sum to 1.0."""
        t = PDFNorm()
        img = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        out = t(img)
        np.testing.assert_allclose(np.sum(out), 1.0)

    def test_preserves_shape(self):
        """Verify output shape matches input."""
        t = PDFNorm()
        img = np.ones((3, 32, 32))
        out = t(img)
        assert out.shape == (3, 32, 32)

    def test_known_value(self):
        """Verify division by total sum."""
        t = PDFNorm()
        img = np.array([[[2.0, 8.0]]])
        out = t(img)
        np.testing.assert_allclose(out, [[[0.2, 0.8]]])

    def test_zero_image_returns_unchanged(self):
        """Verify all-zero image returns zeros without error."""
        t = PDFNorm()
        img = np.zeros((1, 4, 4))
        out = t(img)
        np.testing.assert_array_equal(out, 0.0)

    def test_near_zero_image_returns_unchanged(self):
        """Verify near-zero total returns image unchanged."""
        t = PDFNorm()
        img = np.full((1, 4, 4), 1e-35)
        out = t(img)
        np.testing.assert_array_equal(out, img)
```

Insert `TestPDFNorm` after `TestClipAndPad` and before `TestArcsinhStretch`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_preprocess.py::TestPDFNorm -v`
Expected: `test_zero_image_returns_unchanged` and `test_near_zero_image_returns_unchanged` FAIL (division by zero / no guard).

- [ ] **Step 3: Implement PDFNorm fix**

Replace `PDFNorm` class in `src/data/preprocess.py:77-84` with:

```python
class PDFNorm:
    """Normalize image to a probability distribution (pixel sum = 1).

    Divides every pixel by the total sum of the image. If the total is
    below a safety threshold (``1e-30``), the image is returned unchanged
    to avoid numerical explosion.
    """

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply PDF normalization.

        Args:
            img: ``(C, H, W)`` flux array.

        Returns:
            Array whose pixel values sum to 1.0, same shape as input.
            If the total flux is below ``1e-30``, returns the input
            unchanged.
        """
        total = np.sum(img)
        if total < 1e-30:
            return img
        return img / total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_preprocess.py::TestPDFNorm -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/data/preprocess.py tests/data/test_preprocess.py
git commit -m "Add PDFNorm docstring, zero-guard, and tests"
```

---

### Task 2: Alternative Transform Docstring Notes

**Files:**
- Modify: `src/data/preprocess.py:257-309`

- [ ] **Step 1: Update PercentileClip docstring**

Replace the `PercentileClip` docstring (lines 258-266) with:

```python
class PercentileClip:
    """Clip image intensity by percentile value and normalize to [0, 1].

    .. note::
        Alternative transform, not part of the active pipeline.
        The active pipeline uses ``PDFNorm`` + ``GlobalNorm`` instead.

    Computes the given percentile, divides the image by it, and clips
    to ``[0, 1]``. If the percentile value is zero (e.g., all-zero image),
    the image is returned unchanged.

    Args:
        percentile: Percentile value used for normalization.
    """
```

- [ ] **Step 2: Update LinearNormalize docstring**

Replace the `LinearNormalize` docstring (lines 287-294) with:

```python
class LinearNormalize:
    """Linearly map from [0, 1] to [norm_min, norm_max].

    .. note::
        Alternative transform, not part of the active pipeline.
        The active pipeline uses ``PDFNorm`` + ``GlobalNorm`` instead.

    Input is assumed to be in ``[0, 1]`` (guaranteed by ``PercentileClip``).

    Args:
        norm_min: Minimum of the target range.
        norm_max: Maximum of the target range.
    """
```

- [ ] **Step 3: Run existing tests to confirm no regressions**

Run: `pytest tests/data/test_preprocess.py -v`
Expected: All PASSED

- [ ] **Step 4: Commit**

```bash
git add src/data/preprocess.py
git commit -m "Mark PercentileClip and LinearNormalize as alternative transforms"
```

---

### Task 3: Split Script and Config

**Files:**
- Create: `src/data/split.py`
- Create: `configs/data/split.yaml`
- Modify: `configs/config.yaml`
- Test: `tests/data/test_split.py`

- [ ] **Step 1: Write failing tests for assign_splits**

Create `tests/data/test_split.py`:

```python
"""Tests for src.data.split."""

import numpy as np
import pandas as pd
import pytest

from src.data.split import assign_splits


@pytest.fixture
def metadata_dir(tmp_path):
    """Create a directory with a metadata.csv (100 rows, no split column)."""
    records = [
        {"filename": f"galaxy_{i:05d}.npy", "fits_name": f"snap_{i}"}
        for i in range(100)
    ]
    pd.DataFrame(records).to_csv(tmp_path / "metadata.csv", index=False)
    return str(tmp_path)


class TestAssignSplits:
    """Tests for assign_splits function."""

    def test_adds_split_column(self, metadata_dir):
        """Verify split column is added to metadata.csv."""
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df = pd.read_csv(f"{metadata_dir}/metadata.csv")
        assert "split" in df.columns

    def test_correct_proportions(self, metadata_dir):
        """Verify split proportions match requested ratios."""
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df = pd.read_csv(f"{metadata_dir}/metadata.csv")
        counts = df["split"].value_counts()
        assert counts["train"] == 90
        assert counts["val"] == 5
        assert counts["test"] == 5

    def test_all_rows_assigned(self, metadata_dir):
        """Verify every row gets a split assignment."""
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df = pd.read_csv(f"{metadata_dir}/metadata.csv")
        assert df["split"].notna().all()
        assert len(df) == 100

    def test_reproducible_with_seed(self, metadata_dir):
        """Verify same seed produces same split."""
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df1 = pd.read_csv(f"{metadata_dir}/metadata.csv")

        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df2 = pd.read_csv(f"{metadata_dir}/metadata.csv")

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_produces_different_split(self, metadata_dir):
        """Verify different seeds produce different splits."""
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df1 = pd.read_csv(f"{metadata_dir}/metadata.csv")
        splits1 = df1["split"].tolist()

        assign_splits(metadata_dir, seed=99, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df2 = pd.read_csv(f"{metadata_dir}/metadata.csv")
        splits2 = df2["split"].tolist()

        assert splits1 != splits2

    def test_overwrites_existing_split_column(self, metadata_dir):
        """Verify re-running overwrites existing split column safely."""
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.8, "val": 0.1, "test": 0.1})
        df = pd.read_csv(f"{metadata_dir}/metadata.csv")
        counts = df["split"].value_counts()
        assert counts["train"] == 80
        assert counts["val"] == 10
        assert counts["test"] == 10

    def test_preserves_other_columns(self, metadata_dir):
        """Verify non-split columns are preserved unchanged."""
        df_before = pd.read_csv(f"{metadata_dir}/metadata.csv")
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df_after = pd.read_csv(f"{metadata_dir}/metadata.csv")
        pd.testing.assert_frame_equal(
            df_before[["filename", "fits_name"]],
            df_after[["filename", "fits_name"]],
        )

    def test_ratios_must_sum_to_one(self, metadata_dir):
        """Verify ValueError if ratios don't sum to 1."""
        with pytest.raises(ValueError, match="sum to 1"):
            assign_splits(metadata_dir, seed=42, ratios={"train": 0.5, "val": 0.1, "test": 0.1})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.split'`

- [ ] **Step 3: Implement assign_splits**

Create `src/data/split.py`:

```python
"""Train/val/test split assignment for processed galaxy datasets.

Reads ``metadata.csv``, shuffles row indices with a fixed seed, and
writes a ``split`` column back to the CSV. Re-running overwrites the
existing split column.
"""

import os
import logging

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def assign_splits(
    processed_dir: str,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> None:
    """Assign train/val/test splits to metadata rows.

    Shuffles row indices deterministically using ``seed``, then assigns
    each row to a split based on ``ratios``. The ``split`` column is
    written (or overwritten) in ``metadata.csv``.

    Args:
        processed_dir: Path to directory containing ``metadata.csv``.
        seed: Random seed for reproducible shuffling.
        ratios: Mapping of split name to fraction (must sum to 1.0).
            Defaults to ``{"train": 0.9, "val": 0.05, "test": 0.05}``.

    Raises:
        ValueError: If ratios do not sum to 1.0 (within tolerance).
    """
    if ratios is None:
        ratios = {"train": 0.9, "val": 0.05, "test": 0.05}

    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.6f}: {ratios}"
        )

    csv_path = os.path.join(processed_dir, "metadata.csv")
    df = pd.read_csv(csv_path)
    n = len(df)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    splits = np.empty(n, dtype=object)
    start = 0
    split_names = list(ratios.keys())
    for i, name in enumerate(split_names):
        if i == len(split_names) - 1:
            # Last split gets all remaining rows (avoids rounding gaps)
            end = n
        else:
            end = start + round(ratios[name] * n)
        splits[indices[start:end]] = name
        start = end

    df["split"] = splits
    df.to_csv(csv_path, index=False)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Entry point: assign train/val/test splits to metadata."""
    split_cfg = cfg.data.split
    log.info(f"Assigning splits with ratios: {dict(split_cfg.ratios)}")
    assign_splits(
        processed_dir=split_cfg.processed_dir,
        seed=split_cfg.seed,
        ratios=dict(split_cfg.ratios),
    )
    log.info("Split assignment complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_split.py -v`
Expected: 8 PASSED

- [ ] **Step 5: Create Hydra config**

Create `configs/data/split.yaml`:

```yaml
processed_dir: ${hydra:runtime.cwd}/data/processed/hsc_i_band
seed: ${seed}
ratios:
  train: 0.9
  val: 0.05
  test: 0.05
```

Add the split default to `configs/config.yaml`. Insert `- data@data.split: split` after the `data@data.dataset` line:

```yaml
defaults:
  - data@data.download: download
  - data@data.dataset: dataset
  - data@data.split: split
  - model@model: unet
  - flow@flow.otfm: otfm
  - flow@flow.sample: sample
  - train@train: train
  - _self_

seed: 42
work_dir: ${hydra:runtime.cwd}
```

- [ ] **Step 6: Run all tests to confirm no regressions**

Run: `pytest tests/ -v`
Expected: All PASSED

- [ ] **Step 7: Commit**

```bash
git add src/data/split.py tests/data/test_split.py configs/data/split.yaml configs/config.yaml
git commit -m "Add train/val/test split script with Hydra config"
```

---

### Task 4: TNG50Dataset Split Filtering

**Files:**
- Modify: `src/data/dataset.py:28-41`
- Test: `tests/data/test_dataset.py`

- [ ] **Step 1: Write failing tests for split filtering**

First, update the `sample_dataset` fixture in `tests/data/test_dataset.py` to include a `split` column. Modify the existing fixture:

```python
@pytest.fixture
def sample_dataset(tmp_path):
    """Create a minimal processed directory with .npy files and metadata."""
    records = []
    splits = ["train", "train", "train", "val", "test"]
    for i in range(5):
        name = f"galaxy_{i:05d}.npy"
        data = np.random.default_rng(i).random((1, 64, 64)).astype(np.float32)
        np.save(tmp_path / name, data)
        records.append({
            "filename": name,
            "fits_name": f"snap_{i}",
            "band_map": "g",
            "hdr_mass": float(i) * 1.5,
            "hdr_redshift": float(i) * 0.1,
            "split": splits[i],
        })
    pd.DataFrame(records).to_csv(tmp_path / "metadata.csv", index=False)
    return str(tmp_path)
```

Then add split-filtering tests at the end of the file:

```python
def test_dataset_split_filter_train(sample_dataset):
    """Verify split='train' filters to train rows only."""
    ds = TNG50Dataset(sample_dataset, split="train")
    assert len(ds) == 3


def test_dataset_split_filter_val(sample_dataset):
    """Verify split='val' filters to val rows only."""
    ds = TNG50Dataset(sample_dataset, split="val")
    assert len(ds) == 1


def test_dataset_split_filter_test(sample_dataset):
    """Verify split='test' filters to test rows only."""
    ds = TNG50Dataset(sample_dataset, split="test")
    assert len(ds) == 1


def test_dataset_split_none_returns_all(sample_dataset):
    """Verify split=None returns all rows (default)."""
    ds = TNG50Dataset(sample_dataset, split=None)
    assert len(ds) == 5


def test_dataset_split_correct_items(sample_dataset):
    """Verify split filtering returns correct images."""
    ds_all = TNG50Dataset(sample_dataset)
    ds_val = TNG50Dataset(sample_dataset, split="val")
    # Val is index 3 in the original dataset
    img_all, _ = ds_all[3]
    img_val, _ = ds_val[0]
    torch.testing.assert_close(img_all, img_val)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_dataset.py::test_dataset_split_filter_train -v`
Expected: FAIL with `TypeError: TNG50Dataset.__init__() got an unexpected keyword argument 'split'`

- [ ] **Step 3: Implement split filtering in TNG50Dataset**

Modify `src/data/dataset.py`. Update `__init__` signature and body:

```python
class TNG50Dataset(Dataset):
    """Random-access dataset over extracted TNG50 galaxy ``.npy`` files.

    Always returns ``(image_tensor, meta_tensor)`` tuples. When
    ``metadata_columns`` is ``None``, ``meta_tensor`` is ``torch.empty(0)``.

    Args:
        processed_dir: Path to directory containing ``metadata.csv`` and
            ``.npy`` image files.
        split: If set (e.g., ``"train"``), filter to rows where the
            ``split`` column matches. If ``None``, use all rows.
        metadata_columns: List of float column names from ``metadata.csv``
            to return. If ``None``, an empty tensor placeholder is returned.
        image_transform: Optional callable applied to the NumPy image
            array before tensor conversion.
        metadata_transform: Optional callable applied to the metadata
            tensor after extraction.
    """

    def __init__(
        self,
        processed_dir: str,
        split: str | None = None,
        metadata_columns: list[str] | None = None,
        image_transform=None,
        metadata_transform=None,
    ):
        self.processed_dir = processed_dir
        self.split = split
        self.metadata_columns = metadata_columns
        self.image_transform = image_transform
        self.metadata_transform = metadata_transform
        csv_path = os.path.join(processed_dir, "metadata.csv")
        self.metadata = pd.read_csv(csv_path)
        if split is not None:
            self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.filenames = self.metadata["filename"].tolist()
```

The `__len__` and `__getitem__` methods remain unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_dataset.py -v`
Expected: All PASSED (existing tests + 5 new split tests)

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset.py tests/data/test_dataset.py
git commit -m "Add split filtering to TNG50Dataset"
```

---

### Task 5: Split-Aware TDigest in ArcsinhStretch and GlobalNorm

**Files:**
- Modify: `src/data/preprocess.py:87-167` (ArcsinhStretch) and `src/data/preprocess.py:170-254` (GlobalNorm)
- Test: `tests/data/test_preprocess.py`

- [ ] **Step 1: Write failing tests for split-aware TDigest**

Add to `tests/data/test_preprocess.py`, after the existing `TestArcsinhStretch` class. These tests create a temp dataset with a `split` column and verify TDigest only uses train data.

```python
class TestArcsinhStretchSplit:
    """Tests for ArcsinhStretch split-aware TDigest computation."""

    @pytest.fixture
    def split_dataset(self, tmp_path):
        """Create dataset with split column: 2 train, 1 val, 1 test."""
        records = []
        # Train images: uniform value 2.0
        for i in range(2):
            name = f"galaxy_{i:05d}.npy"
            np.save(tmp_path / name, np.full((1, 4, 4), 2.0))
            records.append({"filename": name, "split": "train"})
        # Val image: uniform value 100.0 (would skew percentile if included)
        name = "galaxy_00002.npy"
        np.save(tmp_path / name, np.full((1, 4, 4), 100.0))
        records.append({"filename": name, "split": "val"})
        # Test image: uniform value 200.0
        name = "galaxy_00003.npy"
        np.save(tmp_path / name, np.full((1, 4, 4), 200.0))
        records.append({"filename": name, "split": "test"})
        pd.DataFrame(records).to_csv(tmp_path / "metadata.csv", index=False)
        return str(tmp_path)

    def test_tdigest_uses_train_only(self, split_dataset):
        """Verify TDigest scale comes from train data only."""
        t = ArcsinhStretch(
            scale=None, percentile=50, data_dir=split_dataset, split="train"
        )
        # Train data is all 2.0, so 50th percentile should be 2.0
        np.testing.assert_allclose(t.scale, 2.0)

    def test_tdigest_cache_includes_split(self, split_dataset):
        """Verify TDigest cache file includes split name."""
        ArcsinhStretch(
            scale=None, percentile=50, data_dir=split_dataset, split="train"
        )
        import os
        assert os.path.isfile(os.path.join(split_dataset, "arcsinh_tdigest_train.json"))

    def test_explicit_scale_ignores_split(self):
        """Verify explicit scale doesn't require split parameter."""
        t = ArcsinhStretch(scale=1.0)
        assert t.scale == 1.0


class TestGlobalNormSplit:
    """Tests for GlobalNorm split-aware TDigest computation."""

    @pytest.fixture
    def split_dataset(self, tmp_path):
        """Create dataset with split column: 2 train, 1 val."""
        records = []
        # Train images: values in [0, 1]
        for i in range(2):
            name = f"galaxy_{i:05d}.npy"
            np.save(tmp_path / name, np.full((1, 4, 4), float(i)))
            records.append({"filename": name, "split": "train"})
        # Val image: value 1000.0 (would change global max if included)
        name = "galaxy_00002.npy"
        np.save(tmp_path / name, np.full((1, 4, 4), 1000.0))
        records.append({"filename": name, "split": "val"})
        pd.DataFrame(records).to_csv(tmp_path / "metadata.csv", index=False)
        return str(tmp_path)

    def test_tdigest_uses_train_only(self, split_dataset):
        """Verify global bounds come from train data only."""
        t = GlobalNorm(
            global_min=None, global_max=None,
            data_dir=split_dataset, percentile=50, split="train"
        )
        # Train data: galaxy_0 is all 0.0, galaxy_1 is all 1.0
        np.testing.assert_allclose(t.global_min, 0.0)
        np.testing.assert_allclose(t.global_max, 1.0)

    def test_tdigest_cache_includes_split(self, split_dataset):
        """Verify TDigest cache file includes split name."""
        GlobalNorm(
            global_min=None, global_max=None,
            data_dir=split_dataset, percentile=50, split="train"
        )
        import os
        assert os.path.isfile(
            os.path.join(split_dataset, "global_norm_tdigest_50_train.json")
        )

    def test_explicit_bounds_ignores_split(self):
        """Verify explicit global bounds don't require split parameter."""
        t = GlobalNorm(global_min=0.0, global_max=1.0)
        assert t.global_min == 0.0
        assert t.global_max == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_preprocess.py::TestArcsinhStretchSplit -v`
Expected: FAIL with `TypeError: ArcsinhStretch.__init__() got an unexpected keyword argument 'split'`

- [ ] **Step 3: Add split parameter to ArcsinhStretch**

Modify `ArcsinhStretch.__init__` in `src/data/preprocess.py`. Add `split: str | None = "train"` parameter. Update `_build_tdigest` to filter by split. Update cache filename to include split.

Updated `__init__` signature:

```python
def __init__(
    self,
    scale: float | None = 1,
    transforms=None,
    percentile=None,
    data_dir: str = None,
    split: str | None = "train",
):
```

Updated `__init__` body — insert `self.split = split` immediately after `self.data_dir = data_dir` (line 120), **before** the `if use_scale:` block. This is critical because `_build_tdigest()` reads `self.split`. Then change the tdigest_path line inside the `if use_percentile:` block:

```python
        self.split = split

        if use_percentile:
            suffix = f"_{split}" if split is not None else ""
            tdigest_path = os.path.join(data_dir, f"arcsinh_tdigest{suffix}.json")
```

Updated `_build_tdigest` — add split filter after reading metadata:

```python
    def _build_tdigest(self):
        csv_path = os.path.join(self.data_dir, "metadata.csv")
        metadata = pd.read_csv(csv_path)
        if self.split is not None:
            metadata = metadata[metadata["split"] == self.split]
        filenames = metadata["filename"].tolist()
```

- [ ] **Step 4: Add split parameter to GlobalNorm**

Modify `GlobalNorm.__init__` in `src/data/preprocess.py`. Add `split: str | None = "train"` parameter. Update `_build_tdigest` to filter by split. Update cache filename to include split.

Updated `__init__` signature:

```python
def __init__(
    self,
    global_min: float | None = None,
    global_max: float | None = None,
    norm_min: float = -1.0,
    norm_max: float = 1.0,
    transforms=None,
    percentile=None,
    data_dir: str = None,
    split: str | None = "train",
):
```

Updated `__init__` body — insert `self.split = split` immediately after `self.data_dir = data_dir` (line 193), **before** the `global_value_not_set` check. Then change the tdigest_path line:

```python
        self.split = split

        if global_value_not_set:
            suffix = f"_{split}" if split is not None else ""
            tdigest_path = os.path.join(
                data_dir, f"global_norm_tdigest_{int(percentile)}{suffix}.json"
            )
```

Also update the **existing** format string on the current line 203 from `{percentile:0f}` to `{int(percentile)}` to avoid filenames like `50.000000`.

Updated `_build_tdigest` — add split filter after reading metadata:

```python
    def _build_tdigest(self):
        csv_path = os.path.join(self.data_dir, "metadata.csv")
        metadata = pd.read_csv(csv_path)
        if self.split is not None:
            metadata = metadata[metadata["split"] == self.split]
        filenames = metadata["filename"].tolist()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/data/test_preprocess.py::TestArcsinhStretchSplit tests/data/test_preprocess.py::TestGlobalNormSplit -v`
Expected: 6 PASSED

- [ ] **Step 6: Run all preprocess tests**

Run: `pytest tests/data/test_preprocess.py -v`
Expected: All PASSED

- [ ] **Step 7: Commit**

```bash
git add src/data/preprocess.py tests/data/test_preprocess.py
git commit -m "Add split-aware TDigest to ArcsinhStretch and GlobalNorm"
```

---

### Task 6: Hydra Config Updates

**Files:**
- Modify: `configs/data/dataset.yaml`

- [ ] **Step 1: Update dataset.yaml**

Replace the entire `configs/data/dataset.yaml` with the split-aware version. Key changes: add `split: train` to ArcsinhStretch and GlobalNorm, add `deterministic_transforms`, add `val` and `test` dataset entries, add `split: train` to the train dataset.

```yaml
pre_arcsinh_image_transforms:
  _target_: torchvision.transforms.Compose
  transforms:
    - _target_: src.data.preprocess.SurfaceBrightnessToNanomaggies
      mag_threshold: 99.0
    - _target_: src.data.preprocess.ClipAndPad
      n: 512
    - _target_: src.data.preprocess.PDFNorm

arcsinh_transform:
  _target_: torchvision.transforms.Compose
  transforms:
    - _target_: src.data.preprocess.ArcsinhStretch
      scale: null
      transforms: ${data.dataset.pre_arcsinh_image_transforms}
      percentile: 10
      data_dir: ${data.dataset.train.processed_dir}
      split: train

post_arcsinh_image_transforms:
  _target_: torchvision.transforms.Compose
  transforms:
    - _target_: src.data.preprocess.GlobalNorm
      global_min: 0.0
      global_max: null
      norm_min: -1.0
      norm_max: 1.0
      data_dir: ${data.dataset.train.processed_dir}
      percentile: ${data.dataset.arcsinh_transform.transforms.0.percentile}
      split: train
      transforms:
        _target_ : torchvision.transforms.Compose
        transforms:
          - ${data.dataset.pre_arcsinh_image_transforms}
          - ${data.dataset.arcsinh_transform}

augmentations:
  _target_: torchvision.transforms.Compose
  transforms:
    - _target_: src.data.preprocess.RandomHorizontalFlip
      p: 0.5
    - _target_: src.data.preprocess.RandomVerticalFlip
      p: 0.5
    - _target_: src.data.preprocess.RandomRotation90

deterministic_transforms:
  _target_: torchvision.transforms.Compose
  transforms:
    - ${data.dataset.pre_arcsinh_image_transforms}
    - ${data.dataset.arcsinh_transform}
    - ${data.dataset.post_arcsinh_image_transforms}

train:
  _target_: src.data.dataset.TNG50Dataset
  processed_dir: ${hydra:runtime.cwd}/data/processed/hsc_i_band
  split: train
  image_transform:
    _target_: torchvision.transforms.Compose
    transforms:
      - ${data.dataset.pre_arcsinh_image_transforms}
      - ${data.dataset.arcsinh_transform}
      - ${data.dataset.post_arcsinh_image_transforms}
      - ${data.dataset.augmentations}
  metadata_columns: null
  metadata_transform: null

val:
  _target_: src.data.dataset.TNG50Dataset
  processed_dir: ${hydra:runtime.cwd}/data/processed/hsc_i_band
  split: val
  image_transform: ${data.dataset.deterministic_transforms}
  metadata_columns: null
  metadata_transform: null

test:
  _target_: src.data.dataset.TNG50Dataset
  processed_dir: ${hydra:runtime.cwd}/data/processed/hsc_i_band
  split: test
  image_transform: ${data.dataset.deterministic_transforms}
  metadata_columns: null
  metadata_transform: null
```

- [ ] **Step 2: Run all tests to confirm no regressions**

Run: `pytest tests/ -v`
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add configs/data/dataset.yaml
git commit -m "Add val/test datasets and split-aware transforms to Hydra config"
```

---

### Task 7: Update End-to-End Test

**Files:**
- Modify: `tests/data/test_preprocess.py:279-300`

- [ ] **Step 1: Update TestComposeEndToEnd to use active pipeline**

Replace the `TestComposeEndToEnd` class to use `PDFNorm` + `ArcsinhStretch` + `GlobalNorm` instead of `PercentileClip` + `LinearNormalize`:

```python
class TestComposeEndToEnd:
    """End-to-end test for the active transform pipeline."""

    def test_full_pipeline_shape_and_range(self):
        """Verify the active pipeline produces correct shape and range."""
        pipeline = Compose([
            SurfaceBrightnessToNanomaggies(),
            ClipAndPad(n=128),
            PDFNorm(),
            ArcsinhStretch(scale=1.0),
            GlobalNorm(global_min=0.0, global_max=1.0, norm_min=-1.0, norm_max=1.0),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation90(),
        ])
        # Synthetic surface-brightness image
        img = np.full((1, 100, 100), 22.0)
        out = pipeline(img)
        assert out.shape == (1, 128, 128)
        assert out.min() >= -1.0 - 1e-7
        assert out.max() <= 1.0 + 1e-7
```

Update the import block at the top of the file to include `GlobalNorm` and `PDFNorm` if not already present (should already be added in Task 1).

- [ ] **Step 2: Run end-to-end test**

Run: `pytest tests/data/test_preprocess.py::TestComposeEndToEnd -v`
Expected: PASSED

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASSED

- [ ] **Step 4: Commit**

```bash
git add tests/data/test_preprocess.py
git commit -m "Update end-to-end test to use active pipeline transforms"
```
