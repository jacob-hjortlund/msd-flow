# Preprocessing Pipeline Improvements Design

## Context

The preprocessing pipeline trains a flow matching model on TNG50 galaxy images
mapped to [-1, 1]. At inference, generated images must be invertible back to
normalised flux (sum of pixels = 1) so that realistic images are constructed as
`y = A * img` with user-defined total flux `A`.

The current pipeline is:

```
SurfaceBrightnessToNanomaggies -> ClipAndPad(512) -> PDFNorm
    -> ArcsinhStretch(percentile) -> GlobalNorm([-1,1])
    -> augmentations
```

Invertibility at inference:

```
generated [-1,1] -> inv_GlobalNorm -> inv_ArcsinhStretch -> img (sum ~= 1)
    -> y = A * img
```

GlobalNorm and ArcsinhStretch are both invertible (linear map and sinh
respectively). PDFNorm before the nonlinear stretch ensures sum=1 is recovered
after inverting ArcsinhStretch.

## Changes

### 1. Train/Val/Test Split Script

New module `src/data/split.py` with function `assign_splits()`:

- Reads `metadata.csv` from `processed_dir`.
- Shuffles row indices using a configurable `seed` (default: `${seed}` from
  root config, i.e. 42).
- Assigns `"train"`, `"val"`, or `"test"` to each row based on configurable
  ratios (default: 90/5/5).
- Writes the `split` column back to `metadata.csv`. Overwrites any existing
  `split` column so re-running is safe.
- Runnable via `python -m src.data.split_dataset` with Hydra config.

New config `configs/data/split.yaml`:

```yaml
processed_dir: ${hydra:runtime.cwd}/data/processed/hsc_i_band
seed: ${seed}
ratios:
  train: 0.9
  val: 0.05
  test: 0.05
```

Added to `configs/config.yaml` defaults as `data@data.split: split`.

### 2. TNG50Dataset Split Filtering

Add optional `split` parameter (default: `None`) to `TNG50Dataset.__init__`:

- When set (e.g., `"train"`), filter `self.metadata` and `self.filenames` to
  rows where `metadata["split"] == split`.
- When `None`, use all rows (backwards compatible).

### 3. ArcsinhStretch and GlobalNorm Split-Aware TDigests

Both classes gain a `split` parameter (default: `"train"`):

- In `_build_tdigest()`, after reading `metadata.csv`, filter rows to
  `metadata["split"] == self.split` before iterating over filenames.
- TDigest cache filenames include the split name, e.g.,
  `arcsinh_tdigest_train.json`, `global_norm_tdigest_10_train.json`.
- If `split` is `None`, use all rows (backwards compatible for explicit
  scale/bounds usage).

The Hydra config passes `split: train` to both transforms, ensuring statistics
are derived from training data only.

### 4. PDFNorm Fixes

- Add Google-style docstring.
- Add zero-guard: if `np.sum(img) == 0`, return image unchanged (same pattern
  as `PercentileClip`).

### 5. Alternative Transform Docstrings

`PercentileClip` and `LinearNormalize` remain in `preprocess.py` as alternative
transforms. Their docstrings are updated to note that they are not part of the
active pipeline (replaced by `PDFNorm` + `GlobalNorm`).

### 6. Hydra Config Updates

`configs/data/dataset.yaml` expands to support split-aware datasets:

- `deterministic_transforms`: Compose of pre_arcsinh + arcsinh + post_arcsinh
  (no augmentations), reused by val/test.
- `train`: `TNG50Dataset` with `split: train`, full transform chain including
  augmentations.
- `val`: `TNG50Dataset` with `split: val`, deterministic transforms only.
- `test`: `TNG50Dataset` with `split: test`, deterministic transforms only.
- `arcsinh_transform` and `post_arcsinh_image_transforms` both receive
  `split: train`.

All three datasets point to the same `processed_dir`, differentiated by
the `split` parameter.

### 7. Test Updates

- New tests for `PDFNorm` (known values, zero-guard, shape preservation).
- New tests for `assign_splits()` (correct proportions, reproducibility with
  seed, split column written, re-run overwrites safely).
- New tests for split filtering in `TNG50Dataset`.
- New tests for split filtering in ArcsinhStretch/GlobalNorm TDigest
  computation.
- Update `TestComposeEndToEnd` to use the active pipeline (`PDFNorm` +
  `ArcsinhStretch` + `GlobalNorm` instead of `PercentileClip` +
  `LinearNormalize`).

## Files Modified

| File | Change |
|------|--------|
| `src/data/split.py` | New: `assign_splits()` + Hydra entrypoint |
| `src/data/split_dataset.py` | New: `__main__` entrypoint for `python -m` |
| `src/data/dataset.py` | Add `split` parameter to `TNG50Dataset` |
| `src/data/preprocess.py` | `split` param on ArcsinhStretch/GlobalNorm; PDFNorm docstring + zero-guard; alternative transform docstring notes |
| `configs/data/split.yaml` | New split config |
| `configs/data/dataset.yaml` | Val/test datasets, deterministic transforms, split params |
| `configs/config.yaml` | Add split to defaults |
| `tests/data/test_preprocess.py` | PDFNorm tests, updated end-to-end test |
| `tests/data/test_dataset.py` | Split filtering tests |
| `tests/data/test_split.py` | New: `assign_splits()` tests |
