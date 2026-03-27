# Preprocessing Pipeline & Dataset Refactor

## Overview

Refactor `preprocess.py` into modular callable transform classes composable via Hydra configs, and extend `TNG50Dataset` to support metadata return and separate image/metadata transforms. Uses `torchvision.transforms.Compose` for chaining, with all transforms operating on NumPy arrays.

## Motivation

The current `preprocess.py` contains free functions and a monolithic `preprocess_image` that hardcodes the pipeline. This refactor makes each preprocessing step a configurable, composable unit — enabling per-experiment transform configuration via Hydra without code changes. The dataset also needs to return galaxy metadata alongside images to support future conditioning.

## Transform Interface

Every transform is a class with `__init__` for parameters and `__call__(self, img: np.ndarray) -> np.ndarray`. Transforms are composed using `torchvision.transforms.Compose`, which is agnostic to data type — it simply chains callables.

All transforms operate on NumPy arrays throughout. The ML backbone is JAX/Equinox; PyTorch is only the data loader. Staying in NumPy avoids unnecessary NumPy → tensor → NumPy → JAX round-trips.

### Dimensionality contract

All transform classes operate on `(C, H, W)` arrays — matching the `.npy` files produced by the download pipeline. Transforms that need spatial dimensions (e.g., `ClipAndPad`) operate on the last two axes. This differs from the old free functions which assumed 2D `(H, W)` input; the class implementations must handle the channel dimension.

## Transform Classes

All classes live in `src/data/preprocess.py`.

### Scientific Transforms (deterministic, fixed order)

- **`SurfaceBrightnessToNanomaggies(mag_threshold=99.0)`** — Converts surface-brightness (AB mag/pixel) to nanomaggies. Pixels fainter than `mag_threshold` are zeroed.

- **`ClipAndPad(n=512)`** — Pads image to at least `n × n`, then centre-crops to exactly `n × n`.

- **`ArcsinhStretch(scale=1.0)`** — Applies `arcsinh(img / scale)` to compress dynamic range. The `scale` parameter corresponds to the `a` parameter in the existing `arcsinh_stretch` free function (renamed for clarity).

- **`PercentileClip(percentile=99.0)`** — Computes the given percentile value, divides by it, and clips to `[0, 1]`. If the percentile value is zero (e.g., all-zero image), returns the image unchanged (all zeros). This normalizes the image intensity range before linear normalization.

- **`LinearNormalize(norm_min=-1.0, norm_max=1.0)`** — Linearly maps `[0, 1]` to `[norm_min, norm_max]`. The input range is fixed at `[0, 1]` (guaranteed by `PercentileClip`). This is a deliberate simplification over the old `linear_normalize` free function, which accepted arbitrary `data_min`/`data_max` — that flexibility is no longer needed since the pipeline order is fixed.

### Augmentations (stochastic)

- **`RandomHorizontalFlip(p=0.5)`** — Flips image horizontally with probability `p`.

- **`RandomVerticalFlip(p=0.5)`** — Flips image vertically with probability `p`.

- **`RandomRotation90()`** — Randomly applies 0, 1, 2, or 3 quarter-turns (90° rotations). All four outcomes are equally likely.

### Removed

- `preprocess_image` function — replaced by `Compose` + config.
- Existing free functions are replaced by the corresponding classes above.

## TNG50Dataset Changes

### Modified: `src/data/dataset.py`

```python
class TNG50Dataset(Dataset):
    def __init__(
        self,
        processed_dir: str,
        metadata_columns: list[str] | None = None,
        image_transform=None,
        metadata_transform=None,
    ):
```

- `metadata_columns` — list of float column names from `metadata.csv` to return. If `None`, returns `torch.empty(0)` as placeholder.
- `image_transform` — callable applied to the NumPy image array before tensor conversion.
- `metadata_transform` — callable applied to the metadata tensor after extraction.
- Old `transform` parameter is removed, replaced by `image_transform`.

### `__getitem__` return signature

Always returns `(image_tensor, meta_tensor)`:

```python
def __getitem__(self, idx):
    img = np.load(path)
    if self.image_transform:
        img = self.image_transform(img)
    img_tensor = torch.from_numpy(img).float()

    if self.metadata_columns is None:
        return img_tensor, torch.empty(0)

    meta = self.metadata.iloc[idx][self.metadata_columns].values.astype(np.float32)
    meta_tensor = torch.from_numpy(meta)
    if self.metadata_transform:
        meta_tensor = self.metadata_transform(meta_tensor)
    return img_tensor, meta_tensor
```

Downstream always unpacks `images, meta = batch`. Metadata presence can be checked via `meta.numel() == 0`.

## Hydra Configuration

### Modified: `configs/config.yaml`

Add the preprocess config to the defaults list:

```yaml
defaults:
  - data@data.download: download
  - data@data.preprocess: preprocess    # new
  - model@model: unet
  - ...
```

### New: `configs/data/preprocess.yaml`

```yaml
image_transforms:
  _target_: torchvision.transforms.Compose
  transforms:
    - _target_: src.data.preprocess.SurfaceBrightnessToNanomaggies
      mag_threshold: 99.0
    - _target_: src.data.preprocess.ClipAndPad
      n: 512
    - _target_: src.data.preprocess.ArcsinhStretch
      scale: 1.0
    - _target_: src.data.preprocess.PercentileClip
      percentile: 99.0
    - _target_: src.data.preprocess.LinearNormalize
      norm_min: -1.0
      norm_max: 1.0

augmentations:
  _target_: torchvision.transforms.Compose
  transforms:
    - _target_: src.data.preprocess.RandomHorizontalFlip
      p: 0.5
    - _target_: src.data.preprocess.RandomVerticalFlip
      p: 0.5
    - _target_: src.data.preprocess.RandomRotation90

metadata_columns: null
metadata_transform: null
```

### Usage

Dataset and DataLoader construction happens upstream of `train()`, which receives a DataLoader as input. The calling code instantiates transforms and builds the dataset:

```python
preprocess = hydra.utils.instantiate(cfg.data.preprocess.image_transforms)
augment = hydra.utils.instantiate(cfg.data.preprocess.augmentations)
parts = [preprocess] if augment is None else [preprocess, augment]
image_transform = torchvision.transforms.Compose(parts)
dataset = TNG50Dataset(
    processed_dir=cfg.data.download.processed_dir,
    image_transform=image_transform,
    metadata_columns=cfg.data.preprocess.metadata_columns,
    metadata_transform=hydra.utils.instantiate(cfg.data.preprocess.metadata_transform),
)
```

Separating `image_transforms` and `augmentations` in config allows disabling augmentations for validation/inference by overriding `augmentations=null`. The instantiation code handles `None` gracefully.

## Training Loop Impact

### Modified: `src/train/trainer.py`

Current:
```python
x1_np = batch.numpy()
```

Becomes:
```python
images, meta = batch
x1_np = images.numpy()
```

Metadata is unpacked but ignored. No other changes to training logic, loss computation, or checkpointing.

## Test Plan

### `tests/data/test_preprocess.py` — rewrite

- Each transform class: correct output shape and values on known `(C, H, W)` inputs.
- `PercentileClip`: verified percentile computation and [0, 1] output range.
- `PercentileClip`: all-zero image returns all zeros (no division by zero).
- `LinearNormalize`: maps [0, 1] to configured range.
- Augmentations: shape preservation; stochastic behavior (outputs vary over multiple calls).
- End-to-end `Compose` chain with synthetic `(1, H, W)` image.
- Remove `test_preprocess_image_output_in_norm_range` (tests deleted `preprocess_image`).

### `tests/data/test_dataset.py` — extend

- `(image, metadata)` tuple return with `metadata_columns` specified.
- `(image, empty_tensor)` return when `metadata_columns=None`.
- `image_transform` applied to NumPy array before tensor conversion.
- `metadata_transform` applied to metadata tensor.
- Existing tests updated to unpack tuple return signature (including DataLoader integration test, which now asserts `images.shape` and `meta.shape` separately).

### No changes to `tests/data/test_download_tng.py`.
