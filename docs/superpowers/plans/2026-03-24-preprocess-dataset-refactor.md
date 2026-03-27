# Preprocessing Pipeline & Dataset Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `preprocess.py` into composable transform classes and extend `TNG50Dataset` to return `(image, metadata)` tuples with separate transforms.

**Architecture:** Each preprocessing step becomes a callable class with `__init__`/`__call__`, composed via `torchvision.transforms.Compose` and instantiated through Hydra `_target_` configs. All transforms operate on `(C, H, W)` NumPy arrays. The dataset always returns `(image_tensor, meta_tensor)` tuples.

**Tech Stack:** NumPy, PyTorch (DataLoader only), torchvision.transforms.Compose, Hydra/OmegaConf, pytest.

**Spec:** `docs/superpowers/specs/2026-03-24-preprocess-dataset-refactor-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/data/preprocess.py` | Rewrite | Transform classes: `SurfaceBrightnessToNanomaggies`, `ClipAndPad`, `ArcsinhStretch`, `PercentileClip`, `LinearNormalize`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation90` |
| `src/data/dataset.py` | Modify | `TNG50Dataset` with `metadata_columns`, `image_transform`, `metadata_transform`; returns `(image, meta)` tuple |
| `configs/data/preprocess.yaml` | Create | Hydra config for transform pipeline composition |
| `configs/config.yaml` | Modify | Add `preprocess` to defaults list |
| `src/train/trainer.py` | Modify | Unpack `(images, meta)` tuple from batch |
| `tests/data/test_preprocess.py` | Rewrite | Tests for all transform classes |
| `tests/data/test_dataset.py` | Modify | Tests for new dataset interface and tuple return |

---

### Task 1: Scientific Transform Classes — `SurfaceBrightnessToNanomaggies`, `ClipAndPad`, `ArcsinhStretch`

**Files:**
- Modify: `src/data/preprocess.py`
- Rewrite: `tests/data/test_preprocess.py`

These three transforms are straightforward wrappers around existing logic, updated to handle `(C, H, W)` input.

- [ ] **Step 1: Write failing tests for `SurfaceBrightnessToNanomaggies`**

```python
"""Tests for src.data.preprocess."""

import numpy as np
import pytest

from src.data.preprocess import SurfaceBrightnessToNanomaggies


class TestSurfaceBrightnessToNanomaggies:
    """Tests for SurfaceBrightnessToNanomaggies transform."""

    def test_known_value(self):
        """Verify mag=22.5 converts to 1.0 nanomaggy."""
        t = SurfaceBrightnessToNanomaggies()
        img = np.array([[[22.5]]])  # (1, 1, 1)
        out = t(img)
        np.testing.assert_allclose(out, [[[1.0]]])

    def test_brighter_is_higher_flux(self):
        """Verify brighter magnitudes produce higher flux."""
        t = SurfaceBrightnessToNanomaggies()
        img = np.array([[[20.0, 22.5]]])  # (1, 1, 2)
        out = t(img)
        assert out[0, 0, 0] > out[0, 0, 1]

    def test_above_threshold_zeroed(self):
        """Verify pixels at or above threshold are zeroed."""
        t = SurfaceBrightnessToNanomaggies(mag_threshold=99.0)
        img = np.array([[[99.0, 100.0]]])
        out = t(img)
        np.testing.assert_array_equal(out, [[[0.0, 0.0]]])

    def test_preserves_shape(self):
        """Verify output shape matches input (C, H, W) shape."""
        t = SurfaceBrightnessToNanomaggies()
        img = np.full((3, 64, 64), 22.0)
        out = t(img)
        assert out.shape == (3, 64, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_preprocess.py::TestSurfaceBrightnessToNanomaggies -v`
Expected: FAIL — `ImportError` because `SurfaceBrightnessToNanomaggies` class does not exist yet.

- [ ] **Step 3: Implement `SurfaceBrightnessToNanomaggies` class**

Replace the contents of `src/data/preprocess.py` with:

```python
"""Preprocessing transforms for TNG50 galaxy images.

Each transform is a callable class with ``__init__`` for parameters and
``__call__(img)`` operating on ``(C, H, W)`` NumPy arrays. Compose via
``torchvision.transforms.Compose``.
"""

import numpy as np


class SurfaceBrightnessToNanomaggies:
    """Convert surface-brightness (AB mag/pixel) to nanomaggies.

    Args:
        mag_threshold: Pixels fainter than this value are zeroed.
    """

    def __init__(self, mag_threshold: float = 99.0):
        self.mag_threshold = mag_threshold

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply flux conversion.

        Args:
            img: ``(C, H, W)`` surface-brightness array.

        Returns:
            Flux array in nanomaggies, same shape as input.
        """
        return np.where(
            img < self.mag_threshold, 10.0 ** (0.4 * (22.5 - img)), 0.0
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_preprocess.py::TestSurfaceBrightnessToNanomaggies -v`
Expected: 4 PASSED

- [ ] **Step 5: Write failing tests for `ClipAndPad`**

Add to `tests/data/test_preprocess.py`:

```python
from src.data.preprocess import ClipAndPad


class TestClipAndPad:
    """Tests for ClipAndPad transform."""

    def test_small_image_pads_to_target(self):
        """Verify a smaller-than-target image is padded and cropped."""
        t = ClipAndPad(n=256)
        img = np.ones((1, 100, 100))
        out = t(img)
        assert out.shape == (1, 256, 256)

    def test_large_image_crops_to_target(self):
        """Verify a larger-than-target image is centre-cropped."""
        t = ClipAndPad(n=256)
        img = np.ones((1, 600, 600))
        out = t(img)
        assert out.shape == (1, 256, 256)

    def test_exact_size_is_identity(self):
        """Verify an image at target size is returned unchanged."""
        t = ClipAndPad(n=4)
        img = np.arange(16).reshape(1, 4, 4).astype(float)
        out = t(img)
        np.testing.assert_array_equal(out, img)

    def test_default_n_is_512(self):
        """Verify the default target size is 512."""
        t = ClipAndPad()
        img = np.ones((1, 512, 512))
        out = t(img)
        assert out.shape == (1, 512, 512)

    def test_nonsquare_input(self):
        """Verify non-square input is padded/cropped correctly."""
        t = ClipAndPad(n=256)
        img = np.ones((1, 100, 300))
        out = t(img)
        assert out.shape == (1, 256, 256)

    def test_multi_channel(self):
        """Verify multi-channel (C, H, W) input is handled."""
        t = ClipAndPad(n=64)
        img = np.ones((3, 50, 50))
        out = t(img)
        assert out.shape == (3, 64, 64)
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/data/test_preprocess.py::TestClipAndPad -v`
Expected: FAIL — `ImportError` because `ClipAndPad` class does not exist yet.

- [ ] **Step 7: Implement `ClipAndPad` class**

Add to `src/data/preprocess.py`:

```python
class ClipAndPad:
    """Pad image to at least *n x n*, then centre-crop to exactly *n x n*.

    Operates on the last two (spatial) axes of a ``(C, H, W)`` array.

    Args:
        n: Target side length in pixels.
    """

    def __init__(self, n: int = 512):
        self.n = n

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply pad and crop.

        Args:
            img: ``(C, H, W)`` array.

        Returns:
            Array of shape ``(C, n, n)``.
        """
        n = self.n
        h, w = img.shape[-2], img.shape[-1]
        pad_h = max(0, n - h)
        pad_w = max(0, n - w)

        if pad_h > 0 or pad_w > 0:
            top, left = pad_h // 2, pad_w // 2
            # Pad only spatial dims; leave channel dim unpadded
            pad_widths = [(0, 0)] * (img.ndim - 2) + [
                (top, pad_h - top),
                (left, pad_w - left),
            ]
            img = np.pad(img, pad_widths, mode="constant", constant_values=0)

        cy, cx = img.shape[-2] // 2, img.shape[-1] // 2
        half = n // 2
        return img[..., cy - half : cy + half, cx - half : cx + half]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/data/test_preprocess.py::TestClipAndPad -v`
Expected: 6 PASSED

- [ ] **Step 9: Write failing tests for `ArcsinhStretch`**

Add to `tests/data/test_preprocess.py`:

```python
from src.data.preprocess import ArcsinhStretch


class TestArcsinhStretch:
    """Tests for ArcsinhStretch transform."""

    def test_known_value(self):
        """Verify stretch matches np.arcsinh(x/scale)."""
        t = ArcsinhStretch(scale=2.0)
        img = np.array([[[0.0, 1.0, 10.0]]])
        out = t(img)
        np.testing.assert_allclose(out, np.arcsinh(img / 2.0))

    def test_preserves_shape(self):
        """Verify output shape matches input."""
        t = ArcsinhStretch(scale=1.0)
        img = np.ones((3, 64, 64))
        out = t(img)
        assert out.shape == (3, 64, 64)

    def test_zero_input(self):
        """Verify zero input produces zero output."""
        t = ArcsinhStretch(scale=1.0)
        img = np.zeros((1, 4, 4))
        out = t(img)
        np.testing.assert_array_equal(out, 0.0)

    def test_default_scale_is_one(self):
        """Verify default scale parameter is 1.0."""
        t = ArcsinhStretch()
        img = np.array([[[5.0]]])
        out = t(img)
        np.testing.assert_allclose(out, np.arcsinh(img / 1.0))
```

- [ ] **Step 10: Run tests to verify they fail**

Run: `pytest tests/data/test_preprocess.py::TestArcsinhStretch -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 11: Implement `ArcsinhStretch` class**

Add to `src/data/preprocess.py`:

```python
class ArcsinhStretch:
    """Apply arcsinh stretch to compress dynamic range.

    Computes ``arcsinh(img / scale)``. The ``scale`` parameter corresponds
    to the ``a`` parameter in the former ``arcsinh_stretch`` free function.

    Args:
        scale: Softening parameter controlling the stretch.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply arcsinh stretch.

        Args:
            img: ``(C, H, W)`` array.

        Returns:
            Stretched array, same shape as input.
        """
        return np.arcsinh(img / self.scale)
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `pytest tests/data/test_preprocess.py::TestArcsinhStretch -v`
Expected: 4 PASSED

- [ ] **Step 13: Run all preprocess tests together**

Run: `pytest tests/data/test_preprocess.py -v`
Expected: 14 PASSED (4 + 6 + 4)

- [ ] **Step 14: Commit**

```bash
git add src/data/preprocess.py tests/data/test_preprocess.py
git commit -m "refactor: replace preprocess free functions with SurfaceBrightnessToNanomaggies, ClipAndPad, ArcsinhStretch classes

Transforms now operate on (C, H, W) arrays and are composable via
torchvision.transforms.Compose."
```

---

### Task 2: Scientific Transform Classes — `PercentileClip`, `LinearNormalize`

**Files:**
- Modify: `src/data/preprocess.py`
- Modify: `tests/data/test_preprocess.py`

These two transforms replace the inline percentile/normalize logic from the old `preprocess_image`.

- [ ] **Step 1: Write failing tests for `PercentileClip`**

Add to `tests/data/test_preprocess.py`:

```python
from src.data.preprocess import PercentileClip


class TestPercentileClip:
    """Tests for PercentileClip transform."""

    def test_output_clipped_to_zero_one(self):
        """Verify output is in [0, 1]."""
        t = PercentileClip(percentile=99.0)
        rng = np.random.default_rng(42)
        img = rng.uniform(0, 100, size=(1, 64, 64))
        out = t(img)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_preserves_shape(self):
        """Verify output shape matches input."""
        t = PercentileClip(percentile=99.0)
        img = np.ones((3, 32, 32))
        out = t(img)
        assert out.shape == (3, 32, 32)

    def test_all_zero_image_no_division_by_zero(self):
        """Verify all-zero image returns all zeros without error."""
        t = PercentileClip(percentile=99.0)
        img = np.zeros((1, 64, 64))
        out = t(img)
        np.testing.assert_array_equal(out, 0.0)

    def test_percentile_computation(self):
        """Verify division by percentile value."""
        t = PercentileClip(percentile=100.0)
        img = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        out = t(img)
        # 100th percentile is 5.0; result is img / 5.0 clipped to [0, 1]
        expected = np.array([[[0.2, 0.4, 0.6, 0.8, 1.0]]])
        np.testing.assert_allclose(out, expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_preprocess.py::TestPercentileClip -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement `PercentileClip` class**

Add to `src/data/preprocess.py`:

```python
class PercentileClip:
    """Clip image intensity by percentile value and normalize to [0, 1].

    Computes the given percentile, divides the image by it, and clips
    to ``[0, 1]``. If the percentile value is zero (e.g., all-zero image),
    the image is returned unchanged.

    Args:
        percentile: Percentile value used for normalization.
    """

    def __init__(self, percentile: float = 99.0):
        self.percentile = percentile

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply percentile clipping.

        Args:
            img: ``(C, H, W)`` array.

        Returns:
            Array clipped to ``[0, 1]``, same shape as input.
        """
        pval = np.percentile(img, self.percentile)
        if pval == 0:
            return img
        return np.clip(img / pval, 0.0, 1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_preprocess.py::TestPercentileClip -v`
Expected: 4 PASSED

- [ ] **Step 5: Write failing tests for `LinearNormalize`**

Add to `tests/data/test_preprocess.py`:

```python
from src.data.preprocess import LinearNormalize


class TestLinearNormalize:
    """Tests for LinearNormalize transform."""

    def test_maps_zero_one_to_target_range(self):
        """Verify [0, 1] maps to [norm_min, norm_max]."""
        t = LinearNormalize(norm_min=-1.0, norm_max=1.0)
        img = np.array([[[0.0, 0.5, 1.0]]])
        out = t(img)
        np.testing.assert_allclose(out, [[[-1.0, 0.0, 1.0]]])

    def test_custom_range(self):
        """Verify mapping to custom range."""
        t = LinearNormalize(norm_min=0.0, norm_max=10.0)
        img = np.array([[[0.0, 0.5, 1.0]]])
        out = t(img)
        np.testing.assert_allclose(out, [[[0.0, 5.0, 10.0]]])

    def test_preserves_shape(self):
        """Verify output shape matches input."""
        t = LinearNormalize(norm_min=-1.0, norm_max=1.0)
        img = np.ones((3, 32, 32)) * 0.5
        out = t(img)
        assert out.shape == (3, 32, 32)

    def test_default_range_is_neg1_to_1(self):
        """Verify default range is [-1, 1]."""
        t = LinearNormalize()
        img = np.array([[[0.0, 1.0]]])
        out = t(img)
        np.testing.assert_allclose(out, [[[-1.0, 1.0]]])
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/data/test_preprocess.py::TestLinearNormalize -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 7: Implement `LinearNormalize` class**

Add to `src/data/preprocess.py`:

```python
class LinearNormalize:
    """Linearly map from [0, 1] to [norm_min, norm_max].

    Input is assumed to be in ``[0, 1]`` (guaranteed by ``PercentileClip``).

    Args:
        norm_min: Minimum of the target range.
        norm_max: Maximum of the target range.
    """

    def __init__(self, norm_min: float = -1.0, norm_max: float = 1.0):
        self.norm_min = norm_min
        self.norm_max = norm_max

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply linear normalization.

        Args:
            img: ``(C, H, W)`` array in ``[0, 1]``.

        Returns:
            Array in ``[norm_min, norm_max]``, same shape as input.
        """
        return img * (self.norm_max - self.norm_min) + self.norm_min
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/data/test_preprocess.py::TestLinearNormalize -v`
Expected: 4 PASSED

- [ ] **Step 9: Remove old `preprocess_image` function and clean up unused imports**

Remove from `src/data/preprocess.py`:
- The `preprocess_image` function
- Unused imports: `hydra`, `logging`, `astropy`, `astropy.units`, `astropy.cosmology`, `omegaconf`
- Old free functions (`clip_and_pad`, `surface_brightness_to_nanomaggies`, `arcsinh_stretch`, `linear_normalize`)

The file should now contain only the module docstring, `import numpy as np`, and the five class definitions.

- [ ] **Step 10: Run all preprocess tests**

Run: `pytest tests/data/test_preprocess.py -v`
Expected: 22 PASSED (4 + 6 + 4 + 4 + 4). No import errors. Old test classes should have been replaced in Task 1.

- [ ] **Step 11: Commit**

```bash
git add src/data/preprocess.py tests/data/test_preprocess.py
git commit -m "feat: add PercentileClip and LinearNormalize transforms, remove old free functions

PercentileClip handles all-zero images safely. LinearNormalize assumes
[0,1] input range (guaranteed by PercentileClip in the pipeline)."
```

---

### Task 3: Augmentation Transform Classes

**Files:**
- Modify: `src/data/preprocess.py`
- Modify: `tests/data/test_preprocess.py`

- [ ] **Step 1: Write failing tests for augmentations**

Add to `tests/data/test_preprocess.py`:

```python
from src.data.preprocess import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation90


class TestRandomHorizontalFlip:
    """Tests for RandomHorizontalFlip transform."""

    def test_preserves_shape(self):
        """Verify output shape matches input."""
        t = RandomHorizontalFlip(p=0.5)
        img = np.ones((1, 64, 64))
        out = t(img)
        assert out.shape == (1, 64, 64)

    def test_deterministic_when_p_one(self):
        """Verify always flips when p=1."""
        t = RandomHorizontalFlip(p=1.0)
        img = np.arange(4).reshape(1, 2, 2).astype(float)
        out = t(img)
        expected = np.flip(img, axis=-1)
        np.testing.assert_array_equal(out, expected)

    def test_no_flip_when_p_zero(self):
        """Verify never flips when p=0."""
        t = RandomHorizontalFlip(p=0.0)
        img = np.arange(4).reshape(1, 2, 2).astype(float)
        out = t(img)
        np.testing.assert_array_equal(out, img)


class TestRandomVerticalFlip:
    """Tests for RandomVerticalFlip transform."""

    def test_preserves_shape(self):
        """Verify output shape matches input."""
        t = RandomVerticalFlip(p=0.5)
        img = np.ones((1, 64, 64))
        out = t(img)
        assert out.shape == (1, 64, 64)

    def test_deterministic_when_p_one(self):
        """Verify always flips when p=1."""
        t = RandomVerticalFlip(p=1.0)
        img = np.arange(4).reshape(1, 2, 2).astype(float)
        out = t(img)
        expected = np.flip(img, axis=-2)
        np.testing.assert_array_equal(out, expected)

    def test_no_flip_when_p_zero(self):
        """Verify never flips when p=0."""
        t = RandomVerticalFlip(p=0.0)
        img = np.arange(4).reshape(1, 2, 2).astype(float)
        out = t(img)
        np.testing.assert_array_equal(out, img)


class TestRandomRotation90:
    """Tests for RandomRotation90 transform."""

    def test_preserves_shape(self):
        """Verify output shape matches input."""
        t = RandomRotation90()
        img = np.ones((1, 64, 64))
        out = t(img)
        assert out.shape == (1, 64, 64)

    def test_stochastic_behavior(self):
        """Verify not all outputs are identical over many calls."""
        t = RandomRotation90()
        img = np.arange(16).reshape(1, 4, 4).astype(float)
        outputs = [t(img).tobytes() for _ in range(20)]
        # With 4 equally likely rotations over 20 trials, seeing only 1
        # unique output is astronomically unlikely
        assert len(set(outputs)) > 1

    def test_output_is_valid_rotation(self):
        """Verify output matches one of the four 90-degree rotations."""
        t = RandomRotation90()
        img = np.arange(16).reshape(1, 4, 4).astype(float)
        valid = [
            np.rot90(img, k=k, axes=(-2, -1)).tobytes() for k in range(4)
        ]
        for _ in range(10):
            out = t(img)
            assert out.tobytes() in valid
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_preprocess.py::TestRandomHorizontalFlip tests/data/test_preprocess.py::TestRandomVerticalFlip tests/data/test_preprocess.py::TestRandomRotation90 -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement augmentation classes**

Add to `src/data/preprocess.py`:

```python
class RandomHorizontalFlip:
    """Randomly flip image horizontally.

    Args:
        p: Probability of flipping.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply random horizontal flip along the last axis.

        Args:
            img: ``(C, H, W)`` array.

        Returns:
            Possibly flipped array, same shape as input.
        """
        if np.random.random() < self.p:
            return np.ascontiguousarray(np.flip(img, axis=-1))
        return img


class RandomVerticalFlip:
    """Randomly flip image vertically.

    Args:
        p: Probability of flipping.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply random vertical flip along the second-to-last axis.

        Args:
            img: ``(C, H, W)`` array.

        Returns:
            Possibly flipped array, same shape as input.
        """
        if np.random.random() < self.p:
            return np.ascontiguousarray(np.flip(img, axis=-2))
        return img


class RandomRotation90:
    """Randomly apply 0, 1, 2, or 3 quarter-turns (90-degree rotations).

    All four outcomes are equally likely.
    """

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply random 90-degree rotation on spatial axes.

        Args:
            img: ``(C, H, W)`` array.

        Returns:
            Rotated array, same shape as input.
        """
        k = np.random.randint(4)
        return np.ascontiguousarray(np.rot90(img, k=k, axes=(-2, -1)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_preprocess.py::TestRandomHorizontalFlip tests/data/test_preprocess.py::TestRandomVerticalFlip tests/data/test_preprocess.py::TestRandomRotation90 -v`
Expected: 9 PASSED (3 + 3 + 3)

- [ ] **Step 5: Write and run end-to-end compose test**

Add to `tests/data/test_preprocess.py`:

```python
from torchvision.transforms import Compose

from src.data.preprocess import (
    SurfaceBrightnessToNanomaggies,
    ClipAndPad,
    ArcsinhStretch,
    PercentileClip,
    LinearNormalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation90,
)


class TestComposeEndToEnd:
    """End-to-end test for the full transform pipeline."""

    def test_full_pipeline_shape_and_range(self):
        """Verify the full pipeline produces correct shape and range."""
        pipeline = Compose([
            SurfaceBrightnessToNanomaggies(),
            ClipAndPad(n=128),
            ArcsinhStretch(scale=1.0),
            PercentileClip(percentile=99.0),
            LinearNormalize(norm_min=-1.0, norm_max=1.0),
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

Run: `pytest tests/data/test_preprocess.py::TestComposeEndToEnd -v`
Expected: 1 PASSED

- [ ] **Step 6: Run full preprocess test suite**

Run: `pytest tests/data/test_preprocess.py -v`
Expected: 32 PASSED (4 + 6 + 4 + 4 + 4 + 3 + 3 + 3 + 1)

- [ ] **Step 7: Commit**

```bash
git add src/data/preprocess.py tests/data/test_preprocess.py
git commit -m "feat: add augmentation transforms and end-to-end compose test

RandomHorizontalFlip, RandomVerticalFlip, RandomRotation90 operate on
(C, H, W) spatial axes. Full pipeline tested with torchvision Compose."
```

---

### Task 4: Refactor `TNG50Dataset`

**Files:**
- Modify: `src/data/dataset.py`
- Modify: `tests/data/test_dataset.py`

- [ ] **Step 1: Write failing tests for new dataset interface**

Rewrite `tests/data/test_dataset.py`:

```python
"""Tests for src.data.dataset."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TNG50Dataset


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a minimal processed directory with .npy files and metadata."""
    records = []
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
        })
    pd.DataFrame(records).to_csv(tmp_path / "metadata.csv", index=False)
    return str(tmp_path)


def test_dataset_length(sample_dataset):
    """Verify __len__ matches number of entries in metadata."""
    ds = TNG50Dataset(sample_dataset)
    assert len(ds) == 5


def test_dataset_returns_tuple(sample_dataset):
    """Verify __getitem__ returns (image_tensor, meta_tensor) tuple."""
    ds = TNG50Dataset(sample_dataset)
    result = ds[0]
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_dataset_no_metadata_returns_empty_tensor(sample_dataset):
    """Verify meta is empty tensor when metadata_columns is None."""
    ds = TNG50Dataset(sample_dataset)
    img, meta = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 64, 64)
    assert img.dtype == torch.float32
    assert isinstance(meta, torch.Tensor)
    assert meta.numel() == 0


def test_dataset_with_metadata_columns(sample_dataset):
    """Verify metadata columns are returned as float tensor."""
    ds = TNG50Dataset(sample_dataset, metadata_columns=["hdr_mass", "hdr_redshift"])
    img, meta = ds[0]
    assert img.shape == (1, 64, 64)
    assert meta.shape == (2,)
    assert meta.dtype == torch.float32
    # First galaxy: mass=0.0, redshift=0.0
    torch.testing.assert_close(meta, torch.tensor([0.0, 0.0]))


def test_dataset_metadata_values_correct(sample_dataset):
    """Verify metadata values match CSV for non-zero entries."""
    ds = TNG50Dataset(sample_dataset, metadata_columns=["hdr_mass", "hdr_redshift"])
    _, meta = ds[2]
    # Third galaxy: mass=3.0, redshift=0.2
    torch.testing.assert_close(meta, torch.tensor([3.0, 0.2]))


def test_dataset_image_transform_applied(sample_dataset):
    """Verify image_transform is called on the NumPy array."""
    transform = lambda x: x * 2.0
    ds = TNG50Dataset(sample_dataset, image_transform=transform)
    raw_ds = TNG50Dataset(sample_dataset)
    img, _ = ds[0]
    raw_img, _ = raw_ds[0]
    torch.testing.assert_close(img, raw_img * 2.0)


def test_dataset_metadata_transform_applied(sample_dataset):
    """Verify metadata_transform is called on the metadata tensor."""
    meta_transform = lambda x: x + 100.0
    ds = TNG50Dataset(
        sample_dataset,
        metadata_columns=["hdr_mass"],
        metadata_transform=meta_transform,
    )
    _, meta = ds[1]
    # Second galaxy: mass=1.5, after transform: 101.5
    torch.testing.assert_close(meta, torch.tensor([101.5]))


def test_dataset_metadata_accessible(sample_dataset):
    """Verify metadata DataFrame is accessible."""
    ds = TNG50Dataset(sample_dataset)
    assert isinstance(ds.metadata, pd.DataFrame)
    assert len(ds.metadata) == 5
    assert "fits_name" in ds.metadata.columns


def test_dataset_works_with_dataloader(sample_dataset):
    """Verify dataset integrates with PyTorch DataLoader."""
    ds = TNG50Dataset(sample_dataset)
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    images, meta = next(iter(loader))
    assert images.shape == (2, 1, 64, 64)
    assert meta.numel() == 0


def test_dataset_with_metadata_works_with_dataloader(sample_dataset):
    """Verify metadata columns work with DataLoader batching."""
    ds = TNG50Dataset(sample_dataset, metadata_columns=["hdr_mass", "hdr_redshift"])
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    images, meta = next(iter(loader))
    assert images.shape == (2, 1, 64, 64)
    assert meta.shape == (2, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_dataset.py -v`
Expected: Multiple FAILs — `TNG50Dataset` does not accept `metadata_columns`, `image_transform`, or `metadata_transform` yet; and `__getitem__` does not return a tuple.

- [ ] **Step 3: Implement refactored `TNG50Dataset`**

Rewrite `src/data/dataset.py`:

```python
"""PyTorch Dataset for processed TNG50 galaxy images."""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TNG50Dataset(Dataset):
    """Random-access dataset over extracted TNG50 galaxy ``.npy`` files.

    Always returns ``(image_tensor, meta_tensor)`` tuples. When
    ``metadata_columns`` is ``None``, ``meta_tensor`` is ``torch.empty(0)``.

    Args:
        processed_dir: Path to directory containing ``metadata.csv`` and
            ``.npy`` image files.
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
        metadata_columns: list[str] | None = None,
        image_transform=None,
        metadata_transform=None,
    ):
        self.processed_dir = processed_dir
        self.metadata_columns = metadata_columns
        self.image_transform = image_transform
        self.metadata_transform = metadata_transform
        csv_path = os.path.join(processed_dir, "metadata.csv")
        self.metadata = pd.read_csv(csv_path)
        self.filenames = self.metadata["filename"].tolist()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and return a galaxy image with optional metadata.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of ``(image_tensor, meta_tensor)``. ``image_tensor`` has
            shape ``(C, H, W)`` with dtype ``float32``. ``meta_tensor`` has
            shape ``(N,)`` where N is the number of metadata columns, or
            ``torch.empty(0)`` if no metadata columns were specified.
        """
        path = os.path.join(self.processed_dir, self.filenames[idx])
        data = np.load(path)

        if self.image_transform is not None:
            data = self.image_transform(data)

        img_tensor = torch.from_numpy(np.ascontiguousarray(data)).float()

        if self.metadata_columns is None:
            return img_tensor, torch.empty(0)

        meta = (
            self.metadata.iloc[idx][self.metadata_columns]
            .values.astype(np.float32)
        )
        meta_tensor = torch.from_numpy(meta)

        if self.metadata_transform is not None:
            meta_tensor = self.metadata_transform(meta_tensor)

        return img_tensor, meta_tensor
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_dataset.py -v`
Expected: 11 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset.py tests/data/test_dataset.py
git commit -m "refactor: TNG50Dataset returns (image, meta) tuple with separate transforms

Accepts metadata_columns, image_transform, metadata_transform. Returns
torch.empty(0) as metadata placeholder when no columns specified."
```

---

### Task 5: Hydra Config and Training Loop Update

**Files:**
- Create: `configs/data/preprocess.yaml`
- Modify: `configs/config.yaml`
- Modify: `src/train/trainer.py:96-101`

- [ ] **Step 1: Create `configs/data/preprocess.yaml`**

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

- [ ] **Step 2: Update `configs/config.yaml` defaults**

Add `- data@data.preprocess: preprocess` after the download entry:

```yaml
defaults:
  - data@data.download: download
  - data@data.preprocess: preprocess
  - model@model: unet
  - flow@flow.otfm: otfm
  - flow@flow.sample: sample
  - train@train: train
  - _self_

seed: 42
work_dir: ${hydra:runtime.cwd}
```

- [ ] **Step 3: Update training loop to unpack tuple**

In `src/train/trainer.py`, change line 101 from:

```python
        x1_np = batch.numpy()
```

to:

```python
        images, _meta = batch
        x1_np = images.numpy()
```

Also update the docstring for the `dataloader` parameter (line 73-74) from:

```python
        dataloader: PyTorch DataLoader (or any iterator yielding batches as
                    torch.Tensor of shape (B, C, H, W))
```

to:

```python
        dataloader: PyTorch DataLoader yielding ``(images, meta)`` tuples
                    where images is a ``(B, C, H, W)`` tensor.
```

- [ ] **Step 4: Run all tests to verify nothing is broken**

Run: `pytest tests/ -v`
Expected: All tests pass. The dataset tests (11) and preprocess tests (32) should all pass. Download tests should be unaffected.

- [ ] **Step 5: Commit**

```bash
git add configs/data/preprocess.yaml configs/config.yaml src/train/trainer.py
git commit -m "feat: add Hydra preprocess config and update trainer for (image, meta) tuples

New configs/data/preprocess.yaml defines transform pipeline via _target_
instantiation. Trainer unpacks (images, meta) from DataLoader batches."
```
