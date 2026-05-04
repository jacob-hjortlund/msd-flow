"""Tests for msdflow.data.preprocess."""

import functools
import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from fastdigest import TDigest
import numpy as np
import pytest

from omegaconf import OmegaConf
from torchvision.transforms import Compose

from msdflow.data.preprocess import (
    ArcsinhStretch,
    ClipAndPad,
    CLRTransform,
    GlobalNorm,
    LinearNormalize,
    PDFNorm,
    PercentileClip,
    RandomHorizontalFlip,
    RandomRotation90,
    RandomVerticalFlip,
    StandardizeTransform,
    SurfaceBrightnessToNanomaggies,
    _identity,
    build_tdigest,
    _filter_positive,
    _flatten,
    _transform_metadata,
)


def _double_image(img):
    """Return an image with all pixel values doubled.

    Args:
        img: Input image array.

    Returns:
        Image array with each value multiplied by two.
    """
    return img * 2.0


def _scale_image_for_cache_test(img, factor: float = 1.0):
    """Scale an image by a configurable factor for cache tests.

    Args:
        img: Input image array.
        factor: Multiplicative scale applied to input images.

    Returns:
        Scaled image array.
    """
    return img * factor


def _make_default_scale_image_for_cache_test(factor: float):
    """Create a function transform with scale stored in defaults.

    Args:
        factor: Multiplicative scale applied to input images.

    Returns:
        Function transform with the scale stored in ``__defaults__``.
    """

    def scale(img, factor=factor):
        """Scale an image using a default argument.

        Args:
            img: Input image array.
            factor: Multiplicative scale applied to input images.

        Returns:
            Scaled image array.
        """
        return img * factor

    return scale


class ScaleImageForCacheTest:
    """Scale images by a constant factor for standardization cache tests.

    Args:
        factor: Multiplicative scale applied to input images.
    """

    def __init__(self, factor: float):
        """Initialize the scale transform.

        Args:
            factor: Multiplicative scale applied to input images.
        """
        self.factor = factor

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Scale an image.

        Args:
            img: Input image array.

        Returns:
            Scaled image array.
        """
        return img * self.factor


class SlottedScaleImageForCacheTest:
    """Scale images using a slot-only public factor.

    Args:
        factor: Multiplicative scale applied to input images.
    """

    __slots__ = ("factor",)

    def __init__(self, factor: float):
        """Initialize the slotted scale transform.

        Args:
            factor: Multiplicative scale applied to input images.
        """
        self.factor = factor

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Scale an image.

        Args:
            img: Input image array.

        Returns:
            Scaled image array.
        """
        return img * self.factor


class ArrayScaleImageForCacheTest:
    """Scale images using an ndarray-valued public config.

    Args:
        scale_config: Array whose first value is the multiplicative scale.
    """

    def __init__(self, scale_config: np.ndarray):
        """Initialize the ndarray-configured scale transform.

        Args:
            scale_config: Array whose first value is the multiplicative scale.
        """
        self.scale_config = scale_config

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Scale an image.

        Args:
            img: Input image array.

        Returns:
            Scaled image array.
        """
        return img * float(self.scale_config[0])


class UnsupportedMetadataTransformForCacheTest:
    """Scale images while exposing unsupported public metadata.

    Args:
        factor: Multiplicative scale applied to input images.
    """

    def __init__(self, factor: float):
        """Initialize the transform with an unsupported public attr.

        Args:
            factor: Multiplicative scale applied to input images.
        """
        self.factor = factor
        self.unsupported_config = object()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Scale an image.

        Args:
            img: Input image array.

        Returns:
            Scaled image array.
        """
        return img * self.factor


class NoPublicAttrsCallableForCacheTest:
    """Scale images without exposing public cache metadata attrs."""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Scale an image.

        Args:
            img: Input image array.

        Returns:
            Scaled image array.
        """
        return img * 2.0


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


class TestCLRTransform:
    """Tests for CLRTransform."""

    def test_handles_zero_pixels_and_returns_finite_output(self):
        """CLRTransform should smooth exact zero pixels before taking logs."""
        transform = CLRTransform(eps_mass=1e-6)
        img = np.array(
            [
                [[0.0, 0.2], [0.3, 0.5]],
                [[0.1, 0.0], [0.4, 0.5]],
            ],
            dtype=np.float32,
        )

        out = transform(img)

        assert np.all(np.isfinite(out))

    def test_preserves_input_shape(self):
        """CLRTransform should preserve the input ``(C, H, W)`` shape."""
        transform = CLRTransform(eps_mass=1e-6)
        img = np.ones((3, 4, 5), dtype=np.float32) / 20.0

        out = transform(img)

        assert out.shape == img.shape

    def test_output_is_sum_zero_per_channel(self):
        """CLRTransform should center each channel independently."""
        transform = CLRTransform(eps_mass=1e-6)
        img = np.array(
            [
                [[0.25, 0.25], [0.25, 0.25]],
                [[0.7, 0.1], [0.1, 0.1]],
            ],
            dtype=np.float32,
        )

        out = transform(img)

        np.testing.assert_allclose(
            np.sum(out, axis=(-2, -1)),
            np.zeros((2,), dtype=out.dtype),
            atol=1e-6,
        )

    def test_rejects_non_image_input(self):
        """CLRTransform should require ``(C, H, W)`` input."""
        transform = CLRTransform(eps_mass=1e-6)

        with pytest.raises(ValueError, match="Expected image with shape"):
            transform(np.ones((4, 4), dtype=np.float32))

    @pytest.mark.parametrize("eps_mass", [-1e-6, 1.0 + 1e-6, np.nan])
    def test_rejects_invalid_eps_mass(self, eps_mass):
        """CLRTransform should require ``eps_mass`` in ``[0, 1]``."""
        with pytest.raises(ValueError, match="eps_mass must be in \\[0, 1\\]"):
            CLRTransform(eps_mass=eps_mass)


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


class TestArcsinhStretchSplit:
    """Tests for ArcsinhStretch split-aware TDigest computation."""

    @pytest.fixture
    def split_dataset(self, tmp_path):
        """Create dataset with split column: 2 train, 1 val, 1 test."""
        import pandas as pd
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
        """Verify TDigest cache file includes split name and image size."""
        ArcsinhStretch(
            scale=None, percentile=50, data_dir=split_dataset, split="train"
        )
        import os
        assert os.path.isfile(
            os.path.join(split_dataset, "arcsinh_tdigest_4_train_function.json")
        )

    def test_explicit_scale_ignores_split(self):
        """Verify explicit scale doesn't require split parameter."""
        t = ArcsinhStretch(scale=1.0)
        assert t.scale == 1.0


class TestGlobalNormSplit:
    """Tests for GlobalNorm split-aware TDigest computation."""

    @pytest.fixture
    def split_dataset(self, tmp_path):
        """Create dataset with split column: 2 train, 1 val."""
        import pandas as pd
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
        """Verify TDigest cache file includes split name and image size."""
        GlobalNorm(
            global_min=None, global_max=None,
            data_dir=split_dataset, percentile=50, split="train"
        )
        import os
        assert os.path.isfile(
            os.path.join(split_dataset, "global_norm_tdigest_50_4_train_function.json")
        )

    def test_explicit_bounds_ignores_split(self):
        """Verify explicit global bounds don't require split parameter."""
        t = GlobalNorm(global_min=0.0, global_max=1.0)
        assert t.global_min == 0.0
        assert t.global_max == 1.0


class TestStandardizeTransformDirect:
    """Tests for direct StandardizeTransform parameters."""

    def test_known_value(self):
        """Verify direct standardization applies ``(img - mu) / sigma``."""
        t = StandardizeTransform(mu=2.0, sigma=2.0)
        img = np.array([[[0.0, 2.0, 4.0]]])

        out = t(img)

        np.testing.assert_allclose(out, [[[-1.0, 0.0, 1.0]]])

    def test_preserves_shape(self):
        """Verify output shape matches input."""
        t = StandardizeTransform(mu=2.0, sigma=0.5)
        img = np.ones((3, 8, 9))

        out = t(img)

        assert out.shape == img.shape

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mu": None, "sigma": 1.0},
            {"mu": 0.0, "sigma": None},
            {},
        ],
    )
    def test_rejects_missing_direct_parameters(self, kwargs):
        """Verify direct mode requires explicit mu and sigma."""
        with pytest.raises(ValueError, match="Must use either"):
            StandardizeTransform(**kwargs)

    @pytest.mark.parametrize("sigma", [0.0, -1.0, np.inf, np.nan])
    def test_rejects_invalid_sigma(self, sigma):
        """Verify invalid sigma values are rejected."""
        with pytest.raises(ValueError, match="sigma must be finite and positive"):
            StandardizeTransform(mu=0.0, sigma=sigma)


class TestStandardizeTransformSplit:
    """Tests for StandardizeTransform split-aware TDigest computation."""

    @pytest.fixture
    def split_dataset(self, tmp_path):
        """Create dataset with train and validation rows."""
        import pandas as pd

        records = []
        for i, value in enumerate([1.0, 3.0]):
            name = f"galaxy_{i:05d}.npy"
            np.save(tmp_path / name, np.full((1, 4, 4), value))
            records.append({"filename": name, "split": "train"})

        name = "galaxy_00002.npy"
        np.save(tmp_path / name, np.full((1, 4, 4), 100.0))
        records.append({"filename": name, "split": "val"})

        pd.DataFrame(records).to_csv(tmp_path / "metadata.csv", index=False)
        return str(tmp_path)

    def test_tdigest_uses_train_only(self, split_dataset):
        """Verify derived mu and sigma come from train data only."""
        t = StandardizeTransform(
            mu=None,
            sigma=None,
            data_dir=split_dataset,
            split="train",
        )

        np.testing.assert_allclose(t.mu, 2.0)
        np.testing.assert_allclose(t.sigma, 1.0)
        np.testing.assert_allclose(
            t(np.array([[[1.0, 2.0, 3.0]]])), [[[-1.0, 0.0, 1.0]]]
        )

    def test_tdigest_applies_transforms(self, split_dataset):
        """Verify derived statistics come from transformed train data."""
        t = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=_double_image,
            data_dir=split_dataset,
            split="train",
        )

        np.testing.assert_allclose(t.mu, 4.0)
        np.testing.assert_allclose(t.sigma, 2.0)

    def test_tdigest_cache_distinguishes_bare_function_transforms(self, split_dataset):
        """Verify custom bare functions do not reuse the identity TDigest cache."""
        StandardizeTransform(
            mu=None,
            sigma=None,
            data_dir=split_dataset,
            split="train",
        )
        t = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=_double_image,
            data_dir=split_dataset,
            split="train",
        )

        np.testing.assert_allclose(t.mu, 4.0)
        np.testing.assert_allclose(t.sigma, 2.0)
        assert os.path.isfile(
            os.path.join(split_dataset, "standardize_tdigest_4_train_function.json")
        )
        assert os.path.isfile(
            os.path.join(
                split_dataset,
                "standardize_tdigest_4_train_function_double_image.json",
            )
        )

    def test_tdigest_cache_includes_split(self, split_dataset):
        """Verify TDigest cache file includes image size, split, and transform name."""
        StandardizeTransform(
            mu=None,
            sigma=None,
            data_dir=split_dataset,
            split="train",
        )

        assert os.path.isfile(
            os.path.join(split_dataset, "standardize_tdigest_4_train_function.json")
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"mu": 0.0},
            {"sigma": 1.0},
            {"mu": 0.0, "sigma": 1.0, "data_dir": "/unused"},
            {"mu": 0.0, "data_dir": "/unused"},
            {"sigma": 1.0, "data_dir": "/unused"},
        ],
    )
    def test_rejects_incomplete_or_mixed_modes(self, kwargs):
        """Verify callers must choose direct mode or dataset-derived mode."""
        with pytest.raises(ValueError, match="Must use either"):
            StandardizeTransform(**kwargs)


class TestStandardizeTransformCacheDir:
    """Tests for StandardizeTransform cache_dir behavior."""

    @pytest.fixture
    def split_dirs(self, tmp_path):
        """Create data_dir with .npy files and a separate empty cache_dir."""
        import pandas as pd

        data_dir = tmp_path / "data"
        cache_dir = tmp_path / "cache"
        data_dir.mkdir()
        cache_dir.mkdir()

        records = []
        for i, value in enumerate([1.0, 3.0]):
            name = f"galaxy_{i:05d}.npy"
            np.save(data_dir / name, np.full((1, 4, 4), value))
            records.append({"filename": name, "split": "train"})

        pd.DataFrame(records).to_csv(data_dir / "metadata.csv", index=False)
        return str(data_dir), str(cache_dir)

    def test_writes_cache_to_cache_dir(self, split_dirs):
        """Verify StandardizeTransform writes TDigest JSON to cache_dir."""
        data_dir, cache_dir = split_dirs

        StandardizeTransform(
            mu=None,
            sigma=None,
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )

        assert os.path.isfile(
            os.path.join(cache_dir, "standardize_tdigest_4_train_function.json")
        )
        assert not os.path.isfile(
            os.path.join(data_dir, "standardize_tdigest_4_train_function.json")
        )

    def test_reads_existing_cache_from_cache_dir(self, split_dirs, monkeypatch):
        """Verify repeated instantiation reads compatible statistics from cache_dir."""
        data_dir, cache_dir = split_dirs

        t1 = StandardizeTransform(
            mu=None,
            sigma=None,
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )

        def fail_build_tdigest(self):
            """Fail if cache lookup falls through to TDigest construction."""
            raise AssertionError("cache was not read")

        monkeypatch.setattr(
            StandardizeTransform, "_build_tdigest", fail_build_tdigest
        )

        t2 = StandardizeTransform(
            mu=None,
            sigma=None,
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )

        np.testing.assert_allclose(t1.mu, t2.mu)
        np.testing.assert_allclose(t1.sigma, t2.sigma)

    def test_rebuilds_cache_when_transform_configuration_changes(self, split_dirs):
        """Verify transform parameter changes invalidate StandardizeTransform cache."""
        data_dir, cache_dir = split_dirs

        t1 = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=ScaleImageForCacheTest(2.0),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )
        t2 = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=ScaleImageForCacheTest(3.0),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )

        np.testing.assert_allclose(t1.mu, 4.0)
        np.testing.assert_allclose(t1.sigma, 2.0)
        np.testing.assert_allclose(t2.mu, 6.0)
        np.testing.assert_allclose(t2.sigma, 3.0)
        assert os.path.isfile(
            os.path.join(
                cache_dir,
                "standardize_tdigest_4_train_ScaleImageForCacheTest.json",
            )
        )

    def test_same_metadata_custom_object_reuses_cache(
        self,
        split_dirs,
        monkeypatch,
    ):
        """Verify public-attr callable objects with same metadata reuse cache."""
        data_dir, cache_dir = split_dirs

        t1 = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=ScaleImageForCacheTest(2.0),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )

        def fail_build_tdigest(self):
            """Fail if compatible custom object cache is not read."""
            raise AssertionError("cache was not read")

        monkeypatch.setattr(
            StandardizeTransform, "_build_tdigest", fail_build_tdigest
        )

        t2 = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=ScaleImageForCacheTest(2.0),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )

        np.testing.assert_allclose(t1.mu, t2.mu)
        np.testing.assert_allclose(t1.sigma, t2.sigma)

    def test_rebuilds_cache_when_slotted_transform_configuration_changes(
        self,
        split_dirs,
    ):
        """Verify slot-only public attrs participate in cache metadata."""
        data_dir, cache_dir = split_dirs

        t1 = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=SlottedScaleImageForCacheTest(2.0),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )
        t2 = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=SlottedScaleImageForCacheTest(3.0),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )

        np.testing.assert_allclose(t1.mu, 4.0)
        np.testing.assert_allclose(t1.sigma, 2.0)
        np.testing.assert_allclose(t2.mu, 6.0)
        np.testing.assert_allclose(t2.sigma, 3.0)
        assert os.path.isfile(
            os.path.join(
                cache_dir,
                "standardize_tdigest_4_train_SlottedScaleImageForCacheTest.json",
            )
        )

    def test_rebuilds_cache_when_ndarray_transform_configuration_changes(
        self,
        split_dirs,
    ):
        """Verify ndarray public attrs participate in cache metadata."""
        data_dir, cache_dir = split_dirs

        t1 = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=ArrayScaleImageForCacheTest(np.array([2.0])),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )
        t2 = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=ArrayScaleImageForCacheTest(np.array([3.0])),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )

        np.testing.assert_allclose(t1.mu, 4.0)
        np.testing.assert_allclose(t1.sigma, 2.0)
        np.testing.assert_allclose(t2.mu, 6.0)
        np.testing.assert_allclose(t2.sigma, 3.0)
        assert os.path.isfile(
            os.path.join(
                cache_dir,
                "standardize_tdigest_4_train_ArrayScaleImageForCacheTest.json",
            )
        )

    def test_rebuilds_cache_when_partial_transform_configuration_changes(
        self,
        split_dirs,
    ):
        """Verify ``functools.partial`` state participates in cache metadata."""
        data_dir, cache_dir = split_dirs

        t1 = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=functools.partial(_scale_image_for_cache_test, factor=2.0),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )
        t2 = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=functools.partial(_scale_image_for_cache_test, factor=3.0),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )

        np.testing.assert_allclose(t1.mu, 4.0)
        np.testing.assert_allclose(t1.sigma, 2.0)
        np.testing.assert_allclose(t2.mu, 6.0)
        np.testing.assert_allclose(t2.sigma, 3.0)
        assert os.path.isfile(
            os.path.join(cache_dir, "standardize_tdigest_4_train_partial.json")
        )

    def test_rebuilds_cache_when_function_defaults_change(self, split_dirs):
        """Verify function default state participates in cache metadata."""
        data_dir, cache_dir = split_dirs

        t1 = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=_make_default_scale_image_for_cache_test(2.0),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )
        t2 = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=_make_default_scale_image_for_cache_test(3.0),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )

        np.testing.assert_allclose(t1.mu, 4.0)
        np.testing.assert_allclose(t1.sigma, 2.0)
        np.testing.assert_allclose(t2.mu, 6.0)
        np.testing.assert_allclose(t2.sigma, 3.0)
        assert os.path.isfile(
            os.path.join(cache_dir, "standardize_tdigest_4_train_function_scale.json")
        )

    def test_ufunc_metadata_includes_function_name(self):
        """Verify NumPy ufunc metadata distinguishes functions by name."""
        log_metadata = _transform_metadata(np.log)
        sqrt_metadata = _transform_metadata(np.sqrt)

        assert log_metadata != sqrt_metadata
        assert log_metadata["type"] == "ufunc"
        assert log_metadata["name"] == "log"
        assert sqrt_metadata["name"] == "sqrt"

    def test_builtin_callable_metadata_includes_function_name(self):
        """Verify builtin callable metadata distinguishes functions by name."""
        abs_metadata = _transform_metadata(abs)
        round_metadata = _transform_metadata(round)

        assert abs_metadata != round_metadata
        assert abs_metadata["type"] == "builtin_function_or_method"
        assert abs_metadata["name"] == "abs"
        assert round_metadata["name"] == "round"

    def test_unsupported_transform_metadata_does_not_trust_cache(
        self,
        split_dirs,
        monkeypatch,
    ):
        """Verify unsupported public attrs make metadata non-cacheable."""
        data_dir, cache_dir = split_dirs
        cache_path = os.path.join(
            cache_dir,
            "standardize_tdigest_4_train_UnsupportedMetadataTransformForCacheTest.json",
        )

        StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=UnsupportedMetadataTransformForCacheTest(2.0),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )
        assert not os.path.exists(cache_path)

        def fail_build_tdigest(self):
            """Fail when cache lookup correctly falls through to rebuilding."""
            raise AssertionError("cache was intentionally not trusted")

        monkeypatch.setattr(
            StandardizeTransform, "_build_tdigest", fail_build_tdigest
        )

        with pytest.raises(AssertionError, match="cache was intentionally not trusted"):
            StandardizeTransform(
                mu=None,
                sigma=None,
                transforms=UnsupportedMetadataTransformForCacheTest(2.0),
                data_dir=data_dir,
                split="train",
                cache_dir=cache_dir,
            )

    def test_unknown_callable_without_public_attrs_does_not_trust_cache(
        self,
        split_dirs,
        monkeypatch,
    ):
        """Verify callable objects without public state are non-cacheable."""
        data_dir, cache_dir = split_dirs
        cache_path = os.path.join(
            cache_dir,
            "standardize_tdigest_4_train_NoPublicAttrsCallableForCacheTest.json",
        )

        StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=NoPublicAttrsCallableForCacheTest(),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )
        assert not os.path.exists(cache_path)

        digest = TDigest()
        digest.batch_update(np.array([100.0, 101.0]))
        digest_payload = digest.to_dict()
        digest_payload["_standardize_transform_metadata"] = {
            "type": "object",
            "module": NoPublicAttrsCallableForCacheTest.__module__,
            "class": NoPublicAttrsCallableForCacheTest.__qualname__,
            "attrs": {},
        }
        with open(cache_path, "w") as fp:
            json.dump(digest_payload, fp, indent=2)
        assert os.path.exists(cache_path)

        def fail_build_tdigest(self):
            """Fail when cache lookup correctly falls through to rebuilding."""
            raise AssertionError("cache was intentionally not trusted")

        monkeypatch.setattr(
            StandardizeTransform, "_build_tdigest", fail_build_tdigest
        )

        with pytest.raises(AssertionError, match="cache was intentionally not trusted"):
            StandardizeTransform(
                mu=None,
                sigma=None,
                transforms=NoPublicAttrsCallableForCacheTest(),
                data_dir=data_dir,
                split="train",
                cache_dir=cache_dir,
            )

    def test_legacy_cache_without_metadata_is_rebuilt(self, split_dirs):
        """Verify metadata-free legacy cache payloads are overwritten."""
        data_dir, cache_dir = split_dirs
        cache_path = os.path.join(
            cache_dir,
            "standardize_tdigest_4_train_ScaleImageForCacheTest.json",
        )

        legacy = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=ScaleImageForCacheTest(2.0),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )
        with open(cache_path, "r") as fp:
            legacy_payload = json.load(fp)
        legacy_payload.pop("_standardize_transform_metadata")
        with open(cache_path, "w") as fp:
            json.dump(legacy_payload, fp, indent=2)

        rebuilt = StandardizeTransform(
            mu=None,
            sigma=None,
            transforms=ScaleImageForCacheTest(3.0),
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )

        np.testing.assert_allclose(legacy.mu, 4.0)
        np.testing.assert_allclose(rebuilt.mu, 6.0)
        with open(cache_path, "r") as fp:
            rebuilt_payload = json.load(fp)
        assert "_standardize_transform_metadata" in rebuilt_payload

    def test_creates_missing_cache_dir(self, split_dirs):
        """Verify StandardizeTransform creates cache_dir before writing JSON."""
        data_dir, _ = split_dirs
        cache_dir = os.path.join(os.path.dirname(data_dir), "new_cache")

        StandardizeTransform(
            mu=None,
            sigma=None,
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
        )

        assert os.path.isfile(
            os.path.join(cache_dir, "standardize_tdigest_4_train_function.json")
        )


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


class TestIdentityPicklable:
    """Verify the identity transform fallback is picklable."""

    def test_identity_is_picklable(self):
        """Default transforms must be picklable for multiprocessing."""
        import pickle
        t = ArcsinhStretch(scale=1.0)
        pickled = pickle.dumps(t.transforms)
        restored = pickle.loads(pickled)
        img = np.ones((1, 4, 4))
        np.testing.assert_array_equal(restored(img), img)


class TestBuildTdigest:
    """Tests for the shared build_tdigest function."""

    @pytest.fixture
    def dataset_dir(self, tmp_path):
        """Create a small dataset with 4 .npy files and metadata.csv."""
        import pandas as pd
        records = []
        for i in range(4):
            name = f"galaxy_{i:05d}.npy"
            np.save(tmp_path / name, np.full((1, 4, 4), float(i)))
            records.append({"filename": name, "split": "train"})
        pd.DataFrame(records).to_csv(tmp_path / "metadata.csv", index=False)
        return str(tmp_path)

    def test_serial_positive_filter(self, dataset_dir):
        """Verify build_tdigest with positive filter matches ArcsinhStretch behavior."""
        import pandas as pd
        metadata = pd.read_csv(os.path.join(dataset_dir, "metadata.csv"))
        filenames = metadata["filename"].tolist()
        digest = build_tdigest(
            data_dir=dataset_dir,
            filenames=filenames,
            transforms=_identity,
            pixel_filter=_filter_positive,
        )
        # galaxy_0 is all 0.0 (filtered out), galaxy_1/2/3 are 1.0/2.0/3.0
        np.testing.assert_allclose(digest.percentile(50), 2.0, atol=0.1)

    def test_serial_flatten_filter(self, dataset_dir):
        """Verify build_tdigest with flatten filter matches GlobalNorm behavior."""
        import pandas as pd
        metadata = pd.read_csv(os.path.join(dataset_dir, "metadata.csv"))
        filenames = metadata["filename"].tolist()
        digest = build_tdigest(
            data_dir=dataset_dir,
            filenames=filenames,
            transforms=_identity,
            pixel_filter=_flatten,
        )
        # All pixels: 0.0, 1.0, 2.0, 3.0 (16 pixels each)
        np.testing.assert_allclose(digest.min(), 0.0)
        np.testing.assert_allclose(digest.max(), 3.0)

    def test_serial_shows_progress_bar(self, dataset_dir, capsys):
        """Verify sequential build_tdigest renders a tqdm progress bar to stderr."""
        import pandas as pd
        metadata = pd.read_csv(os.path.join(dataset_dir, "metadata.csv"))
        filenames = metadata["filename"].tolist()
        build_tdigest(
            data_dir=dataset_dir,
            filenames=filenames,
            transforms=_identity,
            pixel_filter=_filter_positive,
            n_workers=0,
        )
        captured = capsys.readouterr()
        assert "Building TDigest" in captured.err
        assert "file" in captured.err


class TestBuildTdigestSampling:
    """Tests for sampling support in build_tdigest."""

    @pytest.fixture
    def large_dataset(self, tmp_path):
        """Create a dataset with 20 files for sampling tests."""
        import pandas as pd
        records = []
        for i in range(20):
            name = f"galaxy_{i:05d}.npy"
            np.save(tmp_path / name, np.full((1, 4, 4), float(i)))
            records.append({"filename": name, "split": "train"})
        pd.DataFrame(records).to_csv(tmp_path / "metadata.csv", index=False)
        return str(tmp_path)

    def test_sampling_uses_subset(self, large_dataset):
        """Verify sample_fraction reduces the number of files processed."""
        t = ArcsinhStretch(
            scale=None, percentile=50, data_dir=large_dataset, split="train",
            sample_fraction=0.5, sample_seed=42,
        )
        # Should succeed without error; scale should be a positive number
        assert t.scale > 0

    def test_sampling_is_reproducible(self, large_dataset):
        """Verify same seed produces same scale."""
        t1 = ArcsinhStretch(
            scale=None, percentile=50, data_dir=large_dataset, split="train",
            sample_fraction=0.5, sample_seed=42,
        )
        # Remove cache to force recomputation
        import glob
        for f in glob.glob(os.path.join(large_dataset, "arcsinh_tdigest*.json")):
            os.remove(f)
        t2 = ArcsinhStretch(
            scale=None, percentile=50, data_dir=large_dataset, split="train",
            sample_fraction=0.5, sample_seed=42,
        )
        np.testing.assert_allclose(t1.scale, t2.scale)

    def test_sampling_cache_filename_encodes_params(self, large_dataset):
        """Verify cache filename includes image size, sample_fraction and sample_seed."""
        ArcsinhStretch(
            scale=None, percentile=50, data_dir=large_dataset, split="train",
            sample_fraction=0.1, sample_seed=99,
        )
        assert os.path.isfile(
            os.path.join(
                large_dataset, "arcsinh_tdigest_4_train_s0.1_seed99_function.json"
            )
        )

    def test_no_sampling_cache_filename_unchanged(self, large_dataset):
        """Verify no sampling produces the cache filename without sampling suffix."""
        ArcsinhStretch(
            scale=None, percentile=50, data_dir=large_dataset, split="train",
        )
        assert os.path.isfile(
            os.path.join(large_dataset, "arcsinh_tdigest_4_train_function.json")
        )

    def test_global_norm_sampling_cache_filename(self, large_dataset):
        """Verify GlobalNorm cache filename encodes image size and sampling params."""
        GlobalNorm(
            global_min=None, global_max=None,
            data_dir=large_dataset, percentile=50, split="train",
            sample_fraction=0.2, sample_seed=7,
        )
        assert os.path.isfile(
            os.path.join(
                large_dataset,
                "global_norm_tdigest_50_4_train_s0.2_seed7_function.json",
            )
        )


class TestBuildTdigestParallel:
    """Tests for multiprocessing support in build_tdigest."""

    @pytest.fixture
    def dataset_dir(self, tmp_path):
        """Create a dataset with 8 files for parallel tests."""
        import pandas as pd
        records = []
        for i in range(8):
            name = f"galaxy_{i:05d}.npy"
            np.save(tmp_path / name, np.full((1, 4, 4), float(i + 1)))
            records.append({"filename": name, "split": "train"})
        pd.DataFrame(records).to_csv(tmp_path / "metadata.csv", index=False)
        return str(tmp_path)

    def test_parallel_matches_serial_arcsinh(self, dataset_dir):
        """Verify parallel ArcsinhStretch produces same scale as serial."""
        serial = ArcsinhStretch(
            scale=None, percentile=50, data_dir=dataset_dir, split="train",
            n_workers=0,
        )
        # Remove cache to force recomputation
        import glob
        for f in glob.glob(os.path.join(dataset_dir, "arcsinh_tdigest*.json")):
            os.remove(f)
        parallel = ArcsinhStretch(
            scale=None, percentile=50, data_dir=dataset_dir, split="train",
            n_workers=2,
        )
        np.testing.assert_allclose(serial.scale, parallel.scale)

    def test_parallel_matches_serial_global_norm(self, dataset_dir):
        """Verify parallel GlobalNorm produces same bounds as serial."""
        serial = GlobalNorm(
            global_min=None, global_max=None,
            data_dir=dataset_dir, percentile=50, split="train",
            n_workers=0,
        )
        import glob
        for f in glob.glob(os.path.join(dataset_dir, "global_norm_tdigest*.json")):
            os.remove(f)
        parallel = GlobalNorm(
            global_min=None, global_max=None,
            data_dir=dataset_dir, percentile=50, split="train",
            n_workers=2,
        )
        np.testing.assert_allclose(serial.global_min, parallel.global_min)
        np.testing.assert_allclose(serial.global_max, parallel.global_max)

    def test_parallel_shows_progress_bar(self, dataset_dir, capsys):
        """Verify parallel build_tdigest renders a tqdm progress bar to stderr."""
        import pandas as pd
        metadata = pd.read_csv(os.path.join(dataset_dir, "metadata.csv"))
        filenames = metadata["filename"].tolist()
        build_tdigest(
            data_dir=dataset_dir,
            filenames=filenames,
            transforms=_identity,
            pixel_filter=_filter_positive,
            n_workers=2,
        )
        captured = capsys.readouterr()
        assert "Building TDigest" in captured.err
        assert "file" in captured.err


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


class TestCacheDir:
    """Tests for separate cache_dir parameter."""

    @pytest.fixture
    def split_dirs(self, tmp_path):
        """Create data_dir with .npy files and a separate empty cache_dir."""
        import pandas as pd
        data_dir = tmp_path / "data"
        cache_dir = tmp_path / "cache"
        data_dir.mkdir()
        cache_dir.mkdir()

        records = []
        for i in range(3):
            name = f"galaxy_{i:05d}.npy"
            np.save(data_dir / name, np.full((1, 4, 4), float(i + 1)))
            records.append({"filename": name, "split": "train"})
        pd.DataFrame(records).to_csv(data_dir / "metadata.csv", index=False)
        return str(data_dir), str(cache_dir)

    def test_arcsinh_writes_cache_to_cache_dir(self, split_dirs):
        """Verify ArcsinhStretch writes tdigest JSON to cache_dir, not data_dir."""
        data_dir, cache_dir = split_dirs
        ArcsinhStretch(
            scale=None, percentile=50, data_dir=data_dir,
            split="train", cache_dir=cache_dir,
        )
        assert os.path.isfile(
            os.path.join(cache_dir, "arcsinh_tdigest_4_train_function.json")
        )
        assert not os.path.isfile(
            os.path.join(data_dir, "arcsinh_tdigest_4_train_function.json")
        )

    def test_global_norm_writes_cache_to_cache_dir(self, split_dirs):
        """Verify GlobalNorm writes tdigest JSON to cache_dir, not data_dir."""
        data_dir, cache_dir = split_dirs
        GlobalNorm(
            global_min=None, global_max=None,
            data_dir=data_dir, percentile=50,
            split="train", cache_dir=cache_dir,
        )
        assert os.path.isfile(
            os.path.join(cache_dir, "global_norm_tdigest_50_4_train_function.json")
        )
        assert not os.path.isfile(
            os.path.join(data_dir, "global_norm_tdigest_50_4_train_function.json")
        )

    def test_cache_dir_none_falls_back_to_data_dir(self, split_dirs):
        """Verify cache_dir=None writes to data_dir (backward compat)."""
        data_dir, _ = split_dirs
        ArcsinhStretch(
            scale=None, percentile=50, data_dir=data_dir,
            split="train", cache_dir=None,
        )
        assert os.path.isfile(
            os.path.join(data_dir, "arcsinh_tdigest_4_train_function.json")
        )

    def test_cache_dir_reads_existing_cache(self, split_dirs):
        """Verify second instantiation reads from cache_dir without recomputing."""
        data_dir, cache_dir = split_dirs
        t1 = ArcsinhStretch(
            scale=None, percentile=50, data_dir=data_dir,
            split="train", cache_dir=cache_dir,
        )
        t2 = ArcsinhStretch(
            scale=None, percentile=50, data_dir=data_dir,
            split="train", cache_dir=cache_dir,
        )
        np.testing.assert_allclose(t1.scale, t2.scale)


def test_cluster_clip_is_importable():
    """ClusterClip and its helpers are exported from msdflow.data.preprocess."""
    from msdflow.data.preprocess import (
        ClusterClip,
        build_image_percentiles,
        _worker_image_percentile,
    )
    assert ClusterClip is not None
    assert build_image_percentiles is not None
    assert _worker_image_percentile is not None


class TestWorkerImagePercentile:
    """Tests for _worker_image_percentile."""

    def test_returns_percentile_of_image(self, tmp_path):
        """Worker returns the percentile of all pixels in the loaded image."""
        from msdflow.data.preprocess import _worker_image_percentile, _identity

        name = "galaxy_00000.npy"
        img = np.arange(100, dtype=float).reshape(1, 10, 10)
        np.save(tmp_path / name, img)

        out = _worker_image_percentile(
            (str(tmp_path), name, _identity, 50.0, False)
        )
        np.testing.assert_allclose(out, np.percentile(img, 50.0))

    def test_positive_only_filters_zeros(self, tmp_path):
        """When positive_only=True, only positive pixels contribute."""
        from msdflow.data.preprocess import _worker_image_percentile, _identity

        name = "galaxy_00000.npy"
        # Most pixels are zero; one positive pixel
        img = np.zeros((1, 4, 4))
        img[0, 0, 0] = 7.0
        np.save(tmp_path / name, img)

        all_pixels = _worker_image_percentile(
            (str(tmp_path), name, _identity, 99.0, False)
        )
        positive_only = _worker_image_percentile(
            (str(tmp_path), name, _identity, 99.0, True)
        )
        # all-pixels: most values are 0, so 99th percentile is ~0
        # positive-only: only one value (7.0), so 99th percentile is 7.0
        assert all_pixels < positive_only
        np.testing.assert_allclose(positive_only, 7.0)

    def test_returns_python_float(self, tmp_path):
        """Worker returns a builtin float (picklable across processes)."""
        from msdflow.data.preprocess import _worker_image_percentile, _identity

        name = "galaxy_00000.npy"
        np.save(tmp_path / name, np.ones((1, 4, 4)))
        out = _worker_image_percentile(
            (str(tmp_path), name, _identity, 50.0, False)
        )
        assert type(out) is float


class TestBuildImagePercentiles:
    """Tests for build_image_percentiles."""

    @pytest.fixture
    def small_dataset(self, tmp_path):
        """Create 4 .npy files with controlled per-image percentiles."""
        names = []
        # Each file is constant-valued so its percentile equals that value
        for i, value in enumerate([1.0, 2.0, 3.0, 4.0]):
            name = f"galaxy_{i:05d}.npy"
            np.save(tmp_path / name, np.full((1, 4, 4), value))
            names.append(name)
        return str(tmp_path), names

    def test_serial_returns_one_value_per_file(self, small_dataset):
        """Serial path returns a 1-D array of length len(filenames)."""
        from msdflow.data.preprocess import build_image_percentiles, _identity

        data_dir, names = small_dataset
        out = build_image_percentiles(
            data_dir=data_dir,
            filenames=names,
            transforms=_identity,
            percentile=50.0,
            positive_only=False,
            n_workers=0,
        )
        assert out.shape == (4,)
        # Constant images: 50th percentile equals the value itself
        np.testing.assert_allclose(sorted(out.tolist()), [1.0, 2.0, 3.0, 4.0])

    def test_parallel_returns_same_set(self, small_dataset):
        """Parallel path returns the same set of values as serial."""
        from msdflow.data.preprocess import build_image_percentiles, _identity

        data_dir, names = small_dataset
        serial = build_image_percentiles(
            data_dir=data_dir,
            filenames=names,
            transforms=_identity,
            percentile=50.0,
            positive_only=False,
            n_workers=0,
        )
        parallel = build_image_percentiles(
            data_dir=data_dir,
            filenames=names,
            transforms=_identity,
            percentile=50.0,
            positive_only=False,
            n_workers=2,
        )
        # Order may differ under imap_unordered; compare as multisets.
        np.testing.assert_allclose(sorted(serial.tolist()), sorted(parallel.tolist()))


class TestClusterClipDirect:
    """Tests for ClusterClip in direct (explicit clip) mode."""

    def test_call_clips_to_zero_max(self):
        """Default min=0; values above clip are capped, below 0 are zeroed."""
        from msdflow.data.preprocess import ClusterClip

        t = ClusterClip(clip=5.0)
        img = np.array([[[-2.0, 0.0, 1.0, 5.0, 10.0]]])
        out = t(img)
        np.testing.assert_array_equal(out, [[[0.0, 0.0, 1.0, 5.0, 5.0]]])

    def test_call_uses_custom_min(self):
        """Custom min argument is honored as a_min."""
        from msdflow.data.preprocess import ClusterClip

        t = ClusterClip(clip=5.0, min=-1.0)
        img = np.array([[[-2.0, -1.0, 0.0, 5.0, 10.0]]])
        out = t(img)
        np.testing.assert_array_equal(out, [[[-1.0, -1.0, 0.0, 5.0, 5.0]]])

    def test_preserves_shape(self):
        """Output shape matches input."""
        from msdflow.data.preprocess import ClusterClip

        t = ClusterClip(clip=1.0)
        img = np.ones((3, 64, 64)) * 5.0
        out = t(img)
        assert out.shape == (3, 64, 64)
        np.testing.assert_array_equal(out, np.ones((3, 64, 64)))

    def test_xor_validation_both_set(self):
        """Passing both clip and percentile raises ValueError."""
        from msdflow.data.preprocess import ClusterClip

        with pytest.raises(ValueError):
            ClusterClip(clip=1.0, percentile=99.0, data_dir="/nonexistent")

    def test_xor_validation_neither_set(self):
        """Passing neither clip nor percentile raises ValueError."""
        from msdflow.data.preprocess import ClusterClip

        with pytest.raises(ValueError):
            ClusterClip()

    def test_clip_zero_is_valid(self):
        """clip=0.0 is a legal direct value (not treated as 'unset')."""
        from msdflow.data.preprocess import ClusterClip

        t = ClusterClip(clip=0.0)
        img = np.array([[[-1.0, 0.0, 1.0]]])
        out = t(img)
        # min=0 default and max=0 → everything collapses to 0
        np.testing.assert_array_equal(out, [[[0.0, 0.0, 0.0]]])


class TestClusterClipDerived:
    """Tests for ClusterClip derived (KMeans) mode."""

    @pytest.fixture
    def bimodal_dataset(self, tmp_path):
        """Two clear clusters of per-image 99th percentiles: ~1 and ~10."""
        import pandas as pd
        records = []
        # 5 "dim" train images: 99th percentile ≈ 1.0
        for i in range(5):
            name = f"galaxy_dim_{i:05d}.npy"
            np.save(tmp_path / name, np.full((1, 4, 4), 1.0))
            records.append({"filename": name, "split": "train"})
        # 5 "bright" train images: 99th percentile ≈ 10.0
        for i in range(5):
            name = f"galaxy_bright_{i:05d}.npy"
            np.save(tmp_path / name, np.full((1, 4, 4), 10.0))
            records.append({"filename": name, "split": "train"})
        # 1 val image (must be ignored)
        name = "galaxy_val_00000.npy"
        np.save(tmp_path / name, np.full((1, 4, 4), 1000.0))
        records.append({"filename": name, "split": "val"})
        pd.DataFrame(records).to_csv(tmp_path / "metadata.csv", index=False)
        return str(tmp_path)

    def test_derived_clip_is_midpoint_of_cluster_centres(self, bimodal_dataset):
        """Fitted self.max is the mean of the two KMeans centroids."""
        from msdflow.data.preprocess import ClusterClip

        t = ClusterClip(
            percentile=99.0,
            data_dir=bimodal_dataset,
            split="train",
        )
        # Clustering happens in log10 space, so the boundary in linear space
        # is sqrt(10) (geometric mean of 1.0 and 10.0).
        np.testing.assert_allclose(t.max, np.sqrt(10.0), rtol=1e-3)

    def test_derived_min_default_is_zero(self, bimodal_dataset):
        """Default min stays 0.0 in derived mode."""
        from msdflow.data.preprocess import ClusterClip

        t = ClusterClip(
            percentile=99.0,
            data_dir=bimodal_dataset,
            split="train",
        )
        assert t.min == 0.0

    def test_derived_split_is_respected(self, bimodal_dataset):
        """Val image (value 1000) does not influence the fit."""
        from msdflow.data.preprocess import ClusterClip

        t = ClusterClip(
            percentile=99.0,
            data_dir=bimodal_dataset,
            split="train",
        )
        # If the val image had been included, midpoint would be much higher.
        assert t.max < 100.0

    def test_cache_file_is_written_and_reused(self, bimodal_dataset, tmp_path):
        """Second construction reads the cache file (proven by removing fit data)."""
        from msdflow.data.preprocess import ClusterClip
        import pandas as pd
        import shutil

        t1 = ClusterClip(
            percentile=99.0,
            data_dir=bimodal_dataset,
            split="train",
        )
        first_max = t1.max

        # Move most .npy files away so KMeans cannot be refit from disk;
        # keep ONLY the first metadata entry so the img_size probe still
        # succeeds. If the cache is read, that one file is irrelevant; if
        # the cache is ignored, KMeans on a single point would not produce
        # the same midpoint, exposing a regression.
        metadata = pd.read_csv(os.path.join(bimodal_dataset, "metadata.csv"))
        keep = metadata["filename"].iloc[0]
        scratch = tmp_path / "moved"
        scratch.mkdir()
        for entry in os.listdir(bimodal_dataset):
            if entry.endswith(".npy") and entry != keep:
                shutil.move(
                    os.path.join(bimodal_dataset, entry),
                    scratch / entry,
                )

        t2 = ClusterClip(
            percentile=99.0,
            data_dir=bimodal_dataset,
            split="train",
        )
        assert t2.max == first_max

    def test_cache_filename_encodes_percentile_and_positive_only(
        self, bimodal_dataset
    ):
        """Cache filenames differ across percentile/positive_only combinations."""
        from msdflow.data.preprocess import ClusterClip

        ClusterClip(
            percentile=99.0,
            data_dir=bimodal_dataset,
            split="train",
            positive_only=False,
        )
        ClusterClip(
            percentile=99.0,
            data_dir=bimodal_dataset,
            split="train",
            positive_only=True,
        )
        files = sorted(
            f for f in os.listdir(bimodal_dataset) if f.startswith("cluster_clip_")
        )
        # Two distinct cache files — one per positive_only setting.
        assert len(files) == 2
        assert any("_pos" in f for f in files)
        assert any("_pos" not in f for f in files)

    def test_positive_only_changes_result_when_zeros_dominate(self, tmp_path):
        """With many zeros, positive_only=True yields a different (larger) clip."""
        from msdflow.data.preprocess import ClusterClip
        import pandas as pd

        records = []
        # 5 "dim" images: one positive pixel at 1.0, rest zero
        for i in range(5):
            name = f"galaxy_dim_{i:05d}.npy"
            img = np.zeros((1, 4, 4))
            img[0, 0, 0] = 1.0
            np.save(tmp_path / name, img)
            records.append({"filename": name, "split": "train"})
        # 5 "bright" images: one positive pixel at 10.0, rest zero
        for i in range(5):
            name = f"galaxy_bright_{i:05d}.npy"
            img = np.zeros((1, 4, 4))
            img[0, 0, 0] = 10.0
            np.save(tmp_path / name, img)
            records.append({"filename": name, "split": "train"})
        pd.DataFrame(records).to_csv(tmp_path / "metadata.csv", index=False)
        data_dir = str(tmp_path)

        # With all pixels: 99th percentile of [0,0,...,0,v] is dominated by zeros,
        # so both clusters collapse near 0 and midpoint is small.
        t_all = ClusterClip(
            percentile=99.0,
            data_dir=data_dir,
            split="train",
            positive_only=False,
        )
        # With positive_only: 99th percentile equals the single positive value,
        # so clusters split cleanly at 1 and 10, boundary = sqrt(10)
        # (geometric mean from log10-space midpoint).
        t_pos = ClusterClip(
            percentile=99.0,
            data_dir=data_dir,
            split="train",
            positive_only=True,
        )
        assert t_pos.max > t_all.max
        np.testing.assert_allclose(t_pos.max, np.sqrt(10.0), rtol=1e-3)


def test_clr_transform_config_defines_standardized_pipeline():
    """Verify the CLR transform config wires the expected deterministic pipeline."""
    cfg = OmegaConf.load("configs/data/transforms/clr.yaml")
    raw = OmegaConf.to_container(cfg, resolve=False)

    pre_clr_transforms = raw["pre_clr_transforms"]["transforms"]
    assert [
        transform["_target_"] for transform in pre_clr_transforms
    ] == [
        "msdflow.data.preprocess.SurfaceBrightnessToNanomaggies",
        "msdflow.data.preprocess.ClipAndPad",
        "msdflow.data.preprocess.Downsample",
        "msdflow.data.preprocess.PDFNorm",
    ]
    assert pre_clr_transforms[0]["mag_threshold"] == 99.0
    assert pre_clr_transforms[1]["n"] == "${clip_pad_size}"
    assert pre_clr_transforms[2]["target_size"] == "${image_size}"
    assert raw["sample_fraction"] == 0.1
    assert raw["sample_seed"] == 42
    assert raw["n_workers"] == 10

    clr_cfg = raw["clr_transform"]["transforms"][0]
    assert clr_cfg["_target_"] == "msdflow.data.preprocess.CLRTransform"
    assert clr_cfg["eps_mass"] == 1e-6

    standardize_cfg = raw["standardize_transform"]["transforms"][0]
    assert standardize_cfg["_target_"] == "msdflow.data.preprocess.StandardizeTransform"
    assert standardize_cfg["mu"] is None
    assert standardize_cfg["sigma"] is None
    assert standardize_cfg["data_dir"] == "${data.dataloader.data_dir}"
    assert standardize_cfg["cache_dir"] == "${data.dataloader.cache_dir}"
    assert standardize_cfg["transforms"]["transforms"] == [
        "${data.dataloader.transforms.pre_clr_transforms}",
        "${data.dataloader.transforms.clr_transform}",
    ]
    assert standardize_cfg["split"] == "train"
    assert (
        standardize_cfg["sample_fraction"]
        == "${data.dataloader.transforms.sample_fraction}"
    )
    assert standardize_cfg["sample_seed"] == "${data.dataloader.transforms.sample_seed}"
    assert standardize_cfg["n_workers"] == "${data.dataloader.transforms.n_workers}"

    assert raw["deterministic"]["transforms"] == [
        "${data.dataloader.transforms.pre_clr_transforms}",
        "${data.dataloader.transforms.clr_transform}",
        "${data.dataloader.transforms.standardize_transform}",
    ]


def test_clr_transform_config_instantiates_val_dataset(tmp_path):
    """Verify Hydra can instantiate the CLR val dataset and fit standardization."""
    import pandas as pd

    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir()
    cache_dir.mkdir()

    pattern = np.array(
        [
            [22.0, 22.1, 22.2, 22.3],
            [22.2, 22.0, 22.3, 22.1],
            [22.4, 22.1, 22.0, 22.2],
            [22.3, 22.2, 22.1, 22.0],
        ],
        dtype=np.float32,
    )
    records = [
        ("galaxy_train_00000.npy", "train", pattern),
        ("galaxy_train_00001.npy", "train", pattern.T + 0.05),
        ("galaxy_val_00000.npy", "val", np.flip(pattern, axis=0) + 0.1),
    ]
    for filename, split, image in records:
        np.save(data_dir / filename, image[np.newaxis, :, :])
    pd.DataFrame(
        [{"filename": filename, "split": split} for filename, split, _ in records]
    ).to_csv(data_dir / "metadata.csv", index=False)

    with initialize_config_dir(
        config_dir=str(Path("configs").resolve()),
        version_base=None,
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "data/transforms@data.dataloader.transforms=clr",
                f"data.dataloader.data_dir={data_dir}",
                f"data.dataloader.cache_dir={cache_dir}",
                "clip_pad_size=4",
                "image_size=4",
                "data.dataloader.transforms.n_workers=0",
                "data.dataloader.transforms.sample_fraction=null",
            ],
        )

    dataset = instantiate(cfg.data.dataloader.val_dataset)
    image, _ = dataset[0]

    assert image.shape == (1, 4, 4)
    assert np.isfinite(image.numpy()).all()
    assert list(cache_dir.glob("standardize_tdigest*.json"))
