"""Tests for src.data.preprocess."""

import numpy as np
import pytest

from torchvision.transforms import Compose

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
