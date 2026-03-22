"""Tests for src.data.preprocess."""

import numpy as np
import pytest

try:
    from src.data.preprocess import (
        arcsinh_stretch,
        clip_and_pad,
        linear_normalize,
        preprocess_image,
        surface_brightness_to_nanomaggies,
    )
except (ImportError, AttributeError):
    pytest.skip(
        "astropy import failed (likely numpy>=2.0 incompatibility)",
        allow_module_level=True,
    )


# ----------------------------- clip_and_pad --------------------------------- #


def test_clip_and_pad_small_image_pads_to_target():
    """Verify a smaller-than-target image is padded and cropped to (n, n)."""
    img = np.ones((100, 100))
    out = clip_and_pad(img, n=256)
    assert out.shape == (256, 256)


def test_clip_and_pad_large_image_crops_to_target():
    """Verify a larger-than-target image is centre-cropped to (n, n)."""
    img = np.ones((600, 600))
    out = clip_and_pad(img, n=256)
    assert out.shape == (256, 256)


def test_clip_and_pad_exact_size_is_identity():
    """Verify an image already at target size is returned unchanged."""
    img = np.arange(16).reshape(4, 4).astype(float)
    out = clip_and_pad(img, n=4)
    np.testing.assert_array_equal(out, img)


def test_clip_and_pad_default_n_is_512():
    """Verify the default target size is 512."""
    img = np.ones((512, 512))
    out = clip_and_pad(img)
    assert out.shape == (512, 512)


def test_clip_and_pad_nonsquare_input():
    """Verify a non-square input is padded and cropped correctly."""
    img = np.ones((100, 300))
    out = clip_and_pad(img, n=256)
    assert out.shape == (256, 256)


# ----------------------- surface_brightness_to_nanomaggies ------------------ #


def test_sb_to_nanomaggies_known_value():
    """Verify mag=22.5 converts to 1.0 nanomaggy."""
    img = np.array([[22.5]])
    out = surface_brightness_to_nanomaggies(img)
    np.testing.assert_allclose(out, [[1.0]])


def test_sb_to_nanomaggies_brighter_is_higher_flux():
    """Verify brighter magnitudes (lower values) produce higher flux."""
    img = np.array([[20.0, 22.5]])
    out = surface_brightness_to_nanomaggies(img)
    assert out[0, 0] > out[0, 1]


def test_sb_to_nanomaggies_above_threshold_zeroed():
    """Verify pixels at or above threshold are zeroed out."""
    img = np.array([[99.0, 100.0]])
    out = surface_brightness_to_nanomaggies(img, mag_threshold=99.0)
    np.testing.assert_array_equal(out, [[0.0, 0.0]])


def test_sb_to_nanomaggies_output_non_negative():
    """Verify all flux values are non-negative."""
    rng = np.random.default_rng(0)
    img = rng.uniform(15, 30, size=(64, 64))
    out = surface_brightness_to_nanomaggies(img)
    assert np.all(out >= 0)


# ----------------------------- arcsinh_stretch ------------------------------ #


def test_arcsinh_stretch_known_value():
    """Verify arcsinh stretch matches np.arcsinh(x/a)."""
    imgs = np.array([0.0, 1.0, 10.0])
    a = 2.0
    out = arcsinh_stretch(imgs, a)
    np.testing.assert_allclose(out, np.arcsinh(imgs / a))


def test_arcsinh_stretch_preserves_shape():
    """Verify output shape matches input shape."""
    imgs = np.ones((3, 64, 64))
    out = arcsinh_stretch(imgs, a=1.0)
    assert out.shape == imgs.shape


def test_arcsinh_stretch_zero_input():
    """Verify zero input produces zero output."""
    imgs = np.zeros((4, 4))
    out = arcsinh_stretch(imgs, a=1.0)
    np.testing.assert_array_equal(out, 0.0)


# ----------------------------- linear_normalize ----------------------------- #


def test_linear_normalize_maps_endpoints():
    """Verify data_min maps to norm_min and data_max maps to norm_max."""
    data = np.array([0.0, 1.0])
    out = linear_normalize(data, 0.0, 1.0, -1.0, 1.0)
    np.testing.assert_allclose(out, [-1.0, 1.0])


def test_linear_normalize_midpoint():
    """Verify the midpoint of the input range maps to the midpoint of the output range."""
    data = np.array([0.5])
    out = linear_normalize(data, 0.0, 1.0, 0.0, 10.0)
    np.testing.assert_allclose(out, [5.0])


def test_linear_normalize_preserves_shape():
    """Verify output shape matches input shape."""
    data = np.ones((3, 4))
    out = linear_normalize(data, 0.0, 2.0, -1.0, 1.0)
    assert out.shape == data.shape


# ----------------------------- preprocess_image ----------------------------- #


def test_preprocess_image_output_shape():
    """Verify preprocessed output is 512x512."""
    img = np.full((400, 400), 22.0)
    out = preprocess_image(img, percentile=99.9, norm_range=(-1, 1))
    assert out.shape == (512, 512)


def test_preprocess_image_output_in_norm_range():
    """Verify output values fall within the specified normalisation range."""
    img = np.full((512, 512), 21.0)
    out = preprocess_image(img, percentile=99.9, norm_range=(0, 1))
    assert out.min() >= -0.01  # small tolerance for float
    assert out.max() <= 1.01
