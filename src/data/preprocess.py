"""Preprocessing pipeline for TNG50 galaxy images.

Converts raw FITS surface-brightness maps to normalised tensors:
``load_fits`` → ``surface_brightness_to_nanomaggies`` → ``clip_and_pad``
→ ``arcsinh_stretch`` → ``np.clip`` → ``linear_normalize``.
"""

import hydra
import logging
import astropy

import numpy as np
import astropy.units as u
import astropy.cosmology as ap_cosmo

from astropy.io import fits
from omegaconf import DictConfig, OmegaConf

# ------------------------------------ I/O ----------------------------------- #


def load_fits(filename: str, band: str) -> tuple[np.ndarray, dict]:
    """Load a single band from a multi-extension FITS file.

    Args:
        filename: Path to the FITS file.
        band: ``EXTNAME`` value to match (e.g. ``'g'``).

    Returns:
        Tuple of the image data array and its FITS header.

    Raises:
        ValueError: If no extension matches *band*.
    """
    with fits.open(filename) as hdul:
        for hdu in hdul:
            if hdu.header.get("EXTNAME") == band:
                return hdu.data, hdu.header
    raise ValueError(f"Band '{band}' not found in {filename}")


# ----------------------------- IMAGE TRANSFORMS ----------------------------- #


def clip_and_pad(img: np.ndarray, n: int = 512) -> np.ndarray:
    """Pad an image to at least *n x n*, then centre-crop to exactly *n x n*.

    Args:
        img: 2-D image array.
        n: Target side length in pixels.

    Returns:
        Centre-cropped array of shape ``(n, n)``.
    """
    y_len, x_len = img.shape
    pad_y = max(0, n - y_len)
    pad_x = max(0, n - x_len)

    if pad_y > 0 or pad_x > 0:
        top, left = pad_y // 2, pad_x // 2
        img = np.pad(
            img,
            ((top, pad_y - top), (left, pad_x - left)),
            mode="constant",
            constant_values=0,
        )

    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    half = n // 2
    return img[cy - half : cy + half, cx - half : cx + half]


def surface_brightness_to_nanomaggies(
    image: np.ndarray,
    mag_threshold: float = 99.0,
) -> np.ndarray:
    """Convert a surface-brightness image (AB mag / pixel) to nanomaggies.

    Args:
        image: Surface-brightness array in AB magnitudes per pixel.
        mag_threshold: Pixels fainter than this value are zeroed.

    Returns:
        Flux array in nanomaggies.
    """
    flux = np.where(image < mag_threshold, 10.0 ** (0.4 * (22.5 - image)), 0.0)
    return flux


def arcsinh_stretch(imgs: np.ndarray, a: float) -> np.ndarray:
    """Apply an arcsinh stretch to compress dynamic range.

    Args:
        imgs: Input array (any shape).
        a: Softening parameter controlling the stretch.

    Returns:
        Stretched array with the same shape as *imgs*.
    """
    return np.arcsinh(imgs / a)


def linear_normalize(
    data: np.ndarray, data_min: float, data_max: float, norm_min: float, norm_max: float
) -> np.ndarray:
    """Linearly map data from ``[data_min, data_max]`` to ``[norm_min, norm_max]``.

    Args:
        data: Input array.
        data_min: Minimum of the input range.
        data_max: Maximum of the input range.
        norm_min: Minimum of the target range.
        norm_max: Maximum of the target range.

    Returns:
        Rescaled array.
    """

    norm_range = norm_max - norm_min
    data_fraction = (data - data_min) / (data_max - data_min)

    return norm_range * data_fraction + norm_min


def preprocess_image(
    img: np.ndarray,
    percentile: float,
    norm_range: tuple[float],
    stretch_scale: float = 1,
):
    """Run the full preprocessing pipeline on a single image.

    Applies flux conversion, padding/cropping, arcsinh stretch,
    percentile clipping, and linear normalisation.

    Args:
        img: Raw surface-brightness image (AB mag / pixel).
        percentile: Percentile used for clipping after stretch.
        norm_range: ``(min, max)`` target range for normalisation.
        stretch_scale: Softening parameter for ``arcsinh_stretch``.

    Returns:
        Preprocessed image array of shape ``(512, 512)``.
    """

    img = surface_brightness_to_nanomaggies(img)
    img = clip_and_pad(img)
    img = arcsinh_stretch(img, a=stretch_scale)
    imgp = np.percentile(img, percentile)
    img = np.clip(img / imgp, 0, 1.0)
    img = linear_normalize(
        img, img.min(), img.max(), norm_min=norm_range[0], norm_max=norm_range[1]
    )

    return img
