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
            pad_widths = [(0, 0)] * (img.ndim - 2) + [
                (top, pad_h - top),
                (left, pad_w - left),
            ]
            img = np.pad(img, pad_widths, mode="constant", constant_values=0)

        cy, cx = img.shape[-2] // 2, img.shape[-1] // 2
        half = n // 2
        return img[..., cy - half : cy + half, cx - half : cx + half]


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
