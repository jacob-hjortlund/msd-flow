"""Preprocessing transforms for TNG50 galaxy images.

Each transform is a callable class with ``__init__`` for parameters and
``__call__(img)`` operating on ``(C, H, W)`` NumPy arrays. Compose via
``torchvision.transforms.Compose``.
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from fastdigest import TDigest


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
        return np.where(img < self.mag_threshold, 10.0 ** (0.4 * (22.5 - img)), 0.0)


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


class ArcsinhStretch:
    """Apply arcsinh stretch to compress dynamic range.

    Computes ``arcsinh(img / scale)``. The ``scale`` parameter corresponds
    to the ``a`` parameter in the former ``arcsinh_stretch`` free function.

    Args:
        scale: Softening parameter controlling the stretch.
    """

    def __init__(
        self,
        scale: float | None = 1,
        transforms=None,
        percentile=None,
        data_dir: str = None,
    ):
        use_percentile = (percentile is not None) and (data_dir is not None)
        use_scale = scale is not None

        if (not use_percentile and not use_scale) or (use_percentile and use_scale):
            raise ValueError(
                "Must use either a provided scale or a provided percentile and dataset. "
                + "You have provided:\n"
                + f"   - scale: {scale}\n"
                + f"   - percentile: {percentile}\n"
                + f"   - data_dir: {data_dir}"
            )

        if transforms is None:
            transforms = lambda x: x
        self.transforms = transforms
        self.percentile = percentile
        self.data_dir = data_dir

        if use_scale:
            self.scale = scale

        if use_percentile:

            tdigest_path = os.path.join(data_dir, "arcsinh_tdigest.json")

            if os.path.isfile(tdigest_path):
                with open(tdigest_path, "r") as fp:
                    digest_dict = json.load(fp)
                digest = TDigest.from_dict(digest_dict)
            else:
                digest = self._build_tdigest()
                digest_dict = digest.to_dict()
                with open(tdigest_path, "w") as fp:
                    json.dump(digest_dict, fp, indent=2)

            self.scale = digest.percentile(self.percentile)

    def _build_tdigest(self):

        csv_path = os.path.join(self.data_dir, "metadata.csv")
        metadata = pd.read_csv(csv_path)
        filenames = metadata["filename"].tolist()

        digest = TDigest()

        for fn in tqdm(filenames):
            path = os.path.join(self.data_dir, fn)
            img = np.load(path)
            img = self.transforms(img)
            non_zero = img[img > 0]
            digest.batch_update(non_zero)

        return digest

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply arcsinh stretch.

        Args:
            img: ``(C, H, W)`` array.

        Returns:
            Stretched array, same shape as input.
        """
        return np.arcsinh(img / self.scale)


class GlobalNorm:
    """Linearly map a dataset from a global [min, max] to [norm_min, norm_max].

    This preserves relative concentration across the dataset while ensuring
    the output fits within a stable range (e.g., [-1, 1]) for generative
    training. Because the scale factors are global constants, this operation
    is perfectly mathematically invertible.
    """

    def __init__(
        self,
        global_min: float | None = None,
        global_max: float | None = None,
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        transforms=None,
        percentile=None,
        data_dir: str = None,
    ):

        if transforms is None:
            transforms = lambda x: x
        self.transforms = transforms
        self.data_dir = data_dir

        self.norm_min = norm_min
        self.norm_max = norm_max

        global_value_not_set = (global_min is None) or (global_max is None)

        if global_value_not_set:

            tdigest_path = os.path.join(
                data_dir, f"global_norm_tdigest_{percentile:0f}.json"
            )

            if os.path.isfile(tdigest_path):
                with open(tdigest_path, "r") as fp:
                    digest_dict = json.load(fp)
                digest = TDigest.from_dict(digest_dict)
            else:
                digest = self._build_tdigest()
                digest_dict = digest.to_dict()
                with open(tdigest_path, "w") as fp:
                    json.dump(digest_dict, fp, indent=2)

            if global_min is None:
                global_min = digest.min()

            if global_max is None:
                global_max = digest.max()

        self.global_min = global_min
        self.global_max = global_max

    def _build_tdigest(self):

        csv_path = os.path.join(self.data_dir, "metadata.csv")
        metadata = pd.read_csv(csv_path)
        filenames = metadata["filename"].tolist()

        digest = TDigest()

        for fn in tqdm(filenames):
            path = os.path.join(self.data_dir, fn)
            img = np.load(path)
            img = self.transforms(img)
            digest.batch_update(img.flatten())

        return digest

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply global linear normalization.

        Args:
            img: ``(C, H, W)`` array.

        Returns:
            Array mapped to ``[norm_min, norm_max]``, same shape as input.
        """
        # Scale to [0, 1] using global bounds
        img_norm = (img - self.global_min) / (self.global_max - self.global_min)

        # Scale to [norm_min, norm_max]
        return img_norm * (self.norm_max - self.norm_min) + self.norm_min


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

    .. note::
        Alternative transform, not part of the active pipeline.
        The active pipeline uses ``PDFNorm`` + ``GlobalNorm`` instead.

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
