"""Preprocessing transforms for TNG50 galaxy images.

Each transform is a callable class with ``__init__`` for parameters and
``__call__(img)`` operating on ``(C, H, W)`` NumPy arrays. Compose via
``torchvision.transforms.Compose``.
"""

import os
import json

import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from fastdigest import TDigest
from msdflow.data.random import WorkerSeededTransform


def _identity(x):
    """Identity transform (picklable replacement for ``lambda x: x``)."""
    return x


def _filter_positive(img: np.ndarray) -> np.ndarray:
    """Return only positive pixel values (flattened)."""
    return img[img > 0]


def _flatten(img: np.ndarray) -> np.ndarray:
    """Return all pixel values flattened."""
    return img.flatten()


def _process_single_file(data_dir, filename, transforms):

    path = os.path.join(data_dir, filename)
    img = np.load(path)
    img = transforms(img)

    return img


def _worker_single_file(args: tuple) -> TDigest:
    """Build a TDigest from a single .npy file.

    Args:
        args: Tuple of ``(data_dir, filename, transforms, pixel_filter)``.

    Returns:
        Fitted ``TDigest`` for this file.
    """
    data_dir, filename, transforms, pixel_filter = args
    img = _process_single_file(data_dir, filename, transforms)
    digest = TDigest()
    digest.batch_update(pixel_filter(img))
    return digest


def build_tdigest(
    data_dir: str,
    filenames: list[str],
    transforms,
    pixel_filter,
    n_workers: int = 0,
) -> TDigest:
    """Build a TDigest over pixel values from a list of ``.npy`` files.

    Args:
        data_dir: Directory containing the ``.npy`` files.
        filenames: List of filenames (relative to ``data_dir``).
        transforms: Preprocessing pipeline applied to each image.
        pixel_filter: Callable that selects/reshapes pixels from a
            transformed image into a 1-D array for the TDigest.
            Use ``_filter_positive`` for non-zero pixels or
            ``_flatten`` for all pixels.
        n_workers: Number of multiprocessing workers. ``0`` means serial.

    Returns:
        Fitted ``TDigest`` instance.
    """
    args = [(data_dir, fn, transforms, pixel_filter) for fn in filenames]

    if n_workers <= 0:
        result = TDigest()
        for digest in tqdm(
            map(_worker_single_file, args),
            total=len(filenames),
            desc="Building TDigest",
            unit="file",
        ):
            result = result.merge(digest)
        return result

    # ctx = mp.get_context("spawn")

    with mp.Pool(n_workers) as pool:
        result = TDigest()
        for digest in tqdm(
            pool.imap_unordered(_worker_single_file, args),
            total=len(filenames),
            desc="Building TDigest",
            unit="file",
        ):
            result = result.merge(digest)
    return result


def _tdigest_cache_suffix(
    img_size: int,
    split: str | None,
    sample_fraction: float | None = None,
    sample_seed: int = 42,
) -> str:
    """Build the suffix for a tdigest cache filename.

    Args:
        img_size: H/W of images used.
        split: Split name (e.g. ``"train"``).
        sample_fraction: Fraction of files sampled, or ``None`` for all.
        sample_seed: RNG seed used for sampling.

    Returns:
        Suffix string, e.g. ``"_train"`` or ``"_train_s0.1_seed42"``.
    """
    suffix = f"_{img_size}"
    suffix += f"_{split}" if split is not None else ""
    if sample_fraction is not None:
        suffix += f"_s{sample_fraction}_seed{sample_seed}"
    return suffix


def _sample_filenames(
    filenames: list[str],
    sample_fraction: float | None,
    sample_seed: int,
) -> list[str]:
    """Optionally subsample a file list for TDigest estimation.

    Args:
        filenames: Full list of filenames.
        sample_fraction: Fraction to keep, or ``None`` for all.
        sample_seed: RNG seed for reproducibility.

    Returns:
        (Possibly subsampled) list of filenames in original order.
    """
    if sample_fraction is None or sample_fraction == 0:
        return filenames
    rng = np.random.default_rng(sample_seed)
    n_sample = max(1, int(len(filenames) * sample_fraction))
    indices = rng.choice(len(filenames), size=n_sample, replace=False)
    return [filenames[i] for i in sorted(indices)]


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


class Downsample:
    """Flux-conserving downsampling for square images,
    assuming input size / target size is an integer.

    Args:
        target_size: Target side length in pixels.
    """

    def __init__(self, target_size: int = 256):
        self.target_size = target_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply downsampling.

        Args:
            img: ``(C, N, N)`` array.

        Returns:
            Array of shape ``(C, M, M)``.
        """
        c, h, w = img.shape
        if h == self.target_size:
            return img

        if h != w:
            raise ValueError(f"Expected square image, got shape {img.shape}")

        if h % self.target_size != 0:
            raise ValueError(
                f"Input size {h} is not divisible by target_size {self.target_size}"
            )

        factor = h // self.target_size
        out = img.reshape(c, self.target_size, factor, self.target_size, factor).sum(
            axis=(2, 4)
        )

        return out


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

    Computes ``arcsinh(img / scale)``. The scale can be supplied directly
    or derived from a percentile of the pixel-value distribution, estimated
    via a TDigest built over the dataset. When ``split`` is set, only files
    whose ``split`` column matches are used to build the TDigest, so
    statistics are always computed from training data only.

    Args:
        scale: Softening parameter. Mutually exclusive with ``percentile``
            + ``data_dir``.
        transforms: Pre-processing pipeline applied to each image before
            accumulating into the TDigest (e.g. flux conversion and padding).
            Defaults to identity.
        percentile: Percentile of non-zero pixel values used to derive
            ``scale``. Requires ``data_dir``.
        data_dir: Directory containing ``metadata.csv`` and ``.npy`` files.
            Required when ``percentile`` is set.
        split: Split name used to filter ``metadata.csv`` when building the
            TDigest. Defaults to ``"train"``. Pass ``None`` to use all rows.
        sample_fraction: Fraction of the filtered file list to use when
            building the TDigest. ``None`` uses all files. Defaults to
            ``None``.
        sample_seed: RNG seed for reproducible sampling. Only used when
            ``sample_fraction`` is set. Defaults to ``42``.
        n_workers: Number of multiprocessing workers for TDigest
            computation. ``0`` means serial. Defaults to ``0``.
        cache_dir: Directory for writing/reading tdigest cache files.
            Falls back to ``data_dir`` when ``None``. Defaults to ``None``.
    """

    def __init__(
        self,
        scale: float | None = 1,
        transforms=None,
        percentile=None,
        data_dir: str = None,
        split: str | None = "train",
        sample_fraction: float | None = None,
        sample_seed: int = 42,
        n_workers: int = 0,
        cache_dir: str | None = None,
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
            transforms = _identity
        self.transforms = transforms
        self.percentile = percentile
        self.data_dir = data_dir
        self.split = split
        self.sample_fraction = sample_fraction
        self.sample_seed = sample_seed
        self.n_workers = n_workers
        self.cache_dir = cache_dir if cache_dir is not None else data_dir

        if use_scale:
            self.scale = scale

        if use_percentile:

            csv_path = os.path.join(data_dir, "metadata.csv")
            metadata = pd.read_csv(csv_path)
            filename = metadata["filename"].iloc[0]
            img_size = _process_single_file(
                data_dir=data_dir, filename=filename, transforms=transforms
            ).shape[-1]

            suffix = _tdigest_cache_suffix(
                img_size, split, sample_fraction, sample_seed
            )
            tdigest_path = os.path.join(self.cache_dir, f"arcsinh_tdigest{suffix}.json")

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

    def _build_tdigest(self) -> TDigest:
        """Build a TDigest over non-zero pixel values in the dataset.

        Reads filenames from ``metadata.csv`` (filtered to ``self.split`` if
        set), applies ``self.transforms`` to each image, and accumulates all
        positive pixel values into the digest.

        Returns:
            Fitted ``TDigest`` instance.
        """
        csv_path = os.path.join(self.data_dir, "metadata.csv")
        metadata = pd.read_csv(csv_path)
        if self.split is not None:
            metadata = metadata[metadata["split"] == self.split]
        filenames = metadata["filename"].tolist()
        filenames = _sample_filenames(filenames, self.sample_fraction, self.sample_seed)

        return build_tdigest(
            data_dir=self.data_dir,
            filenames=filenames,
            transforms=self.transforms,
            pixel_filter=_filter_positive,
            n_workers=self.n_workers,
        )

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

    This preserves relative intensity across the dataset while mapping
    outputs to a stable range (e.g., [-1, 1]) for generative training.
    The scale factors are global constants, so the operation is invertible.

    Global bounds can be supplied directly or estimated from the dataset via
    a TDigest. When ``split`` is set, only files whose ``split`` column
    matches are used to build the TDigest, so statistics come from training
    data only.

    Args:
        global_min: Lower bound of the input range. If ``None``, derived
            from ``digest.min()`` over the dataset.
        global_max: Upper bound of the input range. If ``None``, derived
            from ``digest.max()`` over the dataset.
        norm_min: Lower bound of the output range. Defaults to ``-1.0``.
        norm_max: Upper bound of the output range. Defaults to ``1.0``.
        transforms: Pipeline applied to each image before accumulating into
            the TDigest. Defaults to identity.
        percentile: Controls TDigest cache filename. Required when either
            bound is ``None``.
        data_dir: Directory containing ``metadata.csv`` and ``.npy`` files.
            Required when either bound is ``None``.
        split: Split name used to filter ``metadata.csv`` when building the
            TDigest. Defaults to ``"train"``. Pass ``None`` to use all rows.
        sample_fraction: Fraction of the filtered file list to use when
            building the TDigest. ``None`` uses all files. Defaults to
            ``None``.
        sample_seed: RNG seed for reproducible sampling. Only used when
            ``sample_fraction`` is set. Defaults to ``42``.
        n_workers: Number of multiprocessing workers for TDigest
            computation. ``0`` means serial. Defaults to ``0``.
        cache_dir: Directory for writing/reading tdigest cache files.
            Falls back to ``data_dir`` when ``None``. Defaults to ``None``.
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
        split: str | None = "train",
        sample_fraction: float | None = None,
        sample_seed: int = 42,
        n_workers: int = 0,
        cache_dir: str | None = None,
    ):

        if transforms is None:
            transforms = _identity
        self.transforms = transforms
        self.data_dir = data_dir
        self.split = split
        self.sample_fraction = sample_fraction
        self.sample_seed = sample_seed
        self.n_workers = n_workers
        self.cache_dir = cache_dir if cache_dir is not None else data_dir

        self.norm_min = norm_min
        self.norm_max = norm_max

        global_value_not_set = (global_min is None) or (global_max is None)

        if global_value_not_set:

            csv_path = os.path.join(data_dir, "metadata.csv")
            metadata = pd.read_csv(csv_path)
            filename = metadata["filename"].iloc[0]
            img_size = _process_single_file(
                data_dir=data_dir, filename=filename, transforms=transforms
            ).shape[-1]

            suffix = _tdigest_cache_suffix(
                img_size, split, sample_fraction, sample_seed
            )
            tdigest_path = os.path.join(
                self.cache_dir,
                f"global_norm_tdigest_{int(percentile)}{suffix}.json",
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

    def _build_tdigest(self) -> TDigest:
        """Build a TDigest over all pixel values in the dataset.

        Reads filenames from ``metadata.csv`` (filtered to ``self.split`` if
        set), applies ``self.transforms`` to each image, and accumulates all
        pixel values into the digest to estimate global min/max.

        Returns:
            Fitted ``TDigest`` instance.
        """
        csv_path = os.path.join(self.data_dir, "metadata.csv")
        metadata = pd.read_csv(csv_path)
        if self.split is not None:
            metadata = metadata[metadata["split"] == self.split]
        filenames = metadata["filename"].tolist()
        filenames = _sample_filenames(filenames, self.sample_fraction, self.sample_seed)

        return build_tdigest(
            data_dir=self.data_dir,
            filenames=filenames,
            transforms=self.transforms,
            pixel_filter=_flatten,
            n_workers=self.n_workers,
        )

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


class RandomHorizontalFlip(WorkerSeededTransform):
    """Randomly flip image horizontally."""

    def __init__(self, p: float = 0.5, seed: int | None = None):
        super().__init__(seed=seed)
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self._get_rng().random() < self.p:
            return np.flip(img, axis=-1)
        return img


class RandomVerticalFlip(WorkerSeededTransform):
    """Randomly flip image vertically."""

    def __init__(self, p: float = 0.5, seed: int | None = None):
        super().__init__(seed=seed)
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self._get_rng().random() < self.p:
            return np.flip(img, axis=-2)
        return img


class RandomRotation90(WorkerSeededTransform):
    """Randomly apply 0, 1, 2, or 3 quarter-turns."""

    def __init__(self, seed: int | None = None):
        super().__init__(seed=seed)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        k = self._get_rng().integers(4)
        return np.rot90(img, k=k, axes=(-2, -1))
