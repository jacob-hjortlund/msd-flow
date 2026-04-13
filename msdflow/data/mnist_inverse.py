"""MNIST dataset with Gaussian blur applied as a forward (degradation) model.

Returns ``(blurred_image, clean_image)`` pairs for use in inverse-problem
experiments such as MCMC posterior sampling.
"""

import numpy as np
import torch
import torchvision.datasets
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset


class BlurredMNIST(Dataset):
    """MNIST test set with deterministic Gaussian blur + optional noise.

    Each sample returns a blurred (and optionally noisy) observation alongside
    the corresponding ground-truth clean image.  Both tensors are ``float32``
    normalised to ``[-1, 1]`` with shape ``(1, 28, 28)``.

    Args:
        root:        Directory where MNIST data is stored / downloaded.
        train:       If ``True``, use the 60 k training split; otherwise use
                     the 10 k test split.
        sigma_blur:  Standard deviation of the Gaussian blur kernel (pixels).
                     Set to ``0`` to skip blurring.
        sigma_noise: Standard deviation of additive Gaussian noise applied
                     after blurring, in the ``[-1, 1]`` normalised space.
                     Set to ``0`` for a noiseless observation.
        download:    If ``True``, download MNIST if not already present.
        seed:        RNG seed used for reproducible noise draws.
    """

    def __init__(
        self,
        root: str,
        train: bool = False,
        sigma_blur: float = 2.0,
        sigma_noise: float = 0.05,
        download: bool = True,
        seed: int = 42,
    ):
        base = torchvision.datasets.MNIST(root=root, train=train, download=download)
        # Normalise to [-1, 1] and store as float32 numpy for scipy processing
        self._clean = base.data.float().unsqueeze(1) / 127.5 - 1.0  # (N, 1, 28, 28)
        self._sigma_blur = sigma_blur
        self._sigma_noise = sigma_noise
        self._rng = np.random.default_rng(seed)

        # Pre-compute all blurred images so __getitem__ is deterministic
        clean_np = self._clean.numpy()  # (N, 1, 28, 28)
        if sigma_blur > 0:
            # gaussian_filter operates per-channel; sigma=(0, 0, sy, sx)
            blurred = gaussian_filter(clean_np, sigma=(0, 0, sigma_blur, sigma_blur))
        else:
            blurred = clean_np.copy()

        if sigma_noise > 0:
            noise = self._rng.normal(0.0, sigma_noise, blurred.shape).astype(np.float32)
            blurred = blurred + noise

        self._blurred = torch.from_numpy(blurred.astype(np.float32))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._clean)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a blurred observation and the corresponding clean image.

        Args:
            idx: Sample index.

        Returns:
            Tuple ``(y, x)`` where ``y`` is the blurred (+ noisy) image and
            ``x`` is the clean ground-truth image, both of shape
            ``(1, 28, 28)`` in ``float32`` normalised to ``[-1, 1]``.
        """
        return self._blurred[idx], self._clean[idx]
