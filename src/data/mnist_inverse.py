import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset


class BlurredMNIST(Dataset):
    """MNIST dataset returning paired (y, x1) where y = blur(x1) + noise.

    Both y and x1 are in [-1, 1] (normalised from [0, 1] with mean=0.5, std=0.5).
    y is the degraded observation; x1 is the clean target.

    Args:
        root:        Directory for MNIST download cache.
        train:       If True, use the training split; otherwise the test split.
        sigma_blur:  Standard deviation of the Gaussian blur kernel (pixels).
        sigma_noise: Standard deviation of additive Gaussian noise (in [-1,1] scale).
        seed:        NumPy seed used for noise draws (ensures reproducibility).
        download:    Whether to download MNIST if not already present.
    """

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        sigma_blur: float = 2.0,
        sigma_noise: float = 0.05,
        seed: int = 0,
        download: bool = True,
    ):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])
        self._mnist = torchvision.datasets.MNIST(
            root=root, train=train, download=download, transform=transform
        )
        self.sigma_blur = sigma_blur
        self.sigma_noise = sigma_noise
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._mnist)

    def __getitem__(self, idx: int):
        x1, label = self._mnist[idx]              # (1, 28, 28) tensor in [-1, 1]
        x1_np = x1.numpy()                        # (1, 28, 28) float32

        # Apply Gaussian blur per channel (only one channel for MNIST)
        blurred = gaussian_filter(x1_np[0], sigma=self.sigma_blur)

        # Add Gaussian noise
        noise = self._rng.normal(0.0, self.sigma_noise, blurred.shape).astype(np.float32)
        y_np = np.clip(blurred + noise, -1.0, 1.0).astype(np.float32)

        y = torch.from_numpy(y_np[None])          # (1, 28, 28)
        return y, x1


def make_paired_loader(
    split: str = "train",
    batch_size: int = 128,
    sigma_blur: float = 2.0,
    sigma_noise: float = 0.05,
    seed: int = 0,
    root: str = "./data",
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Return a DataLoader yielding (y, x1) pairs of shape (B, 1, 28, 28).

    Args:
        split:       "train" or "test".
        batch_size:  Batch size.
        sigma_blur:  Gaussian blur sigma (pixels).
        sigma_noise: Additive noise sigma (in [-1, 1] scale).
        seed:        RNG seed for reproducible noise draws.
        root:        MNIST download directory.
        shuffle:     Whether to shuffle the dataset.
        num_workers: DataLoader worker processes.

    Returns:
        DataLoader yielding (y_batch, x1_batch) tensors.
    """
    dataset = BlurredMNIST(
        root=root,
        train=(split == "train"),
        sigma_blur=sigma_blur,
        sigma_noise=sigma_noise,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
