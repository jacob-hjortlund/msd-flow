"""PyTorch Dataset for processed TNG50 galaxy images."""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TNG50Dataset(Dataset):
    """Random-access dataset over extracted TNG50 galaxy ``.npy`` files.

    Args:
        processed_dir: Path to directory containing ``metadata.csv`` and
            ``.npy`` image files.
        transform: Optional callable applied to each image tensor.
    """

    def __init__(self, processed_dir: str, transform=None):
        self.processed_dir = processed_dir
        self.transform = transform
        csv_path = os.path.join(processed_dir, "metadata.csv")
        self.metadata = pd.read_csv(csv_path)
        self.filenames = self.metadata["filename"].tolist()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and return a single galaxy image as a float tensor.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Image tensor of shape ``(C, H, W)`` with dtype ``float32``.
        """
        path = os.path.join(self.processed_dir, self.filenames[idx])
        data = np.load(path)
        tensor = torch.from_numpy(data).float()
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor
