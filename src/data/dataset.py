"""PyTorch Dataset for processed TNG50 galaxy images."""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TNG50Dataset(Dataset):
    """Random-access dataset over extracted TNG50 galaxy ``.npy`` files.

    Always returns ``(image_tensor, meta_tensor)`` tuples. When
    ``metadata_columns`` is ``None``, ``meta_tensor`` is ``torch.empty(0)``.

    Args:
        processed_dir: Path to directory containing ``metadata.csv`` and
            ``.npy`` image files.
        split: If set (e.g., ``"train"``), filter to rows where the
            ``split`` column matches. If ``None``, use all rows.
        metadata_columns: List of float column names from ``metadata.csv``
            to return. If ``None``, an empty tensor placeholder is returned.
        image_transform: Optional callable applied to the NumPy image
            array before tensor conversion.
        metadata_transform: Optional callable applied to the metadata
            tensor after extraction.
    """

    def __init__(
        self,
        processed_dir: str,
        split: str | None = None,
        metadata_columns: list[str] | None = None,
        image_transform=None,
        metadata_transform=None,
    ):
        self.processed_dir = processed_dir
        self.split = split
        self.metadata_columns = metadata_columns
        self.image_transform = image_transform
        self.metadata_transform = metadata_transform
        csv_path = os.path.join(processed_dir, "metadata.csv")
        self.metadata = pd.read_csv(csv_path)
        if split is not None:
            self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.filenames = self.metadata["filename"].tolist()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and return a galaxy image with optional metadata.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of ``(image_tensor, meta_tensor)``. ``image_tensor`` has
            shape ``(C, H, W)`` with dtype ``float32``. ``meta_tensor`` has
            shape ``(N,)`` where N is the number of metadata columns, or
            ``torch.empty(0)`` if no metadata columns were specified.
        """
        path = os.path.join(self.processed_dir, self.filenames[idx])
        data = np.load(path)

        if self.image_transform is not None:
            data = self.image_transform(data)

        img_tensor = torch.from_numpy(np.ascontiguousarray(data)).float()

        if self.metadata_columns is None:
            return img_tensor, torch.empty(0)

        meta = self.metadata.iloc[idx][self.metadata_columns].values.astype(np.float32)
        if self.metadata_transform is not None:
            meta = self.metadata_transform(meta)

        meta_tensor = torch.from_numpy(meta)

        return img_tensor, meta_tensor
