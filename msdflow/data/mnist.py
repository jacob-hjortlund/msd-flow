"""PyTorch Dataset wrapper for MNIST, compatible with the TNG50Dataset interface.

Returns ``(image_tensor, empty_meta_tensor)`` tuples with images normalised
to ``[-1, 1]`` in ``float32``, matching the shape convention ``(1, 28, 28)``.
"""

import torch
import torchvision.datasets

from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    """MNIST dataset split into train / val / test partitions.

    Torchvision provides 60 000 training images and 10 000 test images.
    This class further splits the training set into:
    - ``train``: first 55 000 samples
    - ``val``:   remaining 5 000 samples

    Images are returned as ``float32`` tensors in ``[-1, 1]``.  A zero-length
    metadata tensor is always returned as a second element so that the dataset
    is drop-in compatible with the existing training loop.

    Args:
        root:     Directory where MNIST data is stored / downloaded.
        split:    One of ``"train"``, ``"val"``, or ``"test"``.
        download: If ``True``, download MNIST if not already present.
    """

    _N_TRAIN = 55_000

    def __init__(self, root: str, split: str = "train", download: bool = True):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        is_train_split = split != "test"
        base = torchvision.datasets.MNIST(root=root, train=is_train_split, download=download)

        if split == "train":
            self._data = base.data[: self._N_TRAIN]
        elif split == "val":
            self._data = base.data[self._N_TRAIN :]
        else:
            self._data = base.data

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a normalised image tensor and an empty metadata tensor.

        Args:
            idx: Sample index.

        Returns:
            Tuple of ``(image, meta)`` where ``image`` has shape ``(1, 28, 28)``
            with dtype ``float32`` in ``[-1, 1]``, and ``meta`` is
            ``torch.empty(0)``.
        """
        img = self._data[idx].float().unsqueeze(0) / 127.5 - 1.0
        return img, torch.empty(0)
