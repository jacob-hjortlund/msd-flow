"""Tests for src.data.dataset."""

import os
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import TNG50Dataset


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a minimal processed directory with .npy files and metadata."""
    records = []
    for i in range(5):
        name = f"galaxy_{i:05d}.npy"
        data = np.random.default_rng(i).random((1, 64, 64)).astype(np.float32)
        np.save(tmp_path / name, data)
        records.append({"filename": name, "fits_name": f"snap_{i}", "band_map": "g"})
    pd.DataFrame(records).to_csv(tmp_path / "metadata.csv", index=False)
    return str(tmp_path)


def test_dataset_length(sample_dataset):
    """Verify __len__ matches number of entries in metadata."""
    ds = TNG50Dataset(sample_dataset)
    assert len(ds) == 5


def test_dataset_getitem_returns_tensor(sample_dataset):
    """Verify __getitem__ returns a float tensor with correct shape."""
    ds = TNG50Dataset(sample_dataset)
    item = ds[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (1, 64, 64)
    assert item.dtype == torch.float32


def test_dataset_transform_applied(sample_dataset):
    """Verify transform is called on the tensor."""
    transform = lambda x: x * 2
    ds = TNG50Dataset(sample_dataset, transform=transform)
    raw_ds = TNG50Dataset(sample_dataset)
    torch.testing.assert_close(ds[0], raw_ds[0] * 2)


def test_dataset_metadata_accessible(sample_dataset):
    """Verify metadata DataFrame is accessible."""
    ds = TNG50Dataset(sample_dataset)
    assert isinstance(ds.metadata, pd.DataFrame)
    assert len(ds.metadata) == 5
    assert "fits_name" in ds.metadata.columns


from torch.utils.data import DataLoader


def test_dataset_works_with_dataloader(sample_dataset):
    """Verify dataset integrates with PyTorch DataLoader."""
    ds = TNG50Dataset(sample_dataset)
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert batch.shape == (2, 1, 64, 64)
