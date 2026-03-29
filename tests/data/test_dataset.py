"""Tests for msdflow.data.dataset."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from msdflow.data.dataset import TNG50Dataset


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a minimal processed directory with .npy files and metadata."""
    records = []
    splits = ["train", "train", "train", "val", "test"]
    for i in range(5):
        name = f"galaxy_{i:05d}.npy"
        data = np.random.default_rng(i).random((1, 64, 64)).astype(np.float32)
        np.save(tmp_path / name, data)
        records.append({
            "filename": name,
            "fits_name": f"snap_{i}",
            "band_map": "g",
            "hdr_mass": float(i) * 1.5,
            "hdr_redshift": float(i) * 0.1,
            "split": splits[i],
        })
    pd.DataFrame(records).to_csv(tmp_path / "metadata.csv", index=False)
    return str(tmp_path)


def test_dataset_length(sample_dataset):
    """Verify __len__ matches number of entries in metadata."""
    ds = TNG50Dataset(sample_dataset)
    assert len(ds) == 5


def test_dataset_returns_tuple(sample_dataset):
    """Verify __getitem__ returns (image_tensor, meta_tensor) tuple."""
    ds = TNG50Dataset(sample_dataset)
    result = ds[0]
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_dataset_no_metadata_returns_empty_tensor(sample_dataset):
    """Verify meta is empty tensor when metadata_columns is None."""
    ds = TNG50Dataset(sample_dataset)
    img, meta = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 64, 64)
    assert img.dtype == torch.float32
    assert isinstance(meta, torch.Tensor)
    assert meta.numel() == 0


def test_dataset_with_metadata_columns(sample_dataset):
    """Verify metadata columns are returned as float tensor."""
    ds = TNG50Dataset(sample_dataset, metadata_columns=["hdr_mass", "hdr_redshift"])
    img, meta = ds[0]
    assert img.shape == (1, 64, 64)
    assert meta.shape == (2,)
    assert meta.dtype == torch.float32
    # First galaxy: mass=0.0, redshift=0.0
    torch.testing.assert_close(meta, torch.tensor([0.0, 0.0]))


def test_dataset_metadata_values_correct(sample_dataset):
    """Verify metadata values match CSV for non-zero entries."""
    ds = TNG50Dataset(sample_dataset, metadata_columns=["hdr_mass", "hdr_redshift"])
    _, meta = ds[2]
    # Third galaxy: mass=3.0, redshift=0.2
    torch.testing.assert_close(meta, torch.tensor([3.0, 0.2]))


def test_dataset_image_transform_applied(sample_dataset):
    """Verify image_transform is called on the NumPy array."""
    transform = lambda x: x * 2.0
    ds = TNG50Dataset(sample_dataset, image_transform=transform)
    raw_ds = TNG50Dataset(sample_dataset)
    img, _ = ds[0]
    raw_img, _ = raw_ds[0]
    torch.testing.assert_close(img, raw_img * 2.0)


def test_dataset_metadata_transform_applied(sample_dataset):
    """Verify metadata_transform is called on the metadata tensor."""
    meta_transform = lambda x: x + 100.0
    ds = TNG50Dataset(
        sample_dataset,
        metadata_columns=["hdr_mass"],
        metadata_transform=meta_transform,
    )
    _, meta = ds[1]
    # Second galaxy: mass=1.5, after transform: 101.5
    torch.testing.assert_close(meta, torch.tensor([101.5]))


def test_dataset_metadata_accessible(sample_dataset):
    """Verify metadata DataFrame is accessible."""
    ds = TNG50Dataset(sample_dataset)
    assert isinstance(ds.metadata, pd.DataFrame)
    assert len(ds.metadata) == 5
    assert "fits_name" in ds.metadata.columns


def test_dataset_works_with_dataloader(sample_dataset):
    """Verify dataset integrates with PyTorch DataLoader."""
    ds = TNG50Dataset(sample_dataset)
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    images, meta = next(iter(loader))
    assert images.shape == (2, 1, 64, 64)
    assert meta.numel() == 0


def test_dataset_with_metadata_works_with_dataloader(sample_dataset):
    """Verify metadata columns work with DataLoader batching."""
    ds = TNG50Dataset(sample_dataset, metadata_columns=["hdr_mass", "hdr_redshift"])
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    images, meta = next(iter(loader))
    assert images.shape == (2, 1, 64, 64)
    assert meta.shape == (2, 2)


def test_dataset_split_filter_train(sample_dataset):
    """Verify split='train' filters to train rows only."""
    ds = TNG50Dataset(sample_dataset, split="train")
    assert len(ds) == 3


def test_dataset_split_filter_val(sample_dataset):
    """Verify split='val' filters to val rows only."""
    ds = TNG50Dataset(sample_dataset, split="val")
    assert len(ds) == 1


def test_dataset_split_filter_test(sample_dataset):
    """Verify split='test' filters to test rows only."""
    ds = TNG50Dataset(sample_dataset, split="test")
    assert len(ds) == 1


def test_dataset_split_none_returns_all(sample_dataset):
    """Verify split=None returns all rows (default)."""
    ds = TNG50Dataset(sample_dataset, split=None)
    assert len(ds) == 5


def test_dataset_split_correct_items(sample_dataset):
    """Verify split filtering returns correct images."""
    ds_all = TNG50Dataset(sample_dataset)
    ds_val = TNG50Dataset(sample_dataset, split="val")
    # Val is index 3 in the original dataset
    img_all, _ = ds_all[3]
    img_val, _ = ds_val[0]
    torch.testing.assert_close(img_all, img_val)
