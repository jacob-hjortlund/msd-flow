"""Tests for msdflow.data.utils hash functions."""

import pytest
from msdflow.data.utils import compute_download_hash, compute_full_hash


BASE = dict(
    version_ids=[0, 1],
    snapshots=[72, 73],
    bands=["SUBARU_HSC.I"],
    num_files_per_view=50,
)


class TestComputeDownloadHash:

    def test_is_deterministic(self):
        assert compute_download_hash(**BASE) == compute_download_hash(**BASE)

    def test_returns_16_char_hex(self):
        h = compute_download_hash(**BASE)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_order_independent(self):
        h1 = compute_download_hash(
            version_ids=[0, 1], snapshots=[72, 73],
            bands=["B", "A"], num_files_per_view=50,
        )
        h2 = compute_download_hash(
            version_ids=[1, 0], snapshots=[73, 72],
            bands=["A", "B"], num_files_per_view=50,
        )
        assert h1 == h2

    def test_differs_for_different_bands(self):
        h1 = compute_download_hash(**BASE)
        h2 = compute_download_hash(**{**BASE, "bands": ["SUBARU_HSC.R"]})
        assert h1 != h2

    def test_differs_for_different_snapshots(self):
        h1 = compute_download_hash(**BASE)
        h2 = compute_download_hash(**{**BASE, "snapshots": [99]})
        assert h1 != h2

    def test_differs_for_different_num_files(self):
        h1 = compute_download_hash(**BASE)
        h2 = compute_download_hash(**{**BASE, "num_files_per_view": 100})
        assert h1 != h2

    def test_ignores_extra_kwargs(self):
        h1 = compute_download_hash(**BASE)
        h2 = compute_download_hash(**BASE, max_workers=99, batch_size=1000, api_key="secret")
        assert h1 == h2


class TestComputeFullHash:

    RATIOS = {"train": 0.9, "val": 0.05, "test": 0.05}

    def test_is_deterministic(self):
        h1 = compute_full_hash("abc123", seed=42, ratios=self.RATIOS)
        h2 = compute_full_hash("abc123", seed=42, ratios=self.RATIOS)
        assert h1 == h2

    def test_returns_16_char_hex(self):
        h = compute_full_hash("abc123", seed=42, ratios=self.RATIOS)
        assert len(h) == 16

    def test_differs_for_different_seed(self):
        h1 = compute_full_hash("abc123", seed=42, ratios=self.RATIOS)
        h2 = compute_full_hash("abc123", seed=99, ratios=self.RATIOS)
        assert h1 != h2

    def test_differs_for_different_ratios(self):
        h1 = compute_full_hash("abc123", seed=42, ratios=self.RATIOS)
        h2 = compute_full_hash("abc123", seed=42, ratios={"train": 0.8, "val": 0.1, "test": 0.1})
        assert h1 != h2

    def test_differs_for_different_download_hash(self):
        h1 = compute_full_hash("abc123", seed=42, ratios=self.RATIOS)
        h2 = compute_full_hash("def456", seed=42, ratios=self.RATIOS)
        assert h1 != h2
