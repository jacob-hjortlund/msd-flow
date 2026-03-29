# tests/data/test_pipeline.py
"""Tests for src.data.pipeline.resolve_dataset."""

import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, call as mock_call
from omegaconf import OmegaConf

from src.data.utils import compute_download_hash, compute_full_hash


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DOWNLOAD_KWARGS = dict(
    version_ids=[0],
    snapshots=[72],
    bands=["SUBARU_HSC.I"],
    num_files_per_view=50,
    _target_="src.data.download_tng.download_tng_data",
    _partial_=True,
    max_workers=5,
    batch_size=100,
    raw_dir="/data/raw",
    api_key="key",
)

_RATIOS = {"train": 0.9, "val": 0.05, "test": 0.05}


def _cfg():
    return OmegaConf.create(_DOWNLOAD_KWARGS)


def _dl_hash():
    return compute_download_hash(**_DOWNLOAD_KWARGS)


def _full_hash():
    return compute_full_hash(_dl_hash(), seed=42, ratios=_RATIOS)


def _make_metadata(directory):
    """Write a minimal metadata.csv with 10 rows into directory (str or Path)."""
    pd.DataFrame([
        {"filename": f"galaxy_{i:05d}.npy", "fits_name": f"snap_{i}"}
        for i in range(10)
    ]).to_csv(os.path.join(str(directory), "metadata.csv"), index=False)


# ---------------------------------------------------------------------------
# Local path
# ---------------------------------------------------------------------------

class TestResolveDatasetLocal:

    def test_case_a_returns_processed_dir_without_splitting(self, tmp_path):
        """Case A: metadata.csv + matching .splits_hash → return immediately, no split."""
        processed_dir = tmp_path / _dl_hash()
        processed_dir.mkdir()
        _make_metadata(processed_dir)
        (processed_dir / ".splits_hash").write_text(_full_hash())

        with patch("src.data.pipeline.assign_splits") as mock_split:
            from src.data.pipeline import resolve_dataset
            result = resolve_dataset(
                task=None,
                dataset_name="TNG50",
                data_dir=str(tmp_path),
                seed=42,
                ratios=_RATIOS,
                download_cfg=_cfg(),
            )

        mock_split.assert_not_called()
        assert result == str(processed_dir)

    def test_case_b_resplits_when_hash_mismatch(self, tmp_path):
        """Case B: metadata.csv exists but .splits_hash differs → assign_splits called."""
        processed_dir = tmp_path / _dl_hash()
        processed_dir.mkdir()
        _make_metadata(processed_dir)
        (processed_dir / ".splits_hash").write_text("stale_hash")

        from src.data.pipeline import resolve_dataset
        result = resolve_dataset(
            task=None,
            dataset_name="TNG50",
            data_dir=str(tmp_path),
            seed=42,
            ratios=_RATIOS,
            download_cfg=_cfg(),
        )

        df = pd.read_csv(os.path.join(str(processed_dir), "metadata.csv"))
        assert "split" in df.columns
        assert result == str(processed_dir)

    def test_case_b_resplits_when_no_splits_hash_file(self, tmp_path):
        """Case B: metadata.csv exists but no .splits_hash → assign_splits called."""
        processed_dir = tmp_path / _dl_hash()
        processed_dir.mkdir()
        _make_metadata(processed_dir)

        from src.data.pipeline import resolve_dataset
        result = resolve_dataset(
            task=None,
            dataset_name="TNG50",
            data_dir=str(tmp_path),
            seed=42,
            ratios=_RATIOS,
            download_cfg=_cfg(),
        )

        df = pd.read_csv(os.path.join(str(processed_dir), "metadata.csv"))
        assert "split" in df.columns

    def test_case_b_writes_updated_splits_hash(self, tmp_path):
        """Case B: .splits_hash is updated with the new full_hash after re-split."""
        processed_dir = tmp_path / _dl_hash()
        processed_dir.mkdir()
        _make_metadata(processed_dir)
        (processed_dir / ".splits_hash").write_text("stale_hash")

        from src.data.pipeline import resolve_dataset
        resolve_dataset(
            task=None,
            dataset_name="TNG50",
            data_dir=str(tmp_path),
            seed=42,
            ratios=_RATIOS,
            download_cfg=_cfg(),
        )

        stored = (processed_dir / ".splits_hash").read_text().strip()
        assert stored == _full_hash()

    def test_case_c_calls_download_when_no_metadata(self, tmp_path):
        """Case C: no metadata.csv → call(download_cfg)(processed_dir=...) is called."""
        mock_partial = MagicMock()

        def fake_download(processed_dir):
            os.makedirs(processed_dir, exist_ok=True)
            _make_metadata(processed_dir)

        mock_partial.side_effect = fake_download

        with patch("src.data.pipeline.call", return_value=mock_partial) as mock_call_fn:
            from src.data.pipeline import resolve_dataset
            resolve_dataset(
                task=None,
                dataset_name="TNG50",
                data_dir=str(tmp_path),
                seed=42,
                ratios=_RATIOS,
                download_cfg=_cfg(),
            )

        mock_call_fn.assert_called_once()
        mock_partial.assert_called_once()
        _, kwargs = mock_partial.call_args
        assert "processed_dir" in kwargs

    def test_case_c_assigns_splits_after_download(self, tmp_path):
        """Case C: splits are assigned after download."""
        mock_partial = MagicMock()

        def fake_download(processed_dir):
            os.makedirs(processed_dir, exist_ok=True)
            _make_metadata(processed_dir)

        mock_partial.side_effect = fake_download

        with patch("src.data.pipeline.call", return_value=mock_partial):
            from src.data.pipeline import resolve_dataset
            result = resolve_dataset(
                task=None,
                dataset_name="TNG50",
                data_dir=str(tmp_path),
                seed=42,
                ratios=_RATIOS,
                download_cfg=_cfg(),
            )

        df = pd.read_csv(os.path.join(result, "metadata.csv"))
        assert "split" in df.columns

    def test_case_c_raises_when_skip_download(self, tmp_path):
        """Case C: skip_download=True with no data → FileNotFoundError."""
        from src.data.pipeline import resolve_dataset
        with pytest.raises(FileNotFoundError, match="skip_download"):
            resolve_dataset(
                task=None,
                dataset_name="TNG50",
                data_dir=str(tmp_path),
                seed=42,
                ratios=_RATIOS,
                download_cfg=_cfg(),
                skip_download=True,
            )

    def test_processed_dir_is_derived_from_download_hash(self, tmp_path):
        """The returned path is data_dir/<download_hash>."""
        processed_dir = tmp_path / _dl_hash()
        processed_dir.mkdir()
        _make_metadata(processed_dir)
        (processed_dir / ".splits_hash").write_text(_full_hash())

        from src.data.pipeline import resolve_dataset
        result = resolve_dataset(
            task=None,
            dataset_name="TNG50",
            data_dir=str(tmp_path),
            seed=42,
            ratios=_RATIOS,
            download_cfg=_cfg(),
        )
        assert os.path.basename(result) == _dl_hash()
