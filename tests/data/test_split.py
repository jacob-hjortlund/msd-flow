"""Tests for msdflow.data.split."""

import numpy as np
import pandas as pd
import pytest

from msdflow.data.split import assign_splits


@pytest.fixture
def metadata_dir(tmp_path):
    """Create a directory with a metadata.csv (100 rows, no split column)."""
    records = [
        {"filename": f"galaxy_{i:05d}.npy", "fits_name": f"snap_{i}"}
        for i in range(100)
    ]
    pd.DataFrame(records).to_csv(tmp_path / "metadata.csv", index=False)
    return str(tmp_path)


class TestAssignSplits:
    """Tests for assign_splits function."""

    def test_adds_split_column(self, metadata_dir):
        """Verify split column is added to metadata.csv."""
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df = pd.read_csv(f"{metadata_dir}/metadata.csv")
        assert "split" in df.columns

    def test_correct_proportions(self, metadata_dir):
        """Verify split proportions match requested ratios."""
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df = pd.read_csv(f"{metadata_dir}/metadata.csv")
        counts = df["split"].value_counts()
        assert counts["train"] == 90
        assert counts["val"] == 5
        assert counts["test"] == 5

    def test_all_rows_assigned(self, metadata_dir):
        """Verify every row gets a split assignment."""
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df = pd.read_csv(f"{metadata_dir}/metadata.csv")
        assert df["split"].notna().all()
        assert len(df) == 100

    def test_reproducible_with_seed(self, metadata_dir):
        """Verify same seed produces same split."""
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df1 = pd.read_csv(f"{metadata_dir}/metadata.csv")

        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df2 = pd.read_csv(f"{metadata_dir}/metadata.csv")

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_produces_different_split(self, metadata_dir):
        """Verify different seeds produce different splits."""
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df1 = pd.read_csv(f"{metadata_dir}/metadata.csv")
        splits1 = df1["split"].tolist()

        assign_splits(metadata_dir, seed=99, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df2 = pd.read_csv(f"{metadata_dir}/metadata.csv")
        splits2 = df2["split"].tolist()

        assert splits1 != splits2

    def test_overwrites_existing_split_column(self, metadata_dir):
        """Verify re-running overwrites existing split column safely."""
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.8, "val": 0.1, "test": 0.1})
        df = pd.read_csv(f"{metadata_dir}/metadata.csv")
        counts = df["split"].value_counts()
        assert counts["train"] == 80
        assert counts["val"] == 10
        assert counts["test"] == 10

    def test_preserves_other_columns(self, metadata_dir):
        """Verify non-split columns are preserved unchanged."""
        df_before = pd.read_csv(f"{metadata_dir}/metadata.csv")
        assign_splits(metadata_dir, seed=42, ratios={"train": 0.9, "val": 0.05, "test": 0.05})
        df_after = pd.read_csv(f"{metadata_dir}/metadata.csv")
        pd.testing.assert_frame_equal(
            df_before[["filename", "fits_name"]],
            df_after[["filename", "fits_name"]],
        )

    def test_ratios_must_sum_to_one(self, metadata_dir):
        """Verify ValueError if ratios don't sum to 1."""
        with pytest.raises(ValueError, match="sum to 1"):
            assign_splits(metadata_dir, seed=42, ratios={"train": 0.5, "val": 0.1, "test": 0.1})
