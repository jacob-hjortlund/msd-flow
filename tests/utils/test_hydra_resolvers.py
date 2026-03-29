"""Tests for msdflow.utils.hydra_resolvers."""

import pytest
from omegaconf import OmegaConf

from msdflow.utils.hydra_resolvers import register_all_resolvers


@pytest.fixture(autouse=True)
def resolvers():
    """Ensure resolvers are registered before each test."""
    register_all_resolvers()


# --- register_all_resolvers idempotency ---


def test_register_all_resolvers_is_idempotent():
    """Calling register_all_resolvers() multiple times must not raise."""
    register_all_resolvers()
    register_all_resolvers()


def test_resolvers_are_registered():
    """Both custom resolvers must be queryable after registration."""
    assert OmegaConf.has_resolver("if_cond")
    assert OmegaConf.has_resolver("generate_snapshot_ids")


# --- if_cond resolver ---


def test_if_cond_non_none_returns_true_val():
    """if_cond returns true_val when metadata_columns is not None."""
    cfg = OmegaConf.create({"val": "${if_cond: some_column, yes, no}"})
    assert cfg.val == "yes"


def test_if_cond_none_returns_false_val():
    """if_cond returns false_val when metadata_columns is null (None)."""
    cfg = OmegaConf.create({"val": "${if_cond: null, yes, no}"})
    assert cfg.val == "no"


def test_if_cond_empty_list_returns_true_val():
    """An empty list is not None, so if_cond returns true_val."""
    cfg = OmegaConf.create({"val": "${if_cond: [], yes, no}"})
    assert cfg.val == "yes"


def test_if_cond_numeric_true_val():
    """if_cond works with numeric true/false values."""
    cfg = OmegaConf.create({"val": "${if_cond: col, 42, 0}"})
    assert cfg.val == 42


def test_if_cond_numeric_false_val():
    """if_cond returns numeric false_val when condition is null."""
    cfg = OmegaConf.create({"val": "${if_cond: null, 42, 0}"})
    assert cfg.val == 0


# --- generate_snapshot_ids resolver ---


def test_generate_snapshot_ids_correct_values():
    """generate_snapshot_ids(start, count) returns [start, start+1, ..., start+count-1]."""
    cfg = OmegaConf.create({"ids": "${generate_snapshot_ids: 99, 3}"})
    assert list(cfg.ids) == [99, 100, 101]


def test_generate_snapshot_ids_zero_count():
    """generate_snapshot_ids with count=0 returns an empty list."""
    cfg = OmegaConf.create({"ids": "${generate_snapshot_ids: 10, 0}"})
    assert list(cfg.ids) == []


def test_generate_snapshot_ids_count_one():
    """generate_snapshot_ids with count=1 returns a single-element list."""
    cfg = OmegaConf.create({"ids": "${generate_snapshot_ids: 5, 1}"})
    assert list(cfg.ids) == [5]


def test_generate_snapshot_ids_start_zero():
    """generate_snapshot_ids starting at 0 returns [0, 1, ..., count-1]."""
    cfg = OmegaConf.create({"ids": "${generate_snapshot_ids: 0, 4}"})
    assert list(cfg.ids) == [0, 1, 2, 3]
