"""Tests for resumable training checkpoint helpers."""

import json

import pytest
from omegaconf import OmegaConf

from msdflow.train.checkpointing import (
    compute_config_hash,
    checkpoint_run_dir,
    latest_pointer_path,
    discover_latest_checkpoint,
)


def test_compute_config_hash_ignores_excluded_paths():
    """Excluded config paths must not affect the stable hash."""
    cfg_a = OmegaConf.create(
        {
            "clearml": {"task_name": "job-a"},
            "train": {
                "num_epochs": 100,
                "resume": {"restart": False},
                "optimizer": {"learning_rate": 1.0e-4},
            },
            "model": {"base_channels": 64},
        }
    )
    cfg_b = OmegaConf.create(
        {
            "clearml": {"task_name": "job-b"},
            "train": {
                "num_epochs": 200,
                "resume": {"restart": True},
                "optimizer": {"learning_rate": 1.0e-4},
            },
            "model": {"base_channels": 64},
        }
    )

    hash_a, payload_a = compute_config_hash(
        cfg_a,
        exclude_paths=["clearml", "train.resume", "train.num_epochs"],
    )
    hash_b, payload_b = compute_config_hash(
        cfg_b,
        exclude_paths=["clearml", "train.resume", "train.num_epochs"],
    )

    assert hash_a == hash_b
    assert payload_a == payload_b
    assert "clearml" not in payload_a
    assert "resume" not in payload_a["train"]
    assert "num_epochs" not in payload_a["train"]


def test_compute_config_hash_changes_for_model_fields():
    """Model compatibility fields must affect the stable hash."""
    cfg_a = OmegaConf.create({"model": {"base_channels": 64}, "train": {}})
    cfg_b = OmegaConf.create({"model": {"base_channels": 128}, "train": {}})

    hash_a, _ = compute_config_hash(cfg_a, exclude_paths=[])
    hash_b, _ = compute_config_hash(cfg_b, exclude_paths=[])

    assert hash_a != hash_b


def test_compute_config_hash_accepts_length_keyword():
    """Hash length should be configurable with the planned keyword."""
    cfg = OmegaConf.create({"model": {"base_channels": 64}, "train": {}})

    stable_hash, _ = compute_config_hash(cfg, exclude_paths=[], length=8)

    assert len(stable_hash) == 8


def test_checkpoint_run_dir_joins_root_and_hash(tmp_path):
    """Hash-specific checkpoint directory should be deterministic."""
    assert checkpoint_run_dir(str(tmp_path), "abc123") == str(tmp_path / "abc123")


def test_checkpoint_run_dir_accepts_root_keyword(tmp_path):
    """Hash-specific checkpoint directory should accept the planned keyword."""
    result = checkpoint_run_dir(root=str(tmp_path), stable_hash="abc123")

    assert result == str(tmp_path / "abc123")


def test_latest_pointer_path_uses_configured_filename(tmp_path):
    """Latest pointer path should live inside the hash-specific directory."""
    run_dir = tmp_path / "hash"

    result = latest_pointer_path(str(run_dir), "latest.json")

    assert result == str(run_dir / "latest.json")


def test_discover_latest_checkpoint_returns_none_when_restart_true(tmp_path):
    """restart=true must force a fresh run even when a pointer exists."""
    run_dir = tmp_path / "hash"
    run_dir.mkdir()
    (run_dir / "latest.json").write_text(json.dumps({"metadata_path": "x.json"}))

    result = discover_latest_checkpoint(
        str(run_dir),
        latest_filename="latest.json",
        restart=True,
    )

    assert result is None


def test_discover_latest_checkpoint_returns_none_when_pointer_absent(tmp_path):
    """restart=false should return None when no latest pointer exists."""
    run_dir = tmp_path / "hash"
    run_dir.mkdir()

    result = discover_latest_checkpoint(
        str(run_dir),
        latest_filename="latest.json",
        restart=False,
    )

    assert result is None


def test_discover_latest_checkpoint_loads_pointer_metadata(tmp_path):
    """restart=false should load metadata referenced by latest.json."""
    run_dir = tmp_path / "hash"
    run_dir.mkdir()
    metadata_path = run_dir / "checkpoint_epoch0001_step0000.json"
    payload_path = run_dir / "checkpoint_epoch0001_step0000.eqx"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stable_hash": "hash",
                "checkpoint_kind": "periodic",
                "epoch": 1,
                "completed_microsteps": 0,
                "payload_path": str(payload_path),
                "clearml_task_id": "task-1",
            }
        )
    )
    (run_dir / "latest.json").write_text(
        json.dumps({"metadata_path": str(metadata_path)})
    )

    result = discover_latest_checkpoint(
        str(run_dir),
        latest_filename="latest.json",
        restart=False,
    )

    assert result is not None
    assert result["metadata_path"] == str(metadata_path)
    assert result["payload_path"] == str(payload_path)
    assert result["clearml_task_id"] == "task-1"
