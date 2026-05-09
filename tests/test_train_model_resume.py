"""Tests for train_model resume orchestration."""

from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf


def _cfg(tmp_path):
    """Return a minimal train_model config for resume orchestration tests."""
    return OmegaConf.create(
        {
            "seed": 42,
            "clearml": {
                "enabled": True,
                "use_dataset": False,
                "project_name": "msd-flow",
                "task_name": "train",
                "offline_dir": str(tmp_path / "offline"),
            },
            "data": {
                "dataset": {
                    "dataset_name": "TNG50",
                    "data_dir": str(tmp_path / "data"),
                    "seed": 42,
                    "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                },
                "download": {
                    "version_ids": [50],
                    "snapshots": [99],
                    "bands": ["r"],
                    "num_files_per_view": 1,
                },
                "dataloader": {
                    "cache_dir": None,
                    "data_dir": None,
                    "train": "train-loader-cfg",
                    "val": "val-loader-cfg",
                    "test": "test-loader-cfg",
                },
            },
            "model": {"_target_": "unused"},
            "train": {
                "_target_": "unused",
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "resume": {
                    "restart": False,
                    "auto": True,
                    "hash": None,
                    "hash_exclude": ["clearml", "train.resume", "train.num_epochs"],
                    "latest_filename": "latest.json",
                    "save_on_sigterm": True,
                },
                "num_epochs": 10,
            },
        }
    )


def test_main_discovers_checkpoint_before_clearml_setup(tmp_path):
    """Checkpoint metadata should provide ClearML task id before setup_task."""
    cfg = _cfg(tmp_path)
    calls = []
    checkpoint_metadata = {
        "metadata_path": str(tmp_path / "checkpoint.json"),
        "payload_path": str(tmp_path / "checkpoint.eqx"),
        "clearml_task_id": "task-1",
        "stable_hash": "hash123",
        "ema_initialized": True,
    }

    def fake_setup_task(clearml_cfg, resume_task_id=None):
        calls.append(("setup_task", resume_task_id))
        task = MagicMock()
        task.id = "task-1"
        return task

    def fake_resolve_dataset(**kwargs):
        calls.append(("resolve_dataset", kwargs["task"].id))
        return str(tmp_path / "resolved-data")

    def fake_call(train_cfg):
        def runner(**kwargs):
            calls.append(
                (
                    "train",
                    train_cfg.checkpoint_dir,
                    train_cfg.checkpoint_hash,
                    kwargs["resume_checkpoint_path"],
                    kwargs["resume_metadata"],
                    kwargs["source_checkpoint_path"],
                )
            )
            return kwargs["model"]

        return runner

    def fake_compute_config_hash(cfg, exclude_paths):
        calls.append(("compute_config_hash", tuple(exclude_paths)))
        return "hash123", {"model": {}}

    def fake_checkpoint_run_dir(root, stable_hash):
        calls.append(("checkpoint_run_dir", root, stable_hash))
        return str(tmp_path / "checkpoints" / "hash123")

    def fake_discover_latest_checkpoint(run_dir, latest_filename, restart):
        calls.append(("discover_latest_checkpoint", run_dir, latest_filename, restart))
        return checkpoint_metadata

    with patch("train_model.compute_config_hash", side_effect=fake_compute_config_hash), patch(
        "train_model.checkpoint_run_dir",
        side_effect=fake_checkpoint_run_dir,
    ), patch(
        "train_model.discover_latest_checkpoint",
        side_effect=fake_discover_latest_checkpoint,
    ), patch(
        "train_model.setup_task",
        side_effect=fake_setup_task,
    ), patch(
        "train_model.resolve_dataset",
        side_effect=fake_resolve_dataset,
    ), patch(
        "train_model.build_dataloader",
        return_value=[],
    ), patch(
        "train_model.instantiate",
        return_value=lambda key: MagicMock(name="model"),
    ), patch(
        "train_model.call",
        side_effect=fake_call,
    ), patch(
        "train_model.seed_everything",
        return_value=__import__("jax").random.PRNGKey(0),
    ):
        import train_model

        train_model.main.__wrapped__(cfg)

    assert calls[0][0] == "compute_config_hash"
    assert calls[1][0] == "checkpoint_run_dir"
    assert calls[2][0] == "discover_latest_checkpoint"
    assert calls[3] == ("setup_task", "task-1")
    assert calls[4] == ("resolve_dataset", "task-1")
    assert calls[-1][0] == "train"
    assert calls[-1][1].endswith("checkpoints/hash123")
    assert calls[-1][2] == "hash123"
    assert calls[-1][3] == str(tmp_path / "checkpoint.eqx")
    assert calls[-1][4] is checkpoint_metadata
    assert calls[-1][5] == str(tmp_path / "checkpoint.eqx")


def test_main_starts_fresh_clearml_task_when_resume_metadata_has_no_task_id(tmp_path):
    """Missing checkpoint task id should not block resume startup."""
    cfg = _cfg(tmp_path)
    calls = []
    checkpoint_metadata = {
        "metadata_path": str(tmp_path / "checkpoint.json"),
        "payload_path": str(tmp_path / "checkpoint.eqx"),
        "stable_hash": "hash123",
        "ema_initialized": True,
    }

    def fake_setup_task(clearml_cfg, resume_task_id=None):
        calls.append(("setup_task", resume_task_id))
        task = MagicMock()
        task.id = "new-task"
        return task

    def fake_call(train_cfg):
        def runner(**kwargs):
            calls.append(("train", kwargs["resume_metadata"]))
            return kwargs["model"]

        return runner

    with patch("train_model.compute_config_hash", return_value=("hash123", {"model": {}})), patch(
        "train_model.checkpoint_run_dir",
        return_value=str(tmp_path / "checkpoints" / "hash123"),
    ), patch(
        "train_model.discover_latest_checkpoint",
        return_value=checkpoint_metadata,
    ), patch(
        "train_model.setup_task",
        side_effect=fake_setup_task,
    ), patch(
        "train_model.resolve_dataset",
        return_value=str(tmp_path / "resolved-data"),
    ), patch(
        "train_model.build_dataloader",
        return_value=[],
    ), patch(
        "train_model.instantiate",
        return_value=lambda key: MagicMock(name="model"),
    ), patch(
        "train_model.call",
        side_effect=fake_call,
    ), patch(
        "train_model.seed_everything",
        return_value=__import__("jax").random.PRNGKey(0),
    ):
        import train_model

        train_model.main.__wrapped__(cfg)

    assert ("setup_task", None) in calls
    assert ("train", checkpoint_metadata) in calls


def test_main_passes_hash_payload_outside_hydra_train_config(tmp_path):
    """Hash payload should not be recursively instantiated by Hydra."""
    cfg = _cfg(tmp_path)
    hash_payload = {
        "data": {
            "dataloader": {
                "data_dir": None,
                "test": {
                    "dataset": {
                        "image_transform": {
                            "_target_": "msdflow.data.preprocess.ArcsinhStretch",
                            "scale": None,
                            "percentile": 50,
                            "data_dir": None,
                        }
                    }
                },
            }
        }
    }
    train_calls = []

    def fake_call(train_cfg):
        assert "hash_payload" not in train_cfg

        def runner(**kwargs):
            train_calls.append(kwargs)
            return kwargs["model"]

        return runner

    with patch("train_model.compute_config_hash", return_value=("hash123", hash_payload)), patch(
        "train_model.checkpoint_run_dir",
        return_value=str(tmp_path / "checkpoints" / "hash123"),
    ), patch(
        "train_model.discover_latest_checkpoint",
        return_value=None,
    ), patch(
        "train_model.setup_task",
        return_value=None,
    ), patch(
        "train_model.resolve_dataset",
        return_value=str(tmp_path / "resolved-data"),
    ), patch(
        "train_model.build_dataloader",
        return_value=[],
    ), patch(
        "train_model.instantiate",
        return_value=lambda key: MagicMock(name="model"),
    ), patch(
        "train_model.call",
        side_effect=fake_call,
    ), patch(
        "train_model.seed_everything",
        return_value=__import__("jax").random.PRNGKey(0),
    ):
        import train_model

        train_model.main.__wrapped__(cfg)

    assert train_calls[0]["hash_payload"] is hash_payload
