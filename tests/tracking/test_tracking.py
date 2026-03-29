"""Tests for src.tracking."""

import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(enabled=True, project_name="msd-flow", task_name="train", offline_dir="/tmp/offline"):
    cfg = MagicMock()
    cfg.enabled = enabled
    cfg.project_name = project_name
    cfg.task_name = task_name
    cfg.offline_dir = offline_dir
    return cfg


# ---------------------------------------------------------------------------
# setup_task
# ---------------------------------------------------------------------------

def test_setup_task_returns_none_when_disabled():
    from src.tracking import setup_task
    result = setup_task(_cfg(enabled=False))
    assert result is None


def test_setup_task_calls_task_init_when_enabled():
    mock_task = MagicMock()
    with patch("src.tracking.Task") as MockTask:
        MockTask.init.return_value = mock_task
        from src.tracking import setup_task
        result = setup_task(_cfg(enabled=True))
    MockTask.init.assert_called_once_with(project_name="msd-flow", task_name="train")
    assert result is mock_task


def test_setup_task_falls_back_to_offline_on_connection_error(tmp_path, monkeypatch):
    monkeypatch.delenv("CLEARML_OFFLINE_MODE", raising=False)
    mock_task = MagicMock()
    call_count = {"n": 0}

    def fake_init(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionError("server unreachable")
        return mock_task

    with patch("src.tracking.Task") as MockTask:
        MockTask.init.side_effect = fake_init
        from src.tracking import setup_task
        result = setup_task(_cfg(enabled=True, offline_dir=str(tmp_path / "offline")))

    assert result is mock_task
    assert os.environ.get("CLEARML_OFFLINE_MODE") == "1"
    MockTask.set_offline.assert_called_once_with(offline_mode=True)


# ---------------------------------------------------------------------------
# log_metrics
# ---------------------------------------------------------------------------

def test_log_metrics_noop_when_task_is_none():
    from src.tracking import log_metrics
    # Should not raise
    log_metrics(None, {"train/loss": 0.5}, epoch=1)


def test_log_metrics_calls_report_scalar_per_key():
    from src.tracking import log_metrics
    mock_task = MagicMock()
    scalars = {"train/loss": 0.5, "val/loss": 0.3}
    log_metrics(mock_task, scalars, epoch=2)
    logger = mock_task.get_logger.return_value
    assert logger.report_scalar.call_count == 2
    logger.report_scalar.assert_any_call(
        title="train/loss", series="train/loss", value=0.5, iteration=2
    )
    logger.report_scalar.assert_any_call(
        title="val/loss", series="val/loss", value=0.3, iteration=2
    )


# ---------------------------------------------------------------------------
# log_checkpoint
# ---------------------------------------------------------------------------

def test_log_checkpoint_noop_when_task_is_none():
    from src.tracking import log_checkpoint
    log_checkpoint(None, "/tmp/checkpoint.eqx", epoch=1)


def test_log_checkpoint_uploads_artifact():
    from src.tracking import log_checkpoint
    mock_task = MagicMock()
    log_checkpoint(mock_task, "/tmp/model_epoch5_ema.eqx", epoch=5)
    mock_task.upload_artifact.assert_called_once_with(
        name="checkpoint_epoch_5", artifact_object="/tmp/model_epoch5_ema.eqx"
    )


# ---------------------------------------------------------------------------
# log_samples
# ---------------------------------------------------------------------------

def test_log_samples_noop_when_task_is_none():
    from src.tracking import log_samples
    images = np.zeros((4, 1, 8, 8), dtype=np.float32)
    log_samples(None, images, epoch=1)


def test_log_samples_calls_report_image_per_sample():
    from src.tracking import log_samples
    mock_task = MagicMock()
    images = np.zeros((3, 1, 8, 8), dtype=np.float32)
    log_samples(mock_task, images, epoch=3)
    logger = mock_task.get_logger.return_value
    assert logger.report_image.call_count == 3
    calls = logger.report_image.call_args_list
    for i in range(3):
        kw = calls[i][1]  # kwargs of i-th call
        assert kw["title"] == "samples"
        assert kw["series"] == "epoch_3"
        assert kw["iteration"] == 3
        assert np.array_equal(kw["image"], np.transpose(images[i], (1, 2, 0)))


# ---------------------------------------------------------------------------
# register_or_get_dataset
# ---------------------------------------------------------------------------

def test_register_or_get_dataset_returns_none_when_task_is_none():
    from src.tracking import register_or_get_dataset
    result = register_or_get_dataset(
        task=None,
        processed_dir="/data",
        bands=["SUBARU_HSC.I"],
        version_ids=[0],
        snapshots=[72],
        num_files_per_view=50,
    )
    assert result is None


def test_register_or_get_dataset_returns_existing_dataset_id():
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    mock_dataset = MagicMock()
    mock_dataset.id = "existing-id-123"

    with patch("src.tracking.Dataset") as MockDataset:
        MockDataset.get.return_value = mock_dataset
        from src.tracking import register_or_get_dataset
        result = register_or_get_dataset(
            task=mock_task,
            processed_dir="/data",
            bands=["SUBARU_HSC.I"],
            version_ids=[0],
            snapshots=[72],
            num_files_per_view=50,
        )

    assert result == "existing-id-123"
    MockDataset.create.assert_not_called()


def test_register_or_get_dataset_creates_new_when_not_found():
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    mock_dataset = MagicMock()
    mock_dataset.id = "new-id-456"

    with patch("src.tracking.Dataset") as MockDataset:
        MockDataset.get.side_effect = ValueError("not found")
        MockDataset.create.return_value = mock_dataset
        from src.tracking import register_or_get_dataset
        result = register_or_get_dataset(
            task=mock_task,
            processed_dir="/data/processed",
            bands=["SUBARU_HSC.I"],
            version_ids=[0],
            snapshots=[72],
            num_files_per_view=50,
        )

    assert result == "new-id-456"
    MockDataset.create.assert_called_once()
    mock_dataset.add_files.assert_called_once_with("/data/processed")
    mock_dataset.finalize.assert_called_once()


def test_register_or_get_dataset_hash_is_deterministic():
    """Same inputs always produce the same ClearML dataset tag."""
    from src.tracking import _compute_dataset_hash
    h1 = _compute_dataset_hash(["SUBARU_HSC.I"], [0, 1], [72, 73], 50)
    h2 = _compute_dataset_hash(["SUBARU_HSC.I"], [0, 1], [72, 73], 50)
    assert h1 == h2


def test_register_or_get_dataset_hash_differs_for_different_inputs():
    from src.tracking import _compute_dataset_hash
    h1 = _compute_dataset_hash(["SUBARU_HSC.I"], [0, 1], [72, 73], 50)
    h2 = _compute_dataset_hash(["SUBARU_HSC.R"], [0, 1], [72, 73], 50)
    assert h1 != h2


def test_register_or_get_dataset_hash_order_independent():
    """Hash is the same regardless of input list order."""
    from src.tracking import _compute_dataset_hash
    h1 = _compute_dataset_hash(["B", "A"], [1, 0], [73, 72], 50)
    h2 = _compute_dataset_hash(["A", "B"], [0, 1], [72, 73], 50)
    assert h1 == h2
