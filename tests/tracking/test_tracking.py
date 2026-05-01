"""Tests for msdflow.tracking."""

import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(
    enabled=True, project_name="msd-flow", task_name="train", offline_dir="/tmp/offline"
):
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
    from msdflow.tracking import setup_task

    result = setup_task(_cfg(enabled=False))
    assert result is None


def test_setup_task_calls_task_init_when_enabled():
    mock_task = MagicMock()
    with patch("msdflow.tracking.Task") as MockTask:
        MockTask.init.return_value = mock_task
        from msdflow.tracking import setup_task

        result = setup_task(_cfg(enabled=True))
    MockTask.init.assert_called_once_with(project_name="msd-flow", task_name="train")
    assert result is mock_task


def test_setup_task_continues_resume_task_id():
    """setup_task should continue the checkpoint task id when provided."""
    mock_task = MagicMock()
    mock_task.id = "resume-task"
    with patch("msdflow.tracking.Task") as MockTask:
        MockTask.init.return_value = mock_task
        from msdflow.tracking import setup_task

        result = setup_task(_cfg(enabled=True), resume_task_id="resume-task")

    MockTask.init.assert_called_once_with(
        project_name="msd-flow",
        task_name="train",
        reuse_last_task_id="resume-task",
        continue_last_task=True,
    )
    assert result is mock_task


def test_setup_task_disabled_ignores_resume_task_id():
    """Disabled ClearML should still return None when resume task id exists."""
    from msdflow.tracking import setup_task

    result = setup_task(_cfg(enabled=False), resume_task_id="resume-task")

    assert result is None


def test_setup_task_falls_back_to_offline_on_connection_error(tmp_path, monkeypatch):
    monkeypatch.delenv("CLEARML_OFFLINE_MODE", raising=False)
    mock_task = MagicMock()
    call_count = {"n": 0}

    def fake_init(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionError("server unreachable")
        return mock_task

    with patch("msdflow.tracking.Task") as MockTask:
        MockTask.init.side_effect = fake_init
        from msdflow.tracking import setup_task

        result = setup_task(_cfg(enabled=True, offline_dir=str(tmp_path / "offline")))

    assert result is mock_task
    assert os.environ.get("CLEARML_OFFLINE_MODE") == "1"
    MockTask.set_offline.assert_called_once_with(offline_mode=True)


# ---------------------------------------------------------------------------
# log_metrics
# ---------------------------------------------------------------------------


def test_log_metrics_noop_when_task_is_none():
    from msdflow.tracking import log_metrics

    # Should not raise
    log_metrics(None, {"train/loss": 0.5}, epoch=1)


def test_log_metrics_calls_report_scalar_per_key():
    from msdflow.tracking import log_metrics

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
    from msdflow.tracking import log_checkpoint

    log_checkpoint(None, "/tmp/checkpoint.eqx", epoch=1)


def test_log_checkpoint_uploads_artifact():
    from msdflow.tracking import log_checkpoint

    mock_task = MagicMock()
    log_checkpoint(mock_task, "/tmp/model_epoch5_ema.eqx", epoch=5)
    mock_task.upload_artifact.assert_called_once_with(
        name="checkpoint_epoch_5", artifact_object="/tmp/model_epoch5_ema.eqx"
    )


# ---------------------------------------------------------------------------
# log_samples
# ---------------------------------------------------------------------------


def test_log_samples_noop_when_task_is_none():
    from msdflow.tracking import log_samples

    images = np.zeros((4, 1, 8, 8), dtype=np.float32)
    log_samples(None, images, epoch=1, title="samples")


def test_log_samples_calls_report_matplotlib_figure():
    """log_samples renders one matplotlib figure per call via report_matplotlib_figure."""
    from msdflow.tracking import log_samples

    mock_task = MagicMock()
    images = np.zeros((3, 1, 8, 8), dtype=np.float32)
    log_samples(mock_task, images, epoch=3, title="samples")
    logger = mock_task.get_logger.return_value
    assert logger.report_matplotlib_figure.call_count == 1
    kwargs = logger.report_matplotlib_figure.call_args.kwargs
    assert kwargs["title"] == "samples"
    assert kwargs["series"] == "grid"
    assert kwargs["iteration"] == 3
    assert kwargs["figure"] is not None
    # report_image must NOT be called - the implementation now uses figures only.
    assert logger.report_image.call_count == 0


def _time_binned_result():
    """Return a small time-binned result and matching history for tracking tests."""
    from msdflow.train.metrics import TimeBinnedLossHistory, TimeBinnedLossResult

    result = TimeBinnedLossResult.empty(num_bins=3)
    result.add_batch(
        loss_sums=np.array([1.0, 0.0, 6.0]),
        counts=np.array([1, 0, 3]),
    )
    history = TimeBinnedLossHistory(bin_edges=result.bin_edges)
    history.append(epoch=2, result=result)
    return result, history


def test_log_time_binned_loss_noop_when_task_is_none():
    """Time-binned loss logging should no-op when ClearML tracking is disabled."""
    from msdflow.tracking import log_time_binned_loss

    result, history = _time_binned_result()

    log_time_binned_loss(
        task=None,
        split="val",
        epoch=2,
        result=result,
        history=history,
    )


def test_log_time_binned_loss_reports_histogram_and_heatmap():
    """Time-binned loss logging should report one histogram and one heatmap."""
    from msdflow.tracking import log_time_binned_loss

    mock_task = MagicMock()
    result, history = _time_binned_result()

    log_time_binned_loss(
        task=mock_task,
        split="val",
        epoch=2,
        result=result,
        history=history,
    )

    logger = mock_task.get_logger.return_value
    assert logger.report_matplotlib_figure.call_count == 2
    calls = logger.report_matplotlib_figure.call_args_list
    assert calls[0].kwargs["title"] == "val/flow_matching_loss_by_t"
    assert calls[0].kwargs["series"] == "histogram"
    assert calls[0].kwargs["iteration"] == 2
    assert calls[1].kwargs["title"] == "val/flow_matching_loss_by_t"
    assert calls[1].kwargs["series"] == "heatmap"
    assert calls[1].kwargs["iteration"] == 2


def test_log_time_binned_loss_can_skip_heatmap():
    """Passing no history should only report the per-epoch histogram."""
    from msdflow.tracking import log_time_binned_loss

    mock_task = MagicMock()
    result, _ = _time_binned_result()

    log_time_binned_loss(
        task=mock_task,
        split="val",
        epoch=2,
        result=result,
        history=None,
    )

    logger = mock_task.get_logger.return_value
    assert logger.report_matplotlib_figure.call_count == 1
    assert logger.report_matplotlib_figure.call_args.kwargs["series"] == "histogram"


# ---------------------------------------------------------------------------
# get_dataset_id
# ---------------------------------------------------------------------------


def test_get_dataset_id_returns_none_when_task_is_none():
    from msdflow.tracking import get_dataset_id

    assert get_dataset_id(None, "TNG50", "abc123") is None


def test_get_dataset_id_queries_with_splits_tag():
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    mock_dataset = MagicMock()
    mock_dataset.id = "found-id"
    with patch("msdflow.tracking.Dataset") as MockDataset:
        MockDataset.get.return_value = mock_dataset
        from msdflow.tracking import get_dataset_id

        result = get_dataset_id(mock_task, "TNG50", "abc123")
    assert result == "found-id"
    MockDataset.get.assert_called_once_with(
        dataset_name="TNG50",
        dataset_project="msd-flow",
        dataset_tags=["splits:abc123"],
        alias="raw_data",
    )


def test_get_dataset_id_returns_none_when_not_found():
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    with patch("msdflow.tracking.Dataset") as MockDataset:
        MockDataset.get.side_effect = ValueError("not found")
        from msdflow.tracking import get_dataset_id

        result = get_dataset_id(mock_task, "TNG50", "abc123")
    assert result is None


# ---------------------------------------------------------------------------
# get_base_dataset_id
# ---------------------------------------------------------------------------


def test_get_base_dataset_id_returns_none_when_task_is_none():
    from msdflow.tracking import get_base_dataset_id

    assert get_base_dataset_id(None, "TNG50", "dl_hash") is None


def test_get_base_dataset_id_returns_latest_by_created():
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    with patch("msdflow.tracking.Dataset") as MockDataset:
        MockDataset.list_datasets.return_value = [
            {"id": "old-id", "created": "2026-01-01T00:00:00"},
            {"id": "new-id", "created": "2026-03-01T00:00:00"},
        ]
        from msdflow.tracking import get_base_dataset_id

        result = get_base_dataset_id(mock_task, "TNG50", "dl_hash")
    assert result == "new-id"
    MockDataset.list_datasets.assert_called_once_with(
        partial_name="TNG50",
        dataset_project="msd-flow",
        tags=["download:dl_hash"],
    )


def test_get_base_dataset_id_returns_none_when_empty():
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    with patch("msdflow.tracking.Dataset") as MockDataset:
        MockDataset.list_datasets.return_value = []
        from msdflow.tracking import get_base_dataset_id

        result = get_base_dataset_id(mock_task, "TNG50", "dl_hash")
    assert result is None


# ---------------------------------------------------------------------------
# register_dataset
# ---------------------------------------------------------------------------


def test_register_dataset_returns_none_when_task_is_none():
    from msdflow.tracking import register_dataset

    assert register_dataset(None, "TNG50", "/data", "dl_hash", "full_hash") is None


def test_register_dataset_creates_with_both_tags(tmp_path):
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    mock_dataset = MagicMock()
    mock_dataset.id = "new-id"
    with patch("msdflow.tracking.Dataset") as MockDataset:
        MockDataset.create.return_value = mock_dataset
        from msdflow.tracking import register_dataset

        result = register_dataset(
            mock_task, "TNG50", str(tmp_path), "dl_hash", "full_hash"
        )
    assert result == "new-id"
    MockDataset.create.assert_called_once_with(
        dataset_name="TNG50",
        dataset_project="msd-flow",
        dataset_tags=["download:dl_hash", "splits:full_hash"],
    )
    mock_dataset.add_files.assert_called_once_with(str(tmp_path))
    mock_dataset.finalize.assert_called_once()


def test_register_dataset_returns_none_on_exception():
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    with patch("msdflow.tracking.Dataset") as MockDataset:
        MockDataset.create.side_effect = RuntimeError("server error")
        from msdflow.tracking import register_dataset

        result = register_dataset(mock_task, "TNG50", "/data", "dl_hash", "full_hash")
    assert result is None


# ---------------------------------------------------------------------------
# create_dataset_version
# ---------------------------------------------------------------------------


def test_create_dataset_version_returns_none_when_task_is_none():
    from msdflow.tracking import create_dataset_version

    assert (
        create_dataset_version(None, "TNG50", "base-id", "/tmp/meta.csv", "dl", "full")
        is None
    )


def test_create_dataset_version_creates_child_with_parent_and_tags(tmp_path):
    import pandas as pd

    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    mock_dataset = MagicMock()
    mock_dataset.id = "child-id"

    metadata_path = str(tmp_path / "metadata.csv")
    pd.DataFrame([{"filename": "galaxy_00000.npy", "split": "train"}]).to_csv(
        metadata_path, index=False
    )

    with patch("msdflow.tracking.Dataset") as MockDataset:
        MockDataset.create.return_value = mock_dataset
        from msdflow.tracking import create_dataset_version

        result = create_dataset_version(
            mock_task, "TNG50", "base-id", metadata_path, "dl_hash", "full_hash"
        )
    assert result == "child-id"
    MockDataset.create.assert_called_once_with(
        dataset_name="TNG50",
        dataset_project="msd-flow",
        parent_datasets=["base-id"],
        dataset_tags=["download:dl_hash", "splits:full_hash"],
    )
    mock_dataset.add_files.assert_called_once_with(
        metadata_path, local_base_folder=str(tmp_path)
    )
    mock_dataset.finalize.assert_called_once()


def test_create_dataset_version_returns_none_on_exception():
    mock_task = MagicMock()
    mock_task.get_project_name.return_value = "msd-flow"
    with patch("msdflow.tracking.Dataset") as MockDataset:
        MockDataset.create.side_effect = RuntimeError("server error")
        from msdflow.tracking import create_dataset_version

        result = create_dataset_version(
            mock_task, "TNG50", "base-id", "/tmp/meta.csv", "dl", "full"
        )
    assert result is None
