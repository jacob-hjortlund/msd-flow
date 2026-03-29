"""ClearML experiment tracking helpers.

All public functions are no-ops when ``task`` is ``None``, allowing the
rest of the pipeline to call them unconditionally.
"""

import os
import json
import hashlib
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from clearml import Task, Dataset
except ImportError:
    Task = None  # type: ignore
    Dataset = None  # type: ignore


def setup_task(clearml_cfg) -> Any:
    """Initialise a ClearML Task, or return None if disabled.

    Falls back to offline mode if the server is unreachable.

    Args:
        clearml_cfg: Hydra config node with fields ``enabled``,
            ``project_name``, ``task_name``, and ``offline_dir``.

    Returns:
        An initialised ClearML Task, or None if ``clearml_cfg.enabled``
        is False.
    """
    if not clearml_cfg.enabled:
        return None

    try:
        return Task.init(
            project_name=clearml_cfg.project_name,
            task_name=clearml_cfg.task_name,
        )
    except Exception as exc:
        logger.warning(
            "ClearML server unreachable (%s). Falling back to offline mode.", exc
        )
        os.environ["CLEARML_OFFLINE_MODE"] = "1"
        os.makedirs(clearml_cfg.offline_dir, exist_ok=True)
        Task.set_offline(offline_mode=True)
        return Task.init(
            project_name=clearml_cfg.project_name,
            task_name=clearml_cfg.task_name,
        )


def _compute_dataset_hash(
    bands: list,
    version_ids: list,
    snapshots: list,
    num_files_per_view: int,
) -> str:
    """Compute a deterministic SHA-256 hash tag for a dataset configuration.

    Args:
        bands: List of band name strings.
        version_ids: List of version integers.
        snapshots: List of snapshot integers.
        num_files_per_view: Max files per view combination.

    Returns:
        16-character hex string.
    """
    config_data = {
        "bands": sorted(bands),
        "version_ids": sorted(version_ids),
        "snapshots": sorted(snapshots),
        "num_files_per_view": num_files_per_view,
    }
    config_str = json.dumps(config_data, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def register_or_get_dataset(
    task: Any,
    processed_dir: str,
    bands: list,
    version_ids: list,
    snapshots: list,
    num_files_per_view: int,
) -> str | None:
    """Register a new ClearML Dataset or retrieve an existing one.

    Builds a deterministic config hash to identify the dataset version.
    If a dataset with that hash already exists, returns its ID without
    re-uploading. Otherwise creates, populates, and finalizes a new one.

    Args:
        task: Active ClearML Task, or None (no-op).
        processed_dir: Local directory containing the processed .npy files.
        bands: Band names used for this dataset.
        version_ids: TNG version integers.
        snapshots: TNG snapshot integers.
        num_files_per_view: Files-per-view limit used during download.

    Returns:
        ClearML dataset ID string, or None if task is None.
    """
    if task is None:
        return None

    config_hash = _compute_dataset_hash(bands, version_ids, snapshots, num_files_per_view)
    dataset_name = "TNG50"
    dataset_project = task.get_project_name()

    try:
        dataset = Dataset.get(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            dataset_tags=[config_hash],
        )
        logger.info("Found existing ClearML dataset: %s", dataset.id)
        return dataset.id
    except ValueError:
        logger.info("Creating new ClearML dataset with tag %s", config_hash)
        dataset = Dataset.create(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            dataset_tags=[config_hash],
        )
        dataset.add_files(processed_dir)
        dataset.finalize()
        task.get_logger().report_text(f"Registered new dataset: {dataset.id}")
        logger.info("Registered new dataset: %s", dataset.id)
        return dataset.id


def log_metrics(task: Any, scalars: dict, epoch: int) -> None:
    """Log scalar metrics to ClearML.

    Args:
        task: Active ClearML Task, or None (no-op).
        scalars: Dict mapping metric name to float value.
        epoch: Current epoch number (used as iteration index).
    """
    if task is None:
        return
    cl_logger = task.get_logger()
    for key, value in scalars.items():
        cl_logger.report_scalar(title=key, series=key, value=value, iteration=epoch)


def log_checkpoint(task: Any, path: str, epoch: int) -> None:
    """Upload a checkpoint file as a ClearML artifact.

    Args:
        task: Active ClearML Task, or None (no-op).
        path: Local path to the checkpoint file.
        epoch: Current epoch number (used in artifact name).
    """
    if task is None:
        return
    task.upload_artifact(name=f"checkpoint_epoch_{epoch}", artifact_object=path)


def log_samples(task: Any, images: np.ndarray, epoch: int) -> None:
    """Upload generated sample images to ClearML.

    Args:
        task: Active ClearML Task, or None (no-op).
        images: Array of shape ``(N, C, H, W)``.
        epoch: Current epoch number.
    """
    if task is None:
        return
    cl_logger = task.get_logger()
    for img in images:
        cl_logger.report_image(
            title="samples",
            series=f"epoch_{epoch}",
            iteration=epoch,
            image=img,
        )
