"""ClearML experiment tracking helpers.

All public functions are no-ops when ``task`` is ``None``, allowing the
rest of the pipeline to call them unconditionally.
"""

import os
import math
import logging
import matplotlib

matplotlib.use("Agg", force=True)  # safe non-interactive backend

import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt

from typing import Any
from tqdm.contrib.logging import logging_redirect_tqdm

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

    Task.set_resource_monitor_iteration_timeout(
        wait_for_first_iteration_to_start_sec=1,  # initial fallback after 3 min
        # max_wait_for_first_iteration_to_start_sec=7200  # allow reverting for up to 2 hours
    )
    try:
        return Task.init(
            project_name=clearml_cfg.project_name,
            task_name=clearml_cfg.task_name,
        )
    except Exception as exc:
        logger.warning(
            "ClearML server unreachable (%s). Falling back to offline mode.", exc
        )
        try:
            os.environ["CLEARML_OFFLINE_MODE"] = "1"
            os.makedirs(clearml_cfg.offline_dir, exist_ok=True)
            Task.set_offline(offline_mode=True)
            return Task.init(
                project_name=clearml_cfg.project_name,
                task_name=clearml_cfg.task_name,
            )
        except Exception as exc2:
            logger.warning(
                "ClearML offline init also failed (%s). Disabling tracking.", exc2
            )
            return None


def get_dataset_id(
    task: Any,
    dataset_name: str,
    full_hash: str,
) -> str | None:
    """Find a ClearML dataset tagged with ``splits:<full_hash>``.

    Args:
        task: Active ClearML Task, or None (no-op).
        dataset_name: Name of the ClearML dataset.
        full_hash: Output of :func:`msdflow.data.utils.compute_full_hash`.

    Returns:
        ClearML dataset ID string, or None if not found.
    """
    if task is None:
        return None
    try:
        dataset_project = task.get_project_name()
        dataset = Dataset.get(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            dataset_tags=[f"splits:{full_hash}"],
            alias="raw_data",
        )
        logger.info("Found existing ClearML dataset: %s", dataset.id)
        return dataset.id
    except ValueError:
        logger.info("No ClearML dataset found with splits tag %s", full_hash)
        return None
    except Exception as exc:
        logger.warning("ClearML dataset retrieval failed (%s). Skipping.", exc)
        return None


def get_base_dataset_id(
    task: Any,
    dataset_name: str,
    download_hash: str,
) -> str | None:
    """Find the most recent ClearML dataset tagged with ``download:<download_hash>``.

    Used to locate a base dataset when only split config (seed/ratios) has changed.

    Args:
        task: Active ClearML Task, or None (no-op).
        dataset_name: Name of the ClearML dataset.
        download_hash: Output of :func:`msdflow.data.utils.compute_download_hash`.

    Returns:
        ClearML dataset ID string of the latest matching dataset, or None.
    """
    if task is None:
        return None
    try:
        dataset_project = task.get_project_name()
        datasets = Dataset.list_datasets(
            partial_name=dataset_name,
            dataset_project=dataset_project,
            tags=[f"download:{download_hash}"],
        )
        if not datasets:
            logger.info(
                "No ClearML base dataset found with download tag %s", download_hash
            )
            return None
        latest = max(datasets, key=lambda d: d.get("created", ""))
        logger.info("Found base ClearML dataset: %s", latest["id"])
        return latest["id"]
    except Exception as exc:
        logger.warning("ClearML base dataset retrieval failed (%s). Skipping.", exc)
        return None


def register_dataset(
    task: Any,
    dataset_name: str,
    processed_dir: str,
    download_hash: str,
    full_hash: str,
) -> str | None:
    """Register a new ClearML dataset from a local processed directory.

    Tags the dataset with both ``download:<download_hash>`` and
    ``splits:<full_hash>`` so it can be found by either hash later.

    Args:
        task: Active ClearML Task, or None (no-op).
        dataset_name: Name for the new ClearML dataset.
        processed_dir: Local directory containing ``.npy`` files and ``metadata.csv``.
        download_hash: Output of :func:`msdflow.data.utils.compute_download_hash`.
        full_hash: Output of :func:`msdflow.data.utils.compute_full_hash`.

    Returns:
        ClearML dataset ID string, or None if registration failed.
    """
    if task is None:
        return None
    try:
        dataset_project = task.get_project_name()
        dataset = Dataset.create(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            dataset_tags=[f"download:{download_hash}", f"splits:{full_hash}"],
        )
        dataset.add_files(processed_dir)
        with logging_redirect_tqdm():
            dataset.upload()
        dataset.finalize()
        logger.info("Registered new dataset: %s", dataset.id)
        return dataset.id
    except Exception as exc:
        logger.warning("ClearML dataset registration failed (%s). Skipping.", exc)
        return None


def create_dataset_version(
    task: Any,
    dataset_name: str,
    base_id: str,
    metadata_csv_path: str,
    download_hash: str,
    full_hash: str,
) -> str | None:
    """Create a child ClearML dataset that overrides only ``metadata.csv``.

    All ``.npy`` files are inherited from ``base_id`` (no re-upload).
    Only the updated ``metadata.csv`` is added to the child.

    Args:
        task: Active ClearML Task, or None (no-op).
        dataset_name: Name for the new ClearML dataset.
        base_id: ClearML dataset ID of the parent dataset.
        metadata_csv_path: Absolute path to the updated ``metadata.csv`` file.
            The file must be in a temporary directory that serves as the
            ``local_base_folder`` so the path inside the dataset is ``metadata.csv``.
        download_hash: Output of :func:`msdflow.data.utils.compute_download_hash`.
        full_hash: Output of :func:`msdflow.data.utils.compute_full_hash`.

    Returns:
        ClearML dataset ID string of the new version, or None if creation failed.
    """
    if task is None:
        return None
    try:
        dataset_project = task.get_project_name()
        dataset = Dataset.create(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            parent_datasets=[base_id],
            dataset_tags=[f"download:{download_hash}", f"splits:{full_hash}"],
        )
        dataset.add_files(
            metadata_csv_path,
            local_base_folder=os.path.dirname(metadata_csv_path),
        )
        with logging_redirect_tqdm():
            dataset.upload()
        dataset.finalize()
        logger.info("Created dataset version: %s (parent: %s)", dataset.id, base_id)
        return dataset.id
    except Exception as exc:
        logger.warning("ClearML dataset version creation failed (%s). Skipping.", exc)
        return None


def get_dataset_path(task: Any, dataset_id: str, *args, **kwargs) -> str:
    """Get the local path to a ClearML Dataset, or return a fallback path if tracking is disabled.

    Args:
        task: Active ClearML Task, or None (no-op).
        dataset_id: ClearML dataset ID string.
        *args, **kwargs: Additional config fields to determine fallback path if task is None.
    """
    if task is None:
        flattened_kwargs = {**kwargs}
        path = (
            flattened_kwargs["processed_dir"]
            if "processed_dir" in flattened_kwargs
            else None
        )
        if path is not None:
            logger.info(
                "ClearML tracking disabled. Using local dataset path from config: %s",
                path,
            )
            return path
        else:
            raise ValueError(
                "ClearML tracking disabled and no local dataset path provided in config."
            )
    return Dataset.get(dataset_id=dataset_id, alias="raw_data").get_local_copy()


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


def prepare_images_for_plotting(
    images,
    plot_method: str = "log",
    arcsinh_percentile=10,
    eps=1e-8,
):
    """
    Prepare grayscale images before plotting.

    Parameters
    ----------
    images : np.ndarray
        Array of shape (N, H, W).
    plot_method : str
        Method to use for plotting ('uint8', 'log' 'arcsinh').
    arcsinh_percentile : float
        Percentile used as the arcsinh softening scale.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    np.ndarray
        Prepared images of shape (N, H, W).
    """
    images = np.asarray(images)

    if images.ndim != 3:
        raise ValueError(f"Expected images with shape (N, H, W), got {images.shape}")

    images = images.copy()

    # If values look like they are in [-1, 1], map to [0, 1]
    if images.min() < 0:
        images = (images + 1.0) / 2.0

    if plot_method == "uint8":
        images = np.clip(images, 0.0, 1.0)
        images = (255.0 * images).round().astype(np.uint8)

    elif plot_method == "log":
        mask = images > 0
        images[mask] = np.log10(images[mask])
        images[~mask] = np.nan  # set non-positive values to NaN for better plotting

    elif plot_method == "arcsinh":

        scale = np.percentile(images, arcsinh_percentile, axis=(1, 2))
        scale = np.maximum(scale, eps)

        images = np.arcsinh(images / scale[:, None, None])

    return images


def plot_image_grid_subplots(
    images,
    *,
    plot_method: str = "log",
    arcsinh_percentile=10,
    cmap="gray",
    figsize=None,
    share_clim=True,
    vmin=None,
    vmax=None,
    title=None,
    wspace=0.02,
    hspace=0.02,
):
    images = prepare_images_for_plotting(
        images,
        plot_method=plot_method,
        arcsinh_percentile=arcsinh_percentile,
    )

    n, h, w = images.shape
    grid_size = math.ceil(math.sqrt(n))

    if figsize is None:
        figsize = (2.0 * grid_size, 2.0 * grid_size)

    fig, axes = plt.subplots(
        grid_size,
        grid_size,
        figsize=figsize,
        squeeze=False,
    )

    if share_clim:
        if vmin is None:
            vmin = np.nanmin(images)
        if vmax is None:
            vmax = np.nanmax(images)

    for idx, ax in enumerate(axes.flat):
        ax.set_axis_off()

        if idx >= n:
            continue

        ax.imshow(
            images[idx],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

    if title is not None:
        fig.suptitle(title)

    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
        wspace=wspace,
        hspace=hspace,
    )
    fig.patch.set_visible(False)
    return fig, axes


def log_samples(
    task: Any,
    images: np.ndarray,
    epoch: int,
    title: str,
    plot_method: str = "log",
    arcsinh_percentile=10,
) -> None:
    """Upload generated sample images to ClearML.

    Args:
        task: Active ClearML Task, or None (no-op).
        images: Array of shape ``(N, C, H, W)``.
        epoch: Current epoch number.
        title: Name of samples being logged
        plot_method: Method to use for plotting ('uint8', 'log', 'arcsinh').
        arcsinh_percentile: Percentile used as the arcsinh softening scale.
    """
    if task is None:
        return

    cl_logger = task.get_logger()
    images = images.squeeze()
    fig, axes = plot_image_grid_subplots(
        images,
        plot_method=plot_method,
        share_clim=False,
        hspace=0,
        wspace=0,
        cmap=cmr.cosmic,
        arcsinh_percentile=arcsinh_percentile,
    )

    cl_logger.report_matplotlib_figure(
        title=title,
        series="grid",
        iteration=epoch,
        figure=fig,
        report_image=False,
        report_interactive=False,
    )

    plt.close(fig)
