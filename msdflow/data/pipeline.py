"""Dataset resolution coordinator.

Determines whether to download, re-split, or reuse an existing dataset.
"""

import os
import shutil
import logging
import tempfile
from typing import Any

from omegaconf import OmegaConf
from hydra.utils import call

from msdflow.data.utils import compute_download_hash, compute_full_hash
from msdflow.data.split import assign_splits
from msdflow.tracking import (
    get_dataset_id,
    get_base_dataset_id,
    register_dataset,
    create_dataset_version,
)

try:
    from clearml import Dataset
except ImportError:
    Dataset = None  # type: ignore

logger = logging.getLogger(__name__)

_SPLITS_HASH_FILE = ".splits_hash"


def resolve_dataset(
    task: Any,
    dataset_name: str,
    data_dir: str,
    seed: int,
    ratios: dict,
    download_cfg,
    skip_download: bool = False,
) -> str:
    """Resolve the local path to a processed dataset.

    Checks whether a dataset matching the current config already exists and
    acts accordingly:

    - **Case A (exact match):** Dataset with current download *and* split
      config already exists. Returns path immediately with no work done.
    - **Case B (re-split only):** Dataset with current download config exists
      but splits differ. Re-assigns splits without re-downloading.
    - **Case C (full download):** No matching data found. Downloads, extracts,
      assigns splits, and registers.

    Args:
        task: Active ClearML Task, or ``None`` for local (no-tracking) mode.
        dataset_name: ClearML dataset name (unused in local mode).
        data_dir: Base data directory. ``processed_dir = data_dir/<download_hash>``.
        seed: Random seed for split assignment.
        ratios: Dict mapping split name to fraction (must sum to 1.0).
        download_cfg: Hydra DictConfig with ``_target_`` and ``_partial_: true``.
            Must *not* contain ``processed_dir`` — it is injected at call time.
        skip_download: If ``True``, raise ``FileNotFoundError`` in Case C instead
            of downloading.

    Returns:
        Absolute local path to the resolved ``processed_dir``.
    """
    resolved = OmegaConf.to_container(download_cfg, resolve=True)
    download_hash = compute_download_hash(**resolved)
    full_hash = compute_full_hash(download_hash, seed, ratios)
    processed_dir = os.path.join(data_dir, download_hash)

    if task is None:
        return _resolve_local(
            processed_dir, full_hash, seed, ratios, download_cfg, skip_download
        )
    return _resolve_clearml(
        task,
        dataset_name,
        processed_dir,
        download_hash,
        full_hash,
        seed,
        ratios,
        download_cfg,
        skip_download,
    )


def _resolve_local(
    processed_dir: str,
    full_hash: str,
    seed: int,
    ratios: dict,
    download_cfg,
    skip_download: bool,
) -> str:
    metadata_path = os.path.join(processed_dir, "metadata.csv")
    splits_hash_path = os.path.join(processed_dir, _SPLITS_HASH_FILE)

    if os.path.exists(metadata_path):
        if os.path.exists(splits_hash_path):
            with open(splits_hash_path) as f:
                stored = f.read().strip()
            if stored == full_hash:
                logger.info("Case A: exact dataset match. Using %s", processed_dir)
                return processed_dir
        # Case B: re-split only
        logger.info("Case B: re-assigning splits in %s", processed_dir)
        assign_splits(processed_dir, seed=seed, ratios=ratios)
        with open(splits_hash_path, "w") as f:
            f.write(full_hash)
        return processed_dir

    # Case C: full download
    if skip_download:
        raise FileNotFoundError(
            f"skip_download=True but no dataset found at {processed_dir}"
        )
    logger.info("Case C: downloading dataset to %s", processed_dir)
    call(download_cfg)(processed_dir=processed_dir)
    assign_splits(processed_dir, seed=seed, ratios=ratios)
    with open(splits_hash_path, "w") as f:
        f.write(full_hash)
    return processed_dir


def _resolve_clearml(
    task: Any,
    dataset_name: str,
    processed_dir: str,
    download_hash: str,
    full_hash: str,
    seed: int,
    ratios: dict,
    download_cfg,
    skip_download: bool,
) -> str:
    # Case A: exact match
    exact_id = get_dataset_id(task, dataset_name, full_hash)
    if exact_id:
        logger.info("Case A: exact ClearML dataset match (%s)", exact_id)
        return Dataset.get(dataset_id=exact_id, alias="raw_data").get_local_copy()

    # Case B: re-split from base
    base_id = get_base_dataset_id(task, dataset_name, download_hash)
    if base_id:
        logger.info("Case B: re-splitting ClearML dataset (base: %s)", base_id)
        base_path = Dataset.get(dataset_id=base_id).get_local_copy()
        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copy(os.path.join(base_path, "metadata.csv"), tmp_dir)
            assign_splits(tmp_dir, seed=seed, ratios=ratios)
            new_id = create_dataset_version(
                task,
                dataset_name,
                base_id,
                os.path.join(tmp_dir, "metadata.csv"),
                download_hash,
                full_hash,
            )
            if not new_id:
                logger.warning(
                    "Dataset versioning failed; applying splits to base path directly"
                )
                shutil.copy(os.path.join(tmp_dir, "metadata.csv"), base_path)
                return base_path
        return Dataset.get(dataset_id=new_id, alias="raw_data").get_local_copy()

    metadata_path = os.path.join(processed_dir, "metadata.csv")
    splits_hash_path = os.path.join(processed_dir, _SPLITS_HASH_FILE)

    # Determine if we have valid, matching local data
    has_metadata = os.path.exists(metadata_path)
    has_valid_hash = False

    if has_metadata and os.path.exists(splits_hash_path):
        with open(splits_hash_path) as f:
            stored = f.read().strip()
            has_valid_hash = stored == full_hash

    # Logic Routing
    if has_metadata and has_valid_hash:
        logger.info(
            "Case C: local dataset with matching metadata found at %s but no ClearML version. "
            "Registering new ClearML dataset version without re-download.",
            processed_dir,
        )
    else:
        # Case D: We either have no data, missing hashes, or mismatched hashes. We must download.
        if skip_download:
            raise FileNotFoundError(
                f"skip_download=True but valid local dataset not found for "
                f"download_hash={download_hash}"
            )

        logger.info("Case D: downloading and registering new ClearML dataset")
        call(download_cfg)(processed_dir=processed_dir)

    # Now we are guaranteed to have the data downloaded (or we raised an error)
    assign_splits(processed_dir, seed=seed, ratios=ratios)
    new_id = register_dataset(
        task, dataset_name, processed_dir, download_hash, full_hash
    )

    if new_id:
        return Dataset.get(dataset_id=new_id, alias="raw_data").get_local_copy()

    return processed_dir

    # metadata_path = os.path.join(processed_dir, "metadata.csv")
    # splits_hash_path = os.path.join(processed_dir, _SPLITS_HASH_FILE)

    # if os.path.exists(metadata_path):
    #     if os.path.exists(splits_hash_path):
    #         with open(splits_hash_path) as f:
    #             stored = f.read().strip()
    #         if stored == full_hash:
    #             logger.info(
    #                 "Case C: local dataset with matching metadata found at %s but no ClearML version. "
    #                 "Registering new ClearML dataset version without re-download.",
    #                 processed_dir,
    #             )
    #         else:
    #             # Case D: full download
    #             if skip_download:
    #                 raise FileNotFoundError(
    #                     f"skip_download=True but no ClearML dataset found for "
    #                     f"download_hash={download_hash}"
    #                 )
    #             logger.info("Case D: downloading and registering new ClearML dataset")
    #             call(download_cfg)(processed_dir=processed_dir)
    # assign_splits(processed_dir, seed=seed, ratios=ratios)
    # new_id = register_dataset(
    #     task, dataset_name, processed_dir, download_hash, full_hash
    # )
    # if new_id:
    #     return Dataset.get(dataset_id=new_id, alias="raw_data").get_local_copy()
    # return processed_dir
