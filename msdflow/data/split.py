"""Train/val/test split assignment for processed galaxy datasets.

Reads ``metadata.csv``, shuffles row indices with a fixed seed, and
writes a ``split`` column back to the CSV. Re-running overwrites the
existing split column.
"""

import os
import hydra
import logging

import numpy as np
import pandas as pd

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def assign_splits(
    processed_dir: str,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
    *args,
    **kwargs,
) -> None:
    """Assign train/val/test splits to metadata rows.

    Shuffles row indices deterministically using ``seed``, then assigns
    each row to a split based on ``ratios``. The ``split`` column is
    written (or overwritten) in ``metadata.csv``.

    Args:
        processed_dir: Path to directory containing ``metadata.csv``.
        seed: Random seed for reproducible shuffling.
        ratios: Mapping of split name to fraction (must sum to 1.0).
            Defaults to ``{"train": 0.9, "val": 0.05, "test": 0.05}``.

    Raises:
        ValueError: If ratios do not sum to 1.0 (within tolerance).
    """
    if ratios is None:
        ratios = {"train": 0.9, "val": 0.05, "test": 0.05}

    log.info("Assigning splits with ratios:")
    for name, ratio in ratios.items():
        log.info(f"  {name}: {ratio:.2%}")

    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}: {ratios}")

    csv_path = os.path.join(processed_dir, "metadata.csv")
    df = pd.read_csv(csv_path)
    n = len(df)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    splits = np.empty(n, dtype=object)
    start = 0
    split_names = list(ratios.keys())
    for i, name in enumerate(split_names):
        if i == len(split_names) - 1:
            # Last split gets all remaining rows (avoids rounding gaps)
            end = n
        else:
            end = start + round(ratios[name] * n)
        splits[indices[start:end]] = name
        start = end

    df["split"] = splits
    df.to_csv(csv_path, index=False)
    log.info("Split assignment complete.")
