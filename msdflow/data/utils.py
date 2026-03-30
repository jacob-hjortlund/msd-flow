"""Dataset configuration hashing utilities."""

import json
import hashlib


def compute_download_hash(
    version_ids,
    snapshots,
    bands,
    num_files_per_view,
    **kwargs,
) -> str:
    """Compute a 16-char SHA-256 hash of download-determining config fields.

    Ignores seed, ratios, max_workers, batch_size, raw_dir, api_key, and
    any other fields not listed above.

    Args:
        version_ids: TNG version integers (order-independent).
        snapshots: Snapshot integers (order-independent).
        bands: Band name strings (order-independent).
        num_files_per_view: Max FITS files per version/snapshot combination.
        **kwargs: Ignored.

    Returns:
        16-character lowercase hex string.
    """
    data = {
        "version_ids": sorted(version_ids),
        "snapshots": sorted(snapshots),
        "bands": sorted(bands),
        "num_files_per_view": num_files_per_view,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def compute_full_hash(download_hash: str, seed: int, ratios: dict) -> str:
    """Compute a 16-char SHA-256 hash of download hash + split configuration.

    Args:
        download_hash: Output of :func:`compute_download_hash`.
        seed: Random seed used for split assignment.
        ratios: Dict mapping split name to fraction (e.g. ``{"train": 0.9}``).

    Returns:
        16-character lowercase hex string.
    """
    data = {"download_hash": download_hash, "seed": seed, "ratios": ratios}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
