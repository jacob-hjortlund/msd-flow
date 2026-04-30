"""Helpers for resumable training checkpoint discovery."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


CHECKPOINT_SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    """Convert config values to deterministic JSON-compatible containers.

    Args:
        value: Arbitrary value from a Hydra/OmegaConf configuration.

    Returns:
        A JSON-compatible representation using dictionaries, lists, scalar
        primitives, and strings for otherwise unsupported values.
    """
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value

    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _drop_path(payload: dict[str, Any], dotted_path: str) -> None:
    """Remove a dotted path from a nested mapping if it is present.

    Args:
        payload: Normalized configuration payload to mutate.
        dotted_path: Dot-separated key path such as ``"train.resume"``.
    """
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        return

    current: Any = payload
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]

    if isinstance(current, dict):
        current.pop(parts[-1], None)


def normalized_config_payload(
    cfg: Any,
    exclude_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe configuration payload with excluded paths removed.

    Args:
        cfg: Hydra/OmegaConf configuration or equivalent mapping.
        exclude_paths: Dot-separated paths to ignore for compatibility hashing.

    Returns:
        Normalized dictionary suitable for stable JSON serialization.

    Raises:
        TypeError: If the normalized configuration root is not a mapping.
    """
    payload = _json_safe(cfg)
    if not isinstance(payload, dict):
        raise TypeError("Stable checkpoint config hash requires a mapping root.")

    for path in exclude_paths or ():
        _drop_path(payload, path)

    return payload


def compute_config_hash(
    cfg: Any,
    exclude_paths: Sequence[str] | None = None,
    digest_length: int = 16,
) -> tuple[str, dict[str, Any]]:
    """Compute a stable truncated SHA-256 hash for a configuration.

    Args:
        cfg: Hydra/OmegaConf configuration or equivalent mapping.
        exclude_paths: Dot-separated paths to remove before hashing.
        digest_length: Number of hex characters to keep from the SHA-256
            digest.

    Returns:
        Tuple of ``(stable_hash, normalized_payload)``.
    """
    payload = normalized_config_payload(cfg, exclude_paths=exclude_paths)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    stable_hash = hashlib.sha256(encoded).hexdigest()[:digest_length]
    return stable_hash, payload


def checkpoint_run_dir(checkpoint_root: str, stable_hash: str) -> str:
    """Return the hash-specific checkpoint directory path.

    Args:
        checkpoint_root: Root directory for resumable checkpoints.
        stable_hash: Stable configuration hash for the run family.

    Returns:
        Path to the directory containing checkpoints for ``stable_hash``.
    """
    return str(Path(checkpoint_root) / stable_hash)


def latest_pointer_path(run_dir: str, latest_filename: str) -> str:
    """Return the latest-checkpoint pointer path for a run directory.

    Args:
        run_dir: Hash-specific checkpoint directory.
        latest_filename: Filename used for the latest checkpoint pointer.

    Returns:
        Path to the latest checkpoint pointer JSON file.
    """
    return str(Path(run_dir) / latest_filename)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Args:
        path: Path to a JSON file.

    Returns:
        Parsed JSON object.

    Raises:
        TypeError: If the JSON root is not an object.
    """
    with Path(path).open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return payload


def discover_latest_checkpoint(
    run_dir: str,
    latest_filename: str,
    restart: bool,
) -> dict[str, Any] | None:
    """Discover the latest checkpoint metadata for a resumable run.

    Args:
        run_dir: Hash-specific checkpoint directory to inspect.
        latest_filename: Filename of the JSON pointer to the latest metadata.
        restart: Whether to force a fresh run and ignore existing pointers.

    Returns:
        Metadata dictionary with ``metadata_path`` added, or ``None`` when no
        checkpoint should be resumed.

    Raises:
        ValueError: If the pointer or metadata file is missing required paths.
    """
    if restart:
        return None

    pointer_path = Path(latest_pointer_path(run_dir, latest_filename))
    if not pointer_path.exists():
        return None

    pointer = load_json(pointer_path)
    metadata_path_value = pointer.get("metadata_path")
    if not metadata_path_value:
        raise ValueError(f"Latest pointer {pointer_path} is missing metadata_path.")

    metadata_path = Path(metadata_path_value)
    if not metadata_path.is_absolute():
        metadata_path = pointer_path.parent / metadata_path

    metadata = load_json(metadata_path)
    if "payload_path" not in metadata:
        raise ValueError(f"Checkpoint metadata {metadata_path} is missing payload_path.")

    metadata["metadata_path"] = str(metadata_path)
    return metadata
