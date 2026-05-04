"""Helpers for resumable training checkpoint discovery."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Any

import equinox as eqx
from omegaconf import OmegaConf


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KINDS = frozenset({"periodic", "sigterm", "manual"})
_MISSING = object()


class SigtermFlag:
    """Context manager that records SIGTERM requests for graceful shutdown."""

    def __init__(self, enabled: bool = True) -> None:
        """Initialize the SIGTERM request flag.

        Args:
            enabled: Whether to install a temporary SIGTERM handler.
        """
        self.enabled = enabled
        self.requested = False
        self.previous_handler: Any | None = None

    def handle(self, signum: int, frame: FrameType | None) -> None:
        """Record that SIGTERM was requested.

        Args:
            signum: Signal number received by the process.
            frame: Current execution frame supplied by the signal module.
        """
        self.requested = True

    def __enter__(self) -> "SigtermFlag":
        """Install the SIGTERM handler when the flag is enabled.

        Returns:
            This flag instance.
        """
        if self.enabled:
            self.previous_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self.handle)
        return self

    def __exit__(self, *args: object) -> None:
        """Restore the previous SIGTERM handler when one was installed.

        Args:
            *args: Context manager exception details.
        """
        if self.enabled and self.previous_handler is not None:
            signal.signal(signal.SIGTERM, self.previous_handler)


class TrainingCheckpoint(eqx.Module):
    """Full training state payload for resumable checkpoints.

    Attributes:
        state: Current train state containing model and optimizer state.
        ema_model: Exponential moving average model snapshot.
        ema_initialized: Whether the EMA model has received its first update.
        key: Main training PRNG key.
        sampling_key: Sampling PRNG key.
        epoch: Zero-based epoch index for the checkpoint.
        completed_microsteps: Completed microsteps within the current epoch.
        epoch_loss: Accumulated epoch loss value.
        best_metric_value: Best monitored metric value observed so far.
        best_epoch: Epoch index associated with the best monitored metric.
        patience_counter: Current early-stopping patience counter.
        total_epoch_time: Cumulative epoch loop wall time.
        total_train_time: Cumulative training step wall time.
        total_val_time: Cumulative validation wall time.
        val_runs: Number of validation runs completed.
        val_time: Most recent validation wall time.
        val_metrics: Validation metric values from the checkpoint epoch.
        train_metrics: Training metric values from the checkpoint epoch.
        epoch_metric_results: Additional epoch-level metric values.
    """

    state: Any
    ema_model: Any
    ema_initialized: bool
    key: Any
    sampling_key: Any
    epoch: int
    completed_microsteps: int
    epoch_loss: float
    best_metric_value: float
    best_epoch: int | None
    patience_counter: int
    total_epoch_time: float
    total_train_time: float
    total_val_time: float
    val_runs: int
    val_time: float
    val_metrics: dict[str, float]
    train_metrics: dict[str, float]
    epoch_metric_results: dict[str, float]


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


def _path_part(container: Any, part: str) -> Any:
    """Return one dotted-path part from a mapping or sequence.

    Args:
        container: Current container while traversing a normalized payload.
        part: Dotted-path component to read as a mapping key or list index.

    Returns:
        The value at the requested part, or a missing-value sentinel when the
        path cannot be traversed.
    """
    if isinstance(container, dict):
        return container.get(part, _MISSING)
    if isinstance(container, list) and part.isdecimal():
        index = int(part)
        if 0 <= index < len(container):
            return container[index]
    return _MISSING


def _get_path(payload: dict[str, Any], dotted_path: str) -> Any:
    """Return a dotted path from a normalized payload when present.

    Args:
        payload: Normalized configuration payload to inspect.
        dotted_path: Dot-separated key path such as
            ``"train.batch_metrics.0.project_velocity"``.

    Returns:
        The value at the path, or a missing-value sentinel when absent.
    """
    current: Any = payload
    for part in (part for part in dotted_path.split(".") if part):
        current = _path_part(current, part)
        if current is _MISSING:
            return _MISSING
    return current


def _drop_path(payload: dict[str, Any], dotted_path: str) -> None:
    """Remove a dotted path from a nested container if it is present.

    Args:
        payload: Normalized configuration payload to mutate.
        dotted_path: Dot-separated key path such as
            ``"train.batch_metrics.0.project_velocity"``.
    """
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        return

    current: Any = payload
    for part in parts[:-1]:
        current = _path_part(current, part)
        if current is _MISSING:
            return

    if isinstance(current, dict):
        current.pop(parts[-1], None)
    elif isinstance(current, list) and parts[-1].isdecimal():
        index = int(parts[-1])
        if 0 <= index < len(current):
            current.pop(index)


def _drop_path_if_default(
    payload: dict[str, Any],
    dotted_path: str,
    default_value: Any,
) -> None:
    """Drop a path only when its normalized value equals a default.

    Args:
        payload: Normalized configuration payload to mutate.
        dotted_path: Dot-separated key path to conditionally remove.
        default_value: Default value to compare against after normalization.
    """
    current_value = _get_path(payload, dotted_path)
    if current_value is _MISSING:
        return
    if current_value == _json_safe(default_value):
        _drop_path(payload, dotted_path)


def _drop_default_excluded_paths(payload: dict[str, Any]) -> None:
    """Apply config-driven conditional default exclusions.

    Args:
        payload: Normalized configuration payload to mutate.
    """
    train = payload.get("train")
    if not isinstance(train, dict):
        return
    resume = train.get("resume")
    if not isinstance(resume, dict):
        return
    default_excludes = resume.get("hash_exclude_if_default", {})
    if not isinstance(default_excludes, Mapping):
        return

    for path, default_value in default_excludes.items():
        _drop_path_if_default(payload, str(path), default_value)


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

    _drop_default_excluded_paths(payload)

    for path in exclude_paths or ():
        _drop_path(payload, path)

    return payload


def compute_config_hash(
    cfg: Any,
    exclude_paths: Sequence[str] | None,
    length: int = 16,
) -> tuple[str, dict[str, Any]]:
    """Compute a stable truncated SHA-256 hash for a configuration.

    Args:
        cfg: Hydra/OmegaConf configuration or equivalent mapping.
        exclude_paths: Dot-separated paths to remove before hashing.
        length: Number of hex characters to keep from the SHA-256 digest.

    Returns:
        Tuple of ``(stable_hash, normalized_payload)``.
    """
    payload = normalized_config_payload(cfg, exclude_paths=exclude_paths)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    stable_hash = hashlib.sha256(encoded).hexdigest()[:length]
    return stable_hash, payload


def checkpoint_filename_stem(epoch: int, completed_microsteps: int) -> str:
    """Return the checkpoint filename stem for an epoch and microstep.

    Args:
        epoch: Zero-based epoch index.
        completed_microsteps: Completed microsteps within the epoch.

    Returns:
        Filename stem using one-based epoch numbering and padded microsteps.
    """
    return f"checkpoint_epoch{epoch + 1:04d}_step{completed_microsteps:04d}"


def _metadata_float(value: float | int | None) -> float | str | None:
    """Convert a metric value to a JSON-safe metadata scalar.

    Args:
        value: Numeric value to store in checkpoint metadata.

    Returns:
        Finite values as ``float`` and non-finite values as strings.
    """
    if value is None:
        return None

    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return number


def build_checkpoint_metadata(
    *,
    stable_hash: str,
    checkpoint_kind: str,
    payload_path: str,
    grad_accum_steps: int,
    microsteps_per_epoch: int,
    monitor: str | None,
    monitor_mode: str | None,
    clearml_task_id: str | None,
    source_checkpoint_path: str | None,
    hash_payload: Mapping[str, Any] | None,
    checkpoint: TrainingCheckpoint | None = None,
    epoch: int | None = None,
    completed_microsteps: int | None = None,
    ema_initialized: bool | None = None,
    best_metric_value: float | int | None = None,
    best_epoch: int | None = None,
) -> dict[str, Any]:
    """Build JSON-safe metadata for a training checkpoint.

    Args:
        stable_hash: Stable configuration compatibility hash.
        checkpoint_kind: Checkpoint kind, one of ``periodic``, ``sigterm``, or
            ``manual``.
        payload_path: Path to the serialized Equinox payload.
        grad_accum_steps: Gradient accumulation steps per optimizer update.
        microsteps_per_epoch: Number of microsteps in a full epoch.
        monitor: Name of the monitored metric.
        monitor_mode: Optimization direction for the monitored metric.
        clearml_task_id: Optional ClearML task id associated with the run.
        source_checkpoint_path: Optional source checkpoint path for resumed runs.
        hash_payload: Normalized configuration payload used to compute the hash.
        checkpoint: Optional checkpoint payload to use for state-derived fields.
        epoch: Zero-based epoch index. Defaults to ``checkpoint.epoch``.
        completed_microsteps: Completed microsteps within the epoch. Defaults to
            ``checkpoint.completed_microsteps``.
        ema_initialized: Whether the EMA model has been initialized.
        best_metric_value: Best monitored metric value observed so far.
        best_epoch: Epoch index associated with the best monitored metric.

    Returns:
        JSON-compatible checkpoint metadata.

    Raises:
        ValueError: If ``checkpoint_kind`` is unsupported.
    """
    if checkpoint_kind not in CHECKPOINT_KINDS:
        expected = ", ".join(sorted(CHECKPOINT_KINDS))
        raise ValueError(
            f"Unsupported checkpoint kind {checkpoint_kind!r}; expected {expected}."
        )

    if checkpoint is not None:
        epoch = checkpoint.epoch if epoch is None else epoch
        completed_microsteps = (
            checkpoint.completed_microsteps
            if completed_microsteps is None
            else completed_microsteps
        )
        ema_initialized = (
            checkpoint.ema_initialized if ema_initialized is None else ema_initialized
        )
        best_metric_value = (
            checkpoint.best_metric_value
            if best_metric_value is None
            else best_metric_value
        )
        best_epoch = checkpoint.best_epoch if best_epoch is None else best_epoch

    if epoch is None:
        raise ValueError("Checkpoint metadata requires epoch.")
    if completed_microsteps is None:
        raise ValueError("Checkpoint metadata requires completed_microsteps.")
    if ema_initialized is None:
        raise ValueError("Checkpoint metadata requires ema_initialized.")

    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "stable_hash": stable_hash,
        "checkpoint_kind": checkpoint_kind,
        "epoch": int(epoch),
        "completed_microsteps": int(completed_microsteps),
        "payload_path": str(payload_path),
        "grad_accum_steps": int(grad_accum_steps),
        "microsteps_per_epoch": int(microsteps_per_epoch),
        "monitor": monitor,
        "monitor_mode": monitor_mode,
        "clearml_task_id": clearml_task_id,
        "source_checkpoint_path": (
            None if source_checkpoint_path is None else str(source_checkpoint_path)
        ),
        "hash_payload": _json_safe(hash_payload),
        "ema_initialized": bool(ema_initialized),
        "best_metric_value": _metadata_float(best_metric_value),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "saved_at_unix": _metadata_float(time.time()),
    }


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    stable_hash: str | None = None,
    expected_hash: str | None = None,
    monitor: str | None = None,
    monitor_mode: str | None = None,
    microsteps_per_epoch: int | None = None,
    allow_hash_override: bool = False,
) -> None:
    """Validate checkpoint metadata before saving or loading.

    Args:
        metadata: Metadata object to validate.
        stable_hash: Expected stable configuration hash.
        expected_hash: Backward-compatible alias for ``stable_hash``.
        monitor: Expected monitored metric name.
        monitor_mode: Expected monitor mode.
        microsteps_per_epoch: Optional active microstep count used for bounds
            validation. Defaults to the value stored in metadata.
        allow_hash_override: Whether to allow stable hash mismatches.

    Raises:
        ValueError: If metadata is incompatible with the current run.
    """
    schema_version = metadata.get("schema_version")
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema version {schema_version!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )

    active_hash = stable_hash if stable_hash is not None else expected_hash
    if active_hash is None:
        raise ValueError("Checkpoint validation requires a stable hash.")

    metadata_hash = metadata.get("stable_hash")
    if not allow_hash_override and metadata_hash != active_hash:
        raise ValueError(
            f"Checkpoint stable hash {metadata_hash!r} does not match {active_hash!r}."
        )

    checkpoint_kind = metadata.get("checkpoint_kind")
    if checkpoint_kind not in CHECKPOINT_KINDS:
        expected = ", ".join(sorted(CHECKPOINT_KINDS))
        raise ValueError(
            "Checkpoint metadata has invalid checkpoint kind "
            f"{checkpoint_kind!r}; expected {expected}."
        )

    if metadata.get("monitor") != monitor:
        raise ValueError(
            f"Checkpoint monitor {metadata.get('monitor')!r} does not match {monitor!r}."
        )

    if metadata.get("monitor_mode") != monitor_mode:
        raise ValueError(
            "Checkpoint monitor mode "
            f"{metadata.get('monitor_mode')!r} does not match {monitor_mode!r}."
        )

    payload_path = metadata.get("payload_path")
    if not isinstance(payload_path, str) or not payload_path:
        raise ValueError("Checkpoint metadata is missing payload_path.")

    microsteps_limit = (
        microsteps_per_epoch
        if microsteps_per_epoch is not None
        else metadata.get("microsteps_per_epoch")
    )
    completed_microsteps = metadata.get("completed_microsteps")
    if not isinstance(microsteps_limit, int) or microsteps_limit < 0:
        raise ValueError("Checkpoint metadata has invalid microsteps_per_epoch.")
    if (
        not isinstance(completed_microsteps, int)
        or completed_microsteps < 0
        or completed_microsteps > microsteps_limit
    ):
        raise ValueError(
            "Checkpoint completed_microsteps must be within "
            "[0, microsteps_per_epoch]."
        )


def _atomic_write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    """Atomically write a JSON object to disk.

    Args:
        path: Destination JSON path.
        data: JSON-compatible object to write.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=destination.parent,
            encoding="utf-8",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(data, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _latest_filename_path(latest_filename: str) -> Path:
    """Return a validated latest pointer filename path.

    Args:
        latest_filename: Relative filename for the latest pointer JSON.

    Returns:
        Relative path that is safe to join under a checkpoint run directory.

    Raises:
        ValueError: If ``latest_filename`` is absolute, empty, or contains
            parent-directory traversal.
    """
    filename = os.fspath(latest_filename)
    path = Path(filename)
    if not filename or path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "latest_filename must be a relative path inside run_dir without "
            "parent traversal."
        )
    return path


def save_training_checkpoint(
    *,
    run_dir: str,
    checkpoint: TrainingCheckpoint,
    stable_hash: str,
    checkpoint_kind: str,
    grad_accum_steps: int,
    microsteps_per_epoch: int,
    monitor: str | None,
    monitor_mode: str | None,
    clearml_task_id: str | None,
    latest_filename: str,
    source_checkpoint_path: str | None,
    hash_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize a training checkpoint and write metadata plus latest pointer.

    Args:
        run_dir: Directory where checkpoint files should be written.
        checkpoint: Full training checkpoint payload to serialize.
        stable_hash: Stable configuration compatibility hash.
        checkpoint_kind: Checkpoint kind, one of ``periodic``, ``sigterm``, or
            ``manual``.
        grad_accum_steps: Gradient accumulation steps per optimizer update.
        microsteps_per_epoch: Number of microsteps in a full epoch.
        monitor: Name of the monitored metric.
        monitor_mode: Optimization direction for the monitored metric.
        clearml_task_id: Optional ClearML task id associated with the run.
        latest_filename: Filename for the latest-checkpoint pointer JSON.
        source_checkpoint_path: Optional source checkpoint path for resumed runs.
        hash_payload: Normalized configuration payload used to compute the hash.

    Returns:
        Metadata written for the checkpoint, including metadata and payload paths.
    """
    latest_pathname = _latest_filename_path(latest_filename)
    run_path = Path(run_dir).expanduser().resolve()
    stem = checkpoint_filename_stem(
        epoch=checkpoint.epoch,
        completed_microsteps=checkpoint.completed_microsteps,
    )
    payload_path = run_path / f"{stem}.eqx"
    metadata_path = run_path / f"{stem}.json"
    latest_path = run_path / latest_pathname

    metadata = build_checkpoint_metadata(
        stable_hash=stable_hash,
        checkpoint_kind=checkpoint_kind,
        epoch=checkpoint.epoch,
        completed_microsteps=checkpoint.completed_microsteps,
        payload_path=str(payload_path),
        grad_accum_steps=grad_accum_steps,
        microsteps_per_epoch=microsteps_per_epoch,
        monitor=monitor,
        monitor_mode=monitor_mode,
        clearml_task_id=clearml_task_id,
        source_checkpoint_path=source_checkpoint_path,
        hash_payload=hash_payload,
        ema_initialized=checkpoint.ema_initialized,
        best_metric_value=checkpoint.best_metric_value,
        best_epoch=checkpoint.best_epoch,
    )
    metadata["metadata_path"] = str(metadata_path)
    validate_checkpoint_metadata(
        metadata,
        stable_hash=stable_hash,
        monitor=monitor,
        monitor_mode=monitor_mode,
        allow_hash_override=False,
    )

    run_path.mkdir(parents=True, exist_ok=True)
    temp_payload_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=run_path,
            suffix=".eqx.tmp",
            delete=False,
        ) as handle:
            temp_payload_path = Path(handle.name)
        eqx.tree_serialise_leaves(temp_payload_path, checkpoint)
        with temp_payload_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_payload_path, payload_path)
    except Exception:
        if temp_payload_path is not None:
            temp_payload_path.unlink(missing_ok=True)
        raise

    _atomic_write_json(metadata_path, metadata)
    _atomic_write_json(latest_path, {"metadata_path": str(metadata_path)})
    return metadata


def load_training_checkpoint(
    path: str | Path,
    like: TrainingCheckpoint,
) -> TrainingCheckpoint:
    """Deserialize a training checkpoint using a matching example tree.

    Args:
        path: Path to a serialized Equinox checkpoint payload.
        like: Checkpoint tree with the same structure as the serialized payload.

    Returns:
        Restored training checkpoint payload.
    """
    return eqx.tree_deserialise_leaves(path, like)


def checkpoint_run_dir(root: str, stable_hash: str) -> str:
    """Return the hash-specific checkpoint directory path.

    Args:
        root: Root directory for resumable checkpoints.
        stable_hash: Stable configuration hash for the run family.

    Returns:
        Path to the directory containing checkpoints for ``stable_hash``.
    """
    return str(Path(root) / stable_hash)


def latest_pointer_path(run_dir: str, latest_filename: str) -> str:
    """Return the latest-checkpoint pointer path for a run directory.

    Args:
        run_dir: Hash-specific checkpoint directory.
        latest_filename: Filename used for the latest checkpoint pointer.

    Returns:
        Path to the latest checkpoint pointer JSON file.
    """
    return str(Path(run_dir) / _latest_filename_path(latest_filename))


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
    payload_path = metadata.get("payload_path")
    if not isinstance(payload_path, str) or not payload_path:
        raise ValueError(
            f"Checkpoint metadata {metadata_path} has invalid payload_path."
        )

    metadata["metadata_path"] = str(metadata_path)
    return metadata
