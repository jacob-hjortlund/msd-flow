"""Helpers for resumable training checkpoint discovery."""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import json
import math
import os
import pickle
import signal
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Any

import equinox as eqx
from omegaconf import OmegaConf
import orbax.checkpoint as ocp
from orbax.checkpoint._src.metadata import tree as tree_metadata


CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_KINDS = frozenset({"periodic", "best", "sigterm", "manual"})
_EQUINOX_STATIC_METADATA_KEY = "equinox_static_fields_v1"


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
        global_optimizer_step: Number of completed optimizer update steps.
        lr_schedule_step: Trainer-side learning-rate schedule step used for
            logging and schedule continuity.
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
    global_optimizer_step: int
    lr_schedule_step: int
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


def checkpoint_directory_name(
    checkpoint_kind: str,
    epoch: int,
    completed_microsteps: int,
) -> str:
    """Return the Orbax checkpoint directory name for a save event.

    Args:
        checkpoint_kind: Checkpoint kind, one of ``periodic``, ``best``,
            ``sigterm``, or ``manual``.
        epoch: Zero-based epoch index.
        completed_microsteps: Completed microsteps within the epoch.

    Returns:
        Human-readable Orbax checkpoint directory name.

    Raises:
        ValueError: If ``checkpoint_kind`` is unsupported.
    """
    if checkpoint_kind not in CHECKPOINT_KINDS:
        expected = ", ".join(sorted(CHECKPOINT_KINDS))
        raise ValueError(
            f"Unsupported checkpoint kind {checkpoint_kind!r}; expected {expected}."
        )
    if checkpoint_kind == "periodic":
        prefix = "checkpoint"
    else:
        prefix = checkpoint_kind
    return f"{prefix}_epoch{epoch + 1:04d}_step{completed_microsteps:04d}"


def raw_model_checkpoint_path(run_dir: str, global_optimizer_step: int) -> str:
    """Return the raw-model history checkpoint path for an optimizer step.

    Args:
        run_dir: Hash-specific checkpoint run directory.
        global_optimizer_step: One-based global optimizer update step.

    Returns:
        Path to the raw model Orbax checkpoint directory.

    Raises:
        ValueError: If ``global_optimizer_step`` is less than one.
    """
    if global_optimizer_step < 1:
        raise ValueError("global_optimizer_step must be >= 1 for raw model saves.")
    return str(Path(run_dir) / "raw_models" / f"update_{global_optimizer_step:09d}")


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


def _metadata_number(value: Any) -> float:
    """Parse finite and string-encoded non-finite metadata numbers.

    Args:
        value: JSON metadata value produced by :func:`_metadata_float`.

    Returns:
        Parsed floating-point value.
    """
    if value == "nan":
        return float("nan")
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    return float(value)


def _pytree_checkpoint_handler(
    *,
    support_rich_types: bool = True,
) -> ocp.PyTreeCheckpointHandler:
    """Return an Orbax PyTree handler that preserves rich PyTree structure.

    Args:
        support_rich_types: Whether to enable Orbax rich PyTree metadata.

    Returns:
        PyTree checkpoint handler configured for rich type metadata.
    """
    return ocp.PyTreeCheckpointHandler(
        pytree_metadata_options=tree_metadata.PyTreeMetadataOptions(
            support_rich_types=support_rich_types,
        ),
    )


def _pickle_to_text(value: Any) -> str:
    """Return a JSON-compatible pickle representation for metadata values.

    Args:
        value: Python value to encode.

    Returns:
        Base64-encoded pickle payload.
    """
    payload = pickle.dumps(value)
    return base64.b64encode(payload).decode("ascii")


def _try_pickle_to_text(value: Any) -> str | None:
    """Return encoded pickle text when a metadata value can be serialized.

    Args:
        value: Python value to encode.

    Returns:
        Base64-encoded pickle payload, or None when ``value`` cannot be
        pickled for checkpoint metadata.
    """
    try:
        return _pickle_to_text(value)
    except (pickle.PickleError, TypeError, AttributeError):
        return None


def _pickle_from_text(value: str) -> Any:
    """Decode a metadata value produced by :func:`_pickle_to_text`.

    Checkpoints are trusted local artifacts; this helper must not be used on
    untrusted metadata.

    Args:
        value: Base64-encoded pickle payload.

    Returns:
        Decoded Python value.
    """
    return pickle.loads(base64.b64decode(value.encode("ascii")))


def _path_element(kind: str, **payload: Any) -> dict[str, Any]:
    """Return a JSON-compatible static metadata path element.

    Args:
        kind: Path element kind.
        **payload: Additional JSON-compatible path element fields.

    Returns:
        Path element dictionary.
    """
    return {"kind": kind, **payload}


def _collect_equinox_static_fields(
    value: Any,
    path: tuple[dict[str, Any], ...],
    fields: list[dict[str, Any]],
    active: set[int],
) -> None:
    """Collect Equinox static fields reachable from a PyTree.

    Args:
        value: Current value to inspect.
        path: JSON-compatible path from the root item to ``value``.
        fields: Accumulator for static field metadata.
        active: Object ids in the current recursion stack.
    """
    value_id = id(value)
    if value_id in active:
        return

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        active.add(value_id)
        try:
            for field in dataclasses.fields(value):
                field_value = getattr(value, field.name)
                field_path = (*path, _path_element("attr", name=field.name))
                if field.metadata.get("static") is True:
                    encoded_value = _try_pickle_to_text(field_value)
                    if encoded_value is not None:
                        fields.append(
                            {
                                "path": list(field_path),
                                "value": encoded_value,
                            }
                        )
                else:
                    _collect_equinox_static_fields(
                        field_value,
                        field_path,
                        fields,
                        active,
                    )
        finally:
            active.remove(value_id)
        return

    if isinstance(value, Mapping):
        active.add(value_id)
        try:
            for key, item in value.items():
                _collect_equinox_static_fields(
                    item,
                    (*path, _path_element("dict_key", key=_pickle_to_text(key))),
                    fields,
                    active,
                )
        finally:
            active.remove(value_id)
        return

    if isinstance(value, tuple):
        active.add(value_id)
        try:
            for index, item in enumerate(value):
                _collect_equinox_static_fields(
                    item,
                    (*path, _path_element("index", index=index)),
                    fields,
                    active,
                )
        finally:
            active.remove(value_id)
        return

    if isinstance(value, list):
        active.add(value_id)
        try:
            for index, item in enumerate(value):
                _collect_equinox_static_fields(
                    item,
                    (*path, _path_element("index", index=index)),
                    fields,
                    active,
                )
        finally:
            active.remove(value_id)


def _equinox_static_custom_metadata(item: Any) -> dict[str, Any] | None:
    """Return Orbax custom metadata for Equinox static fields.

    Args:
        item: PyTree item that may contain Equinox modules.

    Returns:
        Custom metadata dictionary, or ``None`` when no static fields exist.
    """
    fields: list[dict[str, Any]] = []
    _collect_equinox_static_fields(item, (), fields, set())
    if not fields:
        return None
    return {
        _EQUINOX_STATIC_METADATA_KEY: {
            "version": 1,
            "fields": fields,
        }
    }


def _replace_path_value(
    value: Any,
    path: Sequence[Mapping[str, Any]],
    replacement: Any,
) -> Any:
    """Return ``value`` with one nested path replaced.

    Args:
        value: Root value to update.
        path: Path elements produced by :func:`_collect_equinox_static_fields`.
        replacement: Replacement value for the path.

    Returns:
        Updated value.
    """
    if not path:
        return replacement

    head = path[0]
    tail = path[1:]
    kind = head["kind"]
    if kind == "attr":
        name = str(head["name"])
        updated_child = _replace_path_value(getattr(value, name), tail, replacement)
        updated_value = copy.copy(value)
        object.__setattr__(updated_value, name, updated_child)
        return updated_value

    if kind == "index":
        index = int(head["index"])
        updated_child = _replace_path_value(value[index], tail, replacement)
        if isinstance(value, tuple):
            items = list(value)
            items[index] = updated_child
            return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
        items = list(value)
        items[index] = updated_child
        return items

    if kind == "dict_key":
        key = _pickle_from_text(str(head["key"]))
        updated_child = _replace_path_value(value[key], tail, replacement)
        items = dict(value)
        items[key] = updated_child
        return items

    raise ValueError(f"Unsupported Equinox static metadata path kind {kind!r}.")


def _apply_equinox_static_custom_metadata(
    item: Any,
    custom_metadata: Mapping[str, Any] | None,
) -> Any:
    """Apply saved Equinox static field metadata to a restored PyTree.

    Args:
        item: Restored PyTree item.
        custom_metadata: Orbax checkpoint custom metadata.

    Returns:
        Restored item with saved Equinox static fields applied.
    """
    if not custom_metadata:
        return item

    payload = custom_metadata.get(_EQUINOX_STATIC_METADATA_KEY)
    if not isinstance(payload, Mapping):
        return item
    if payload.get("version") != 1:
        raise ValueError("Unsupported Equinox static metadata version.")

    restored = item
    for field in payload.get("fields", []):
        if not isinstance(field, Mapping):
            raise ValueError("Invalid Equinox static metadata field.")
        path = field.get("path")
        value = field.get("value")
        if not isinstance(path, list) or not isinstance(value, str):
            raise ValueError("Invalid Equinox static metadata field.")
        restored = _replace_path_value(restored, path, _pickle_from_text(value))
    return restored


def _item_custom_metadata(step_metadata: Any) -> Mapping[str, Any] | None:
    """Return item-level Orbax custom metadata from checkpoint metadata.

    Args:
        step_metadata: Orbax step metadata object returned by a checkpointer.

    Returns:
        Item-level custom metadata when present.
    """
    item_metadata = getattr(step_metadata, "item_metadata", None)
    custom_metadata = getattr(item_metadata, "custom_metadata", None)
    if isinstance(custom_metadata, Mapping):
        return custom_metadata
    return None


class OrbaxAsyncCheckpointIO:
    """Own asynchronous Orbax checkpointers for training and raw model saves."""

    def __init__(self) -> None:
        """Create separate async save streams for checkpoints and raw history."""
        handler = _pytree_checkpoint_handler()
        raw_handler = _pytree_checkpoint_handler()
        self.training_checkpointer = ocp.AsyncCheckpointer(handler)
        self.raw_model_checkpointer = ocp.AsyncCheckpointer(raw_handler)
        self._pending_training_sidecars: list[
            tuple[Path, Mapping[str, Any], Path | None]
        ] = []

    def defer_training_sidecar_publication(
        self,
        metadata_path: str | Path,
        metadata: Mapping[str, Any],
        latest_path: str | Path | None,
    ) -> None:
        """Publish checkpoint sidecars after the next training save finalizes.

        Args:
            metadata_path: Destination ``metadata.json`` path.
            metadata: JSON-compatible checkpoint metadata.
            latest_path: Destination latest-pointer path, or ``None`` when the
                latest pointer should not be updated.
        """
        self._pending_training_sidecars.append(
            (
                Path(metadata_path),
                dict(metadata),
                None if latest_path is None else Path(latest_path),
            )
        )

    def _publish_pending_training_sidecars(self) -> None:
        """Write deferred checkpoint sidecars for finalized training saves."""
        while self._pending_training_sidecars:
            metadata_path, metadata, latest_path = self._pending_training_sidecars[0]
            _atomic_write_json(metadata_path, metadata)
            if latest_path is not None:
                _atomic_write_json(latest_path, {"metadata_path": str(metadata_path)})
            self._pending_training_sidecars.pop(0)

    def save_training_item(self, path: str | Path, item: Any) -> None:
        """Start an async save for a resumable training checkpoint.

        Args:
            path: Destination Orbax checkpoint directory.
            item: PyTree item to save.
        """
        self.training_checkpointer.save(
            Path(path),
            args=ocp.args.PyTreeSave(
                item,
                custom_metadata=_equinox_static_custom_metadata(item),
            ),
            force=True,
        )

    def save_raw_model_item(self, path: str | Path, item: Any) -> None:
        """Start an async save for a raw model history checkpoint.

        Args:
            path: Destination Orbax checkpoint directory.
            item: PyTree item to save.
        """
        self.raw_model_checkpointer.save(
            Path(path),
            args=ocp.args.PyTreeSave(
                item,
                custom_metadata=_equinox_static_custom_metadata(item),
            ),
            force=True,
        )

    def wait_training(self) -> None:
        """Wait for the current training checkpoint save to finish."""
        self.training_checkpointer.wait_until_finished()
        self._publish_pending_training_sidecars()

    def wait_all(self) -> None:
        """Wait for all outstanding async saves to finish."""
        self.wait_training()
        self.raw_model_checkpointer.wait_until_finished()

    def restore_item(self, path: str | Path, target: Any) -> Any:
        """Restore an Orbax item using a target item shape.

        Args:
            path: Source Orbax checkpoint directory.
            target: Target PyTree used to define restore structure.

        Returns:
            Restored PyTree item.
        """
        checkpoint_path = Path(path)
        checkpointer = ocp.Checkpointer(_pytree_checkpoint_handler())
        try:
            try:
                restored = checkpointer.restore(
                    checkpoint_path,
                    args=ocp.args.PyTreeRestore(target),
                )
                step_metadata = checkpointer.metadata(checkpoint_path)
            except (NotImplementedError, TypeError):
                checkpointer.close()
                checkpointer = ocp.Checkpointer(
                    _pytree_checkpoint_handler(support_rich_types=False)
                )
                restored = checkpointer.restore(
                    checkpoint_path,
                    args=ocp.args.PyTreeRestore(target),
                )
                step_metadata = checkpointer.metadata(checkpoint_path)
            return _apply_equinox_static_custom_metadata(
                restored,
                _item_custom_metadata(step_metadata),
            )
        finally:
            checkpointer.close()

    def close(self) -> None:
        """Close Orbax checkpointers after all work is finished."""
        self.training_checkpointer.close()
        self.raw_model_checkpointer.close()


def build_checkpoint_metadata(
    *,
    stable_hash: str,
    checkpoint_kind: str,
    checkpoint_path: str,
    grad_accum_steps: int,
    microsteps_per_epoch: int,
    global_optimizer_step: int | None = None,
    lr_schedule_step: int | None = None,
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
        checkpoint_kind: Checkpoint kind, one of ``periodic``, ``best``,
            ``sigterm``, or ``manual``.
        checkpoint_path: Path to the checkpoint directory.
        grad_accum_steps: Gradient accumulation steps per optimizer update.
        microsteps_per_epoch: Number of microsteps in a full epoch.
        global_optimizer_step: Number of completed optimizer update steps.
            Defaults to ``checkpoint.global_optimizer_step``.
        lr_schedule_step: Trainer-side learning-rate schedule step used for
            logging and schedule continuity. Defaults to
            ``checkpoint.lr_schedule_step``.
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
        global_optimizer_step = (
            checkpoint.global_optimizer_step
            if global_optimizer_step is None
            else global_optimizer_step
        )
        lr_schedule_step = (
            checkpoint.lr_schedule_step
            if lr_schedule_step is None
            else lr_schedule_step
        )

    if epoch is None:
        raise ValueError("Checkpoint metadata requires epoch.")
    if completed_microsteps is None:
        raise ValueError("Checkpoint metadata requires completed_microsteps.")
    if ema_initialized is None:
        raise ValueError("Checkpoint metadata requires ema_initialized.")
    if global_optimizer_step is None:
        raise ValueError("Checkpoint metadata requires global_optimizer_step.")
    if lr_schedule_step is None:
        raise ValueError("Checkpoint metadata requires lr_schedule_step.")

    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "stable_hash": stable_hash,
        "checkpoint_kind": checkpoint_kind,
        "epoch": int(epoch),
        "completed_microsteps": int(completed_microsteps),
        "checkpoint_path": str(checkpoint_path),
        "grad_accum_steps": int(grad_accum_steps),
        "microsteps_per_epoch": int(microsteps_per_epoch),
        "global_optimizer_step": int(global_optimizer_step),
        "lr_schedule_step": int(lr_schedule_step),
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

    checkpoint_path = metadata.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("Checkpoint metadata is missing checkpoint_path.")

    for field_name in ("global_optimizer_step", "lr_schedule_step"):
        value = metadata.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Checkpoint metadata has invalid {field_name}.")

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
            "Checkpoint completed_microsteps must be within [0, microsteps_per_epoch]."
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
    checkpoint_io: OrbaxAsyncCheckpointIO | None = None,
    update_latest: bool = True,
    wait: bool = True,
    directory_epoch: int | None = None,
) -> dict[str, Any]:
    """Save a resumable Orbax training checkpoint and optional latest pointer.

    Args:
        run_dir: Directory where checkpoint files should be written.
        checkpoint: Full training checkpoint payload to serialize.
        stable_hash: Stable configuration compatibility hash.
        checkpoint_kind: Checkpoint kind, one of ``periodic``, ``best``,
            ``sigterm``, or ``manual``.
        grad_accum_steps: Gradient accumulation steps per optimizer update.
        microsteps_per_epoch: Number of microsteps in a full epoch.
        monitor: Name of the monitored metric.
        monitor_mode: Optimization direction for the monitored metric.
        clearml_task_id: Optional ClearML task id associated with the run.
        latest_filename: Filename for the latest-checkpoint pointer JSON.
        source_checkpoint_path: Optional source checkpoint path for resumed runs.
        hash_payload: Normalized configuration payload used to compute the hash.
        checkpoint_io: Optional shared Orbax async checkpoint I/O owner.
        update_latest: Whether to update the latest pointer JSON.
        wait: Whether to wait for the save and write metadata immediately. When
            ``False``, metadata publication is deferred until
            ``checkpoint_io.wait_training()`` or ``checkpoint_io.wait_all()``.
            If no shared ``checkpoint_io`` is supplied, the owned I/O is still
            finalized before return.
        directory_epoch: Optional zero-based epoch index used only for the
            checkpoint directory name. Defaults to ``checkpoint.epoch``.

    Returns:
        Metadata written for the checkpoint, including metadata path.
    """
    io = checkpoint_io or OrbaxAsyncCheckpointIO()
    owns_io = checkpoint_io is None
    latest_pathname = _latest_filename_path(latest_filename)
    run_path = Path(run_dir).expanduser().resolve()
    path_epoch = checkpoint.epoch if directory_epoch is None else int(directory_epoch)
    checkpoint_path = run_path / checkpoint_directory_name(
        checkpoint_kind,
        path_epoch,
        checkpoint.completed_microsteps,
    )
    metadata_path = checkpoint_path / "metadata.json"
    latest_path = run_path / latest_pathname

    metadata = build_checkpoint_metadata(
        stable_hash=stable_hash,
        checkpoint_kind=checkpoint_kind,
        checkpoint=checkpoint,
        checkpoint_path=str(checkpoint_path),
        grad_accum_steps=grad_accum_steps,
        microsteps_per_epoch=microsteps_per_epoch,
        global_optimizer_step=checkpoint.global_optimizer_step,
        lr_schedule_step=checkpoint.lr_schedule_step,
        monitor=monitor,
        monitor_mode=monitor_mode,
        clearml_task_id=clearml_task_id,
        source_checkpoint_path=source_checkpoint_path,
        hash_payload=hash_payload,
    )
    metadata["metadata_path"] = str(metadata_path)
    metadata.update(
        {
            "epoch_loss": _metadata_float(checkpoint.epoch_loss),
            "patience_counter": int(checkpoint.patience_counter),
            "total_epoch_time": _metadata_float(checkpoint.total_epoch_time),
            "total_train_time": _metadata_float(checkpoint.total_train_time),
            "total_val_time": _metadata_float(checkpoint.total_val_time),
            "val_runs": int(checkpoint.val_runs),
            "val_time": _metadata_float(checkpoint.val_time),
            "val_metrics": _json_safe(checkpoint.val_metrics),
            "train_metrics": _json_safe(checkpoint.train_metrics),
            "epoch_metric_results": _json_safe(checkpoint.epoch_metric_results),
        }
    )
    validate_checkpoint_metadata(
        metadata,
        stable_hash=stable_hash,
        monitor=monitor,
        monitor_mode=monitor_mode,
        microsteps_per_epoch=microsteps_per_epoch,
    )

    try:
        run_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        io.save_training_item(checkpoint_path, checkpoint)
        if wait:
            io.wait_training()
            _atomic_write_json(metadata_path, metadata)
            if update_latest:
                _atomic_write_json(latest_path, {"metadata_path": str(metadata_path)})
        else:
            io.defer_training_sidecar_publication(
                metadata_path,
                metadata,
                latest_path if update_latest else None,
            )
    finally:
        if owns_io:
            try:
                io.wait_all()
            finally:
                io.close()

    return metadata


def save_raw_model_checkpoint(
    *,
    run_dir: str,
    model: Any,
    global_optimizer_step: int,
    checkpoint_io: OrbaxAsyncCheckpointIO,
) -> str:
    """Save a raw model history checkpoint for one optimizer update.

    Args:
        run_dir: Hash-specific checkpoint run directory.
        model: Model PyTree to save.
        global_optimizer_step: One-based global optimizer update step.
        checkpoint_io: Shared Orbax async checkpoint I/O owner.

    Returns:
        Path to the raw model Orbax checkpoint directory.
    """
    path = raw_model_checkpoint_path(run_dir, global_optimizer_step)
    checkpoint_io.save_raw_model_item(path, model)
    return path


def load_training_checkpoint(
    path: str | Path,
    like: TrainingCheckpoint,
    *,
    metadata: Mapping[str, Any] | None = None,
    checkpoint_io: OrbaxAsyncCheckpointIO | None = None,
) -> TrainingCheckpoint:
    """Deserialize an Orbax training checkpoint using a matching example tree.

    Args:
        path: Path to an Orbax checkpoint directory.
        like: Checkpoint tree with the same structure as the serialized payload.
        metadata: Optional metadata associated with the checkpoint.
        checkpoint_io: Optional shared Orbax async checkpoint I/O owner.

    Returns:
        Restored training checkpoint payload.
    """
    checkpoint_path = Path(path)
    if metadata is None:
        metadata = load_json(checkpoint_path / "metadata.json")
    _ = metadata
    io = checkpoint_io or OrbaxAsyncCheckpointIO()
    owns_io = checkpoint_io is None
    try:
        checkpoint = io.restore_item(checkpoint_path, like)
    finally:
        if owns_io:
            io.close()
    return checkpoint


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
    checkpoint_path = metadata.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError(
            f"Checkpoint metadata {metadata_path} has invalid checkpoint_path."
        )
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.is_absolute():
        checkpoint_path_obj = metadata_path.parent / checkpoint_path_obj
        metadata["checkpoint_path"] = str(checkpoint_path_obj)
    if not checkpoint_path_obj.is_dir():
        raise ValueError(f"Checkpoint directory {checkpoint_path_obj} does not exist.")

    metadata["metadata_path"] = str(metadata_path)
    return metadata
