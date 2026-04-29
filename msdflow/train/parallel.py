"""Data-parallel sharding helpers for training and metric batches."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.sharding as jshard


@dataclass(frozen=True)
class DataParallelConfig:
    """Runtime configuration for optional local data parallel training.

    Attributes:
        enabled: Whether data parallel sharding is enabled.
        axis_name: Mesh axis name used for sharding the batch dimension.
        min_devices: Minimum local device count required when enabled.
        num_devices: Number of local devices selected for the mesh.
        data_sharding: Sharding for batch arrays, or None when disabled.
        model_sharding: Replicated sharding for model/state arrays, or None
            when disabled.
    """

    enabled: bool = False
    axis_name: str = "batch"
    min_devices: int = 2
    num_devices: int = 1
    data_sharding: Any | None = None
    model_sharding: Any | None = None


def _parse_data_parallel_enabled(value: Any) -> bool:
    """Parse a user-provided data-parallel enabled flag.

    Args:
        value: Boolean-like value from a direct mapping.

    Returns:
        Parsed boolean value.

    Raises:
        ValueError: If value is not a supported boolean representation.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(
        "data_parallel.enabled must be a boolean or one of "
        "'true', 'false', '1', '0', 'yes', 'no', 'on', or 'off'; "
        f"got {value!r}"
    )


def _validate_data_parallel_config(config: DataParallelConfig) -> DataParallelConfig:
    """Validate invariants for a resolved data-parallel config.

    Args:
        config: Runtime data-parallel configuration.

    Returns:
        The validated config.

    Raises:
        ValueError: If an enabled config is missing required runtime fields.
    """
    if not config.enabled:
        return config
    if config.num_devices < config.min_devices:
        raise ValueError(
            "enabled data_parallel config requires "
            f"num_devices >= min_devices; got num_devices={config.num_devices}, "
            f"min_devices={config.min_devices}"
        )
    if config.data_sharding is None:
        raise ValueError("enabled data_parallel config requires data_sharding")
    if config.model_sharding is None:
        raise ValueError("enabled data_parallel config requires model_sharding")
    return config


def make_data_parallel_config(
    enabled: bool = False,
    axis_name: str = "batch",
    min_devices: int = 2,
) -> DataParallelConfig:
    """Create runtime sharding objects for optional local data parallelism.

    Args:
        enabled: Whether to enable data parallel sharding.
        axis_name: Name for the one-dimensional mesh axis.
        min_devices: Minimum required number of local JAX devices.

    Returns:
        DataParallelConfig with sharding objects populated only when enabled.

    Raises:
        ValueError: If min_devices is less than one, or if enabled is true and
            too few local JAX devices are visible.
    """
    min_devices = int(min_devices)
    if min_devices < 1:
        raise ValueError(f"min_devices must be >= 1, got {min_devices}")

    axis_name = str(axis_name)
    if not enabled:
        return DataParallelConfig(
            enabled=False,
            axis_name=axis_name,
            min_devices=min_devices,
            num_devices=1,
        )

    devices = jax.local_devices()
    num_devices = len(devices)
    if num_devices < min_devices:
        raise ValueError(
            "data parallel training requested but only "
            f"{num_devices} local JAX device(s) are visible; "
            f"min_devices={min_devices}"
        )

    axis_types = None
    if hasattr(jshard, "AxisType") and hasattr(jshard.AxisType, "Auto"):
        axis_types = (jshard.AxisType.Auto,)

    mesh = jax.make_mesh(
        (num_devices,),
        (axis_name,),
        devices=devices,
        axis_types=axis_types,
    )
    data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec(axis_name))
    model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())
    return DataParallelConfig(
        enabled=True,
        axis_name=axis_name,
        min_devices=min_devices,
        num_devices=num_devices,
        data_sharding=data_sharding,
        model_sharding=model_sharding,
    )


def resolve_data_parallel_config(data_parallel: Any = None) -> DataParallelConfig:
    """Resolve user/Hydra data-parallel settings into a runtime config.

    Args:
        data_parallel: None, a DataParallelConfig, or a mapping-like object
            containing enabled, axis_name, and min_devices keys.

    Returns:
        Runtime DataParallelConfig.

    Raises:
        TypeError: If data_parallel is not a supported configuration type.
    """
    if isinstance(data_parallel, DataParallelConfig):
        return _validate_data_parallel_config(data_parallel)
    if data_parallel is None:
        return make_data_parallel_config(enabled=False)

    if isinstance(data_parallel, Mapping) or hasattr(data_parallel, "get"):
        return make_data_parallel_config(
            enabled=_parse_data_parallel_enabled(data_parallel.get("enabled", False)),
            axis_name=data_parallel.get("axis_name", "batch"),
            min_devices=int(data_parallel.get("min_devices", 2)),
        )

    raise TypeError(
        "data_parallel must be None, a mapping, or DataParallelConfig; "
        f"got {type(data_parallel).__name__}"
    )


def shard_train_state(
    state: Any,
    data_parallel: DataParallelConfig,
) -> Any:
    """Replicate train state arrays when data parallelism is enabled.

    Args:
        state: Train state or compatible pytree to place on devices.
        data_parallel: Runtime data-parallel configuration.

    Returns:
        Original state when disabled, otherwise a sharded state.
    """
    if not data_parallel.enabled:
        return state
    return eqx.filter_shard(state, data_parallel.model_sharding)


def shard_model(model: Any, data_parallel: DataParallelConfig) -> Any:
    """Replicate model arrays when data parallelism is enabled.

    Args:
        model: Equinox model pytree.
        data_parallel: Runtime data-parallel configuration.

    Returns:
        Original model when disabled, otherwise a sharded model.
    """
    if not data_parallel.enabled:
        return model
    return eqx.filter_shard(model, data_parallel.model_sharding)


def _validate_batch_for_data_parallel(
    named_batch: tuple[tuple[str, jax.Array], ...],
    num_devices: int,
) -> None:
    """Validate prepared batch arrays for data-parallel sharding.

    Args:
        named_batch: Tuple of name/array pairs that share a batch axis.
        num_devices: Number of devices that will shard the batch axis.

    Raises:
        ValueError: If arrays have no batch axis, inconsistent batch sizes, or
            a batch size not divisible by num_devices.
    """
    batch_size = None
    for name, array in named_batch:
        shape = getattr(array, "shape", None)
        if shape is None or len(shape) == 0:
            raise ValueError(f"{name} must have a leading batch dimension")
        current = int(shape[0])
        if batch_size is None:
            batch_size = current
        elif current != batch_size:
            raise ValueError(
                "all data-parallel batch arrays must share the same leading "
                f"dimension; expected {batch_size}, got {current} for {name}"
            )

    if batch_size is None:
        raise ValueError("data-parallel batch cannot be empty")
    if batch_size % num_devices != 0:
        raise ValueError(
            "data-parallel batch size must be divisible by the number of "
            f"devices; got batch_size={batch_size}, num_devices={num_devices}"
        )


def shard_batch(
    batch: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    data_parallel: DataParallelConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Shard a prepared training or metric batch when data parallelism is enabled.

    Args:
        batch: Tuple ``(x_t, u_t, t, cond, cond_mask, dropout_keys)``.
        data_parallel: Runtime data-parallel configuration.

    Returns:
        Original batch when disabled, otherwise a batch placed on data sharding.

    Raises:
        ValueError: If enabled and batch dimensions are incompatible with the
            selected device mesh.
    """
    if not data_parallel.enabled:
        return batch

    x_t, u_t, t, cond, cond_mask, dropout_keys = batch
    named_batch = (
        ("x_t", x_t),
        ("u_t", u_t),
        ("t", t),
        ("cond", cond),
        ("cond_mask", cond_mask),
        ("dropout_keys", dropout_keys),
    )
    _validate_batch_for_data_parallel(named_batch, data_parallel.num_devices)
    return jax.device_put(batch, data_parallel.data_sharding)
