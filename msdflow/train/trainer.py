"""Training loop and state management for flow matching.

Provides ``TrainState``, a JIT-compiled train step factory, and
the main training loop with periodic checkpointing.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import inspect
import time
import queue
import threading
import numbers
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import functools

import os
import logging
import numpy as np

from tqdm import tqdm
from msdflow.utils import register_all_resolvers
from tqdm.contrib.logging import logging_redirect_tqdm
from msdflow.tracking import (
    log_checkpoint,
    log_metrics,
    log_samples,
    log_time_binned_loss,
)
from msdflow.train.checkpointing import (
    SigtermFlag,
    TrainingCheckpoint,
    load_json,
    load_training_checkpoint,
    save_training_checkpoint,
    validate_checkpoint_metadata,
)
from msdflow.train.metrics import (
    TimeBinnedLossHistory,
    TimeBinnedLossResult,
    make_time_binned_loss_step,
)
from msdflow.train.parallel import (
    DataParallelConfig,
    _parse_data_parallel_enabled,
    _validate_batch_for_data_parallel,
    _validate_data_parallel_config,
    make_data_parallel_config,
    resolve_data_parallel_config,
    shard_batch,
    shard_model,
    shard_train_state,
)

logger = logging.getLogger(__name__)

register_all_resolvers()


class TrainState(eqx.Module):
    """Bundles the model and optimizer state for checkpointing.

    Attributes:
        model: UNet velocity-field network.
        opt_state: Optax optimizer state.
    """

    model: Any  # UNet
    opt_state: Any  # optax.OptState


@dataclass(frozen=True)
class TimeLossDiagnosticConfig:
    """Resolved configuration for the time-binned loss diagnostic.

    Attributes:
        enabled: Whether to run the diagnostic.
        split: Split to evaluate: ``"val"``, ``"train"``, or ``"both"``.
        num_bins: Number of equal-width bins over ``[0, 1]``.
        num_batches: Maximum batches per split; ``0`` means all batches.
        log_heatmap: Whether to log cumulative heatmap figures.
    """

    enabled: bool = False
    split: str = "val"
    num_bins: int = 20
    num_batches: int = 0
    log_heatmap: bool = True


def _parse_diagnostic_bool(value: Any, name: str) -> bool:
    """Parse a boolean-like diagnostic config value.

    Args:
        value: Boolean or supported boolean string.
        name: Config field name for error messages.

    Returns:
        Parsed boolean.

    Raises:
        ValueError: If the value is not a supported boolean representation.
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
        f"{name} must be a boolean or one of "
        "'true', 'false', '1', '0', 'yes', 'no', 'on', or 'off'; "
        f"got {value!r}"
    )


def _parse_diagnostic_int(value: Any, name: str) -> int:
    """Parse an integer-like diagnostic config value.

    Args:
        value: Value to parse as an integer.
        name: Config field name for error messages.

    Returns:
        Parsed integer.

    Raises:
        ValueError: If the value is not a valid integer representation.
    """
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer; got {value!r}")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        if float(value).is_integer():
            return int(value)
        raise ValueError(f"{name} must be an integer; got {value!r}")
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        raise ValueError(f"{name} must be an integer; got {value!r}")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("+-").isdigit():
            return int(stripped)
        raise ValueError(f"{name} must be an integer; got {value!r}")
    raise ValueError(f"{name} must be an integer; got {value!r}")


def resolve_time_loss_diagnostic_config(
    time_loss_diagnostic: Any = None,
) -> TimeLossDiagnosticConfig:
    """Resolve user/Hydra diagnostic settings into a runtime config.

    Args:
        time_loss_diagnostic: None, TimeLossDiagnosticConfig, or mapping-like
            object with enabled, split, num_bins, num_batches, and log_heatmap.

    Returns:
        Resolved diagnostic config.

    Raises:
        TypeError: If the config object is unsupported.
        ValueError: If a config value is invalid.
    """
    if time_loss_diagnostic is None:
        return TimeLossDiagnosticConfig()
    if isinstance(time_loss_diagnostic, TimeLossDiagnosticConfig):
        config_values = {
            "enabled": time_loss_diagnostic.enabled,
            "split": time_loss_diagnostic.split,
            "num_bins": time_loss_diagnostic.num_bins,
            "num_batches": time_loss_diagnostic.num_batches,
            "log_heatmap": time_loss_diagnostic.log_heatmap,
        }
        get_config_value = config_values.get
    elif isinstance(time_loss_diagnostic, Mapping) or hasattr(
        time_loss_diagnostic,
        "get",
    ):
        get_config_value = time_loss_diagnostic.get
    else:
        raise TypeError(
            "time_loss_diagnostic must be None, a mapping, or "
            "TimeLossDiagnosticConfig; "
            f"got {type(time_loss_diagnostic).__name__}"
        )

    enabled = _parse_diagnostic_bool(
        get_config_value("enabled", False),
        "time_loss_diagnostic.enabled",
    )
    split = str(get_config_value("split", "val")).strip().lower()
    num_bins = _parse_diagnostic_int(
        get_config_value("num_bins", 20),
        "time_loss_diagnostic.num_bins",
    )
    num_batches = _parse_diagnostic_int(
        get_config_value("num_batches", 0),
        "time_loss_diagnostic.num_batches",
    )
    log_heatmap = _parse_diagnostic_bool(
        get_config_value("log_heatmap", True),
        "time_loss_diagnostic.log_heatmap",
    )
    config = TimeLossDiagnosticConfig(
        enabled=enabled,
        split=split,
        num_bins=num_bins,
        num_batches=num_batches,
        log_heatmap=log_heatmap,
    )

    if config.split not in {"val", "train", "both"}:
        raise ValueError(
            "time_loss_diagnostic.split must be one of 'val', 'train', or "
            f"'both'; got {config.split!r}"
        )
    if config.num_bins < 1:
        raise ValueError(
            f"time_loss_diagnostic.num_bins must be >= 1, got {config.num_bins}"
        )
    if config.num_batches < 0:
        raise ValueError(
            "time_loss_diagnostic.num_batches must be >= 0, "
            f"got {config.num_batches}"
        )
    return config


def _time_loss_diagnostic_splits(config: TimeLossDiagnosticConfig) -> tuple[str, ...]:
    """Return selected split names for a resolved diagnostic config.

    Args:
        config: Resolved diagnostic configuration.

    Returns:
        Tuple of split names to evaluate.
    """
    if config.split == "both":
        return ("val", "train")
    return (config.split,)


def make_train_state(model, optimizer: optax.GradientTransformation) -> TrainState:
    """Initialise training state from a model and an Optax optimizer."""
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return TrainState(model=model, opt_state=opt_state)


def make_train_step(
    optimizer: optax.GradientTransformation,
    loss_fn: callable,
    data_parallel: DataParallelConfig | None = None,
):
    """Return a JIT-compiled train step closed over the optimizer and loss function.

    The optimizer and loss_fn are static Python objects — closing over them
    avoids passing them as traced arguments to filter_jit.

    Args:
        optimizer: Optax GradientTransformation.
        loss_fn:   Differentiable loss callable with signature
                   ``(model, x_t, u_t, t, cond, cond_mask, key) -> scalar``.
        data_parallel: Optional data-parallel runtime configuration.
    """
    data_parallel = resolve_data_parallel_config(data_parallel)

    @eqx.filter_jit(donate="all")
    def train_step(
        state: TrainState,
        x_t: jax.Array,
        u_t: jax.Array,
        t: jax.Array,
        cond: jax.Array,
        cond_mask: jax.Array,
        key: jax.Array,
    ) -> tuple[TrainState, jax.Array]:
        if data_parallel.enabled:
            state = eqx.filter_shard(state, data_parallel.model_sharding)
            x_t, u_t, t, cond, cond_mask, key = eqx.filter_shard(
                (x_t, u_t, t, cond, cond_mask, key),
                data_parallel.data_sharding,
            )
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            state.model, x_t, u_t, t, cond, cond_mask, key
        )
        updates, new_opt_state = optimizer.update(
            grads, state.opt_state, eqx.filter(state.model, eqx.is_array)
        )
        new_model = eqx.apply_updates(state.model, updates)
        new_state = TrainState(model=new_model, opt_state=new_opt_state)
        if data_parallel.enabled:
            new_state = eqx.filter_shard(new_state, data_parallel.model_sharding)
        return new_state, loss

    return train_step


@eqx.filter_jit
def ema_update(ema_model, new_model, decay: float):
    """Update EMA model weights using exponential moving average.

    Non-array leaves (static model configuration) are carried over from
    ``ema_model`` unchanged.

    Args:
        ema_model: Current EMA model.
        new_model: Latest trained model whose weights are blended in.
        decay:     EMA decay rate. Typical value: 0.9999.

    Returns:
        Updated EMA model with blended array leaves.
    """
    ema_arrays, static = eqx.partition(ema_model, eqx.is_array)
    new_arrays, _ = eqx.partition(new_model, eqx.is_array)
    updated = jax.tree_util.tree_map(
        lambda e, m: decay * e + (1.0 - decay) * m,
        ema_arrays,
        new_arrays,
    )
    return eqx.combine(updated, static)


def _copy_array_tree(tree):
    """Return a tree with array leaves copied and static leaves preserved."""
    return jax.tree_util.tree_map(
        lambda x: jnp.copy(x) if eqx.is_array(x) else x,
        tree,
    )


def prepare_batch(batch) -> tuple[np.ndarray, np.ndarray]:
    """Convert a PyTorch dataloader batch to numpy arrays.

    Args:
        batch: Tuple of (images, meta) from the dataloader.

    Returns:
        Tuple of (images_np, cond_np) as numpy arrays.
    """
    images, meta = batch
    return images.numpy(), meta.numpy()


def make_prepare_batch_jax(
    coupling: callable,
    time_sampler: callable,
    path_sampler: callable,
    p_uncond: float,
):
    """Return a JIT-compiled batch preparation function.

    Closes over coupling, time_sampler, path_sampler, and p_uncond as static
    Python objects, following the ``make_train_step`` closure pattern.

    Args:
        coupling:     Callable ``(x0, x1) -> x0_paired``. Must be JIT-compatible
                      (e.g. ``independent_coupling``). ``ot_coupling`` is rejected.
        time_sampler: Callable ``(key, batch_size) -> t``.
        path_sampler: Callable ``(x0, x1, t, *, key) -> (x_t, u_t)``.
        p_uncond:     Probability of dropping the condition per sample.

    Returns:
        A ``jax.jit``-compiled callable with signature
        ``(images_np, cond_np, key) -> (t, x_t, u_t, cond, cond_mask, dropout_keys)``.

    Raises:
        ValueError: If ``coupling`` is ``ot_coupling`` (not JIT-compatible).
    """
    from msdflow.flow.coupling import ot_coupling

    if coupling is ot_coupling:
        raise ValueError(
            "ot_coupling requires scipy and is not JIT-able. "
            "Use independent_coupling or implement a JAX-native OT coupling."
        )

    @jax.jit
    def prepare_batch_jax(
        images_np: np.ndarray,
        cond_np: np.ndarray,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        key_noise, key_time, key_path, key_cfg, key_dropout = jax.random.split(key, 5)
        B = images_np.shape[0]

        x1 = jnp.asarray(images_np)
        cond = jnp.asarray(cond_np)

        x0 = jax.random.normal(key_noise, x1.shape)
        x0 = coupling(x0, x1)
        t = time_sampler(key_time, B)
        cond_mask = jax.random.bernoulli(key_cfg, 1.0 - p_uncond, shape=(B,))
        x_t, u_t = path_sampler(x0, x1, t, key=key_path)
        dropout_keys = jax.random.split(key_dropout, B)

        return t, x_t, u_t, cond, cond_mask, dropout_keys

    return prepare_batch_jax


class BatchPrefetcher:
    """Threaded prefetcher that converts PyTorch batches to numpy in the background.

    Runs a daemon thread that pulls batches from the DataLoader, converts
    them to numpy arrays via ``prepare_batch``, and pushes them into a
    bounded queue for the main thread to consume.

    Args:
        dataloader: PyTorch DataLoader or list of batches.
        num_items: Number of prepared batches to produce.
        buffer_size: Maximum number of prepared batches to buffer. Defaults to 3.
    """

    def __init__(
        self,
        dataloader,
        num_items: int,
        buffer_size: int = 3,
    ):
        self._dataloader = dataloader
        self._num_items = num_items
        self._queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._fill, daemon=True)
        self._thread.start()

    def _fill(self):
        """Background thread: iterate dataloader, convert to numpy, enqueue."""
        data_iter = iter(self._dataloader)
        for i in range(self._num_items):
            if self._stop_event.is_set():
                return
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self._dataloader)
                batch = next(data_iter)
            result = prepare_batch(batch)
            if self._stop_event.is_set():
                return
            while not self._stop_event.is_set():
                try:
                    self._queue.put(result, timeout=1.0)
                    break
                except queue.Full:
                    continue
        if not self._stop_event.is_set():
            while not self._stop_event.is_set():
                try:
                    self._queue.put(None, timeout=1.0)  # sentinel
                    break
                except queue.Full:
                    continue

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is None:
            raise StopIteration
        return item

    def shutdown(self):
        """Signal the background thread to stop and wait for it to exit."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, *exc):
        """Exit context manager, ensuring thread cleanup."""
        self.shutdown()


def make_batch_metric_step(
    batch_metrics: list,
    data_parallel: DataParallelConfig | None = None,
):
    """Return a JIT-compiled step that evaluates a list of batch metrics.

    Args:
        batch_metrics: List of callables, each with signature
                       ``(model, x_t, u_t, t, cond, cond_mask, key) -> scalar``.
        data_parallel: Optional data-parallel runtime configuration.

    Returns:
        A ``filter_jit``-compiled callable with signature
        ``(model, x_t, u_t, t, cond, cond_mask) -> dict[str, jax.Array]``,
        keyed by the underlying function name for each metric.
    """
    data_parallel = resolve_data_parallel_config(data_parallel)

    names = [
        fn.func.__name__ if isinstance(fn, functools.partial) else fn.__name__
        for fn in batch_metrics
    ]

    if len(names) != len(set(names)):
        duplicates = list(set([n for n in names if names.count(n) > 1]))
        raise ValueError(
            f"make_batch_metric_step: duplicate metric names {duplicates}. "
            "Each metric must have a unique __name__."
        )

    @eqx.filter_jit(donate="all-except-first")
    def batch_metric_step(
        model,
        x_t: jax.Array,
        u_t: jax.Array,
        t: jax.Array,
        cond: jax.Array,
        cond_mask: jax.Array,
        key: jax.Array,
    ) -> dict:
        if data_parallel.enabled:
            model = eqx.filter_shard(model, data_parallel.model_sharding)
            x_t, u_t, t, cond, cond_mask, key = eqx.filter_shard(
                (x_t, u_t, t, cond, cond_mask, key),
                data_parallel.data_sharding,
            )
        return {
            name: fn(model, x_t, u_t, t, cond, cond_mask, key)
            for name, fn in zip(names, batch_metrics)
        }

    return batch_metric_step


def batch_metric_loop(
    key: jax.Array,
    ema_model,
    dataloader,
    step_fn: callable,
    prepare_jax: callable,
    num_batches: int = 0,
    data_parallel: DataParallelConfig | None = None,
) -> dict:
    """Stream a dataloader through a batch metric step and return per-metric means.

    Args:
        key:          JAX PRNG key (consumed internally via splitting).
        ema_model:    EMA model used for inference.
        dataloader:   Iterable of ``(images, meta)`` batches (PyTorch tensors).
        step_fn:      JIT-compiled step function from ``make_batch_metric_step()``.
        prepare_jax:  JIT-compiled batch prep from ``make_prepare_batch_jax()``.
        num_batches:  Maximum number of batches to process. ``0`` processes the
                      full dataloader.
        data_parallel: Optional data-parallel runtime configuration.

    Returns:
        ``dict[str, float]`` mapping metric name to its mean value over all
        processed batches. Batches are weighted equally regardless of size;
        use uniform batch sizes for unbiased estimates. Returns an empty dict
        if the dataloader yields no batches.
    """
    data_parallel = resolve_data_parallel_config(data_parallel)
    ema_model = shard_model(ema_model, data_parallel)

    totals: dict = {}
    n_batches = 0
    total = num_batches if num_batches > 0 else len(dataloader)
    data_iter = iter(dataloader)

    for _ in tqdm(range(total), desc="Batch metrics", leave=False, dynamic_ncols=True):
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        batch_key, key = jax.random.split(key, 2)
        images_np, cond_np = prepare_batch(batch)
        t, x_t, u_t, cond, cond_mask, dropout_keys = prepare_jax(
            images_np, cond_np, batch_key
        )
        x_t, u_t, t, cond, cond_mask, dropout_keys = shard_batch(
            (x_t, u_t, t, cond, cond_mask, dropout_keys),
            data_parallel,
        )
        results = step_fn(ema_model, x_t, u_t, t, cond, cond_mask, dropout_keys)
        for k, v in results.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n_batches += 1

    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in totals.items()}


def time_binned_loss_loop(
    key: jax.Array,
    model,
    dataloader,
    step_fn: callable,
    prepare_jax: callable,
    num_bins: int,
    num_batches: int = 0,
    data_parallel: DataParallelConfig | None = None,
) -> TimeBinnedLossResult:
    """Stream a dataloader through the time-binned loss diagnostic step.

    Args:
        key: JAX PRNG key consumed internally via splitting.
        model: Model used for inference.
        dataloader: Iterable of ``(images, meta)`` batches.
        step_fn: JIT-compiled step from ``make_time_binned_loss_step()``.
        prepare_jax: JIT-compiled batch preparation function.
        num_bins: Number of equal-width bins over ``[0, 1]``.
        num_batches: Maximum number of batches; ``0`` processes the full
            dataloader.
        data_parallel: Optional data-parallel runtime configuration.

    Returns:
        Host-side result containing bin edges, loss sums, counts, and means.
    """
    data_parallel = resolve_data_parallel_config(data_parallel)
    model = shard_model(model, data_parallel)
    result = TimeBinnedLossResult.empty(num_bins)

    total = num_batches if num_batches > 0 else len(dataloader)
    data_iter = iter(dataloader)
    for _ in tqdm(
        range(total),
        desc="Time-binned loss",
        leave=False,
        dynamic_ncols=True,
    ):
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        batch_key, key = jax.random.split(key, 2)
        images_np, cond_np = prepare_batch(batch)
        t, x_t, u_t, cond, cond_mask, dropout_keys = prepare_jax(
            images_np,
            cond_np,
            batch_key,
        )
        x_t, u_t, t, cond, cond_mask, dropout_keys = shard_batch(
            (x_t, u_t, t, cond, cond_mask, dropout_keys),
            data_parallel,
        )
        loss_sums, counts = step_fn(
            model,
            x_t,
            u_t,
            t,
            cond,
            cond_mask,
            dropout_keys,
        )
        result.add_batch(
            loss_sums=np.asarray(loss_sums),
            counts=np.asarray(counts),
        )

    return result


def _call_epoch_metric(
    metric: callable,
    model,
    val_dataloader,
    key: jax.Array,
    data_parallel: DataParallelConfig,
):
    """Call an epoch metric with optional data-parallel context.

    Args:
        metric: Epoch metric callable.
        model: EMA model passed to the metric.
        val_dataloader: Validation dataloader passed to the metric.
        key: JAX PRNG key passed to the metric.
        data_parallel: Resolved trainer data-parallel configuration.

    Returns:
        Metric result.
    """
    try:
        signature = inspect.signature(metric)
    except (TypeError, ValueError):
        return metric(model, val_dataloader, key)

    parameters = tuple(signature.parameters.values())
    has_positional_only_data_parallel = (
        len(parameters) >= 4
        and parameters[3].name == "data_parallel"
        and parameters[3].kind is inspect.Parameter.POSITIONAL_ONLY
    )
    accepts_data_parallel_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == "data_parallel"
            and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
        )
        for parameter in parameters
    )
    accepts_data_parallel_positional = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    )
    if has_positional_only_data_parallel:
        return metric(model, val_dataloader, key, data_parallel)
    if accepts_data_parallel_keyword:
        return metric(model, val_dataloader, key, data_parallel=data_parallel)
    if accepts_data_parallel_positional:
        return metric(model, val_dataloader, key, data_parallel)
    return metric(model, val_dataloader, key)


def train(
    key,
    model,
    dataloader,
    val_dataloader,
    optimizer: optax.GradientTransformation,
    loss_fn: callable,
    batch_metrics: list,
    epoch_metrics: list,
    coupling: callable,
    time_sampler: callable,
    path_sampler: callable,
    num_epochs: int,
    num_steps_per_epoch: int,
    p_uncond: float,
    ema_decay: float,
    log_every: int,
    val_every: int,
    checkpoint_every: int,
    checkpoint_dir: str,
    num_train_eval_batches: int = 0,
    clearml_task: Any = None,
    sample_fn=None,
    sample_every: int = 0,
    num_samples: int = 4,
    samples_share_clim: bool = False,
    samples_plot_method: str = "arcsinh",
    samples_arcsinh_percentile: float = 10.0,
    samples_dir: str | None = None,
    monitor: str = "flow_matching_loss",
    monitor_mode: str = "min",
    early_stopping_patience: int | None = None,
    grad_accum_steps: int = 1,
    buffer_size: int = 4,
    data_parallel: Any = None,
    checkpoint_hash: str | None = None,
    hash_payload: dict | None = None,
    latest_filename: str = "latest.json",
    save_on_sigterm: bool = True,
    resume_checkpoint: TrainingCheckpoint | None = None,
    resume_checkpoint_path: str | None = None,
    resume_metadata: dict | None = None,
    source_checkpoint_path: str | None = None,
    sigterm_flag_factory=SigtermFlag,
    *args,
    time_loss_diagnostic: Any = None,
    **kwargs,
):
    """Main training loop with EMA and periodic validation.

    Args:
        key:                    JAX PRNG key.
        model:                  Velocity-field network to train.
        dataloader:             PyTorch DataLoader yielding ``(images, meta)`` tuples
                                where images is ``(B, C, H, W)`` and meta is
                                ``(B, cond_dim)`` or ``(B, 0)`` if unconditional.
        val_dataloader:         DataLoader for the validation split.
        optimizer:              Optax GradientTransformation.
        loss_fn:                Differentiable loss callable
                                ``(model, x_t, u_t, t, cond, cond_mask, key) -> scalar``.
                                Drives gradient computation.
        batch_metrics:          List of callables with signature
                                ``(model, x_t, u_t, t, cond, cond_mask, key) -> scalar``.
                                Evaluated every ``val_every`` epochs over the full
                                val loader and ``num_train_eval_batches`` train batches.
        epoch_metrics:          List of callables with signature
                                ``(model, val_dataloader, key) -> scalar | dict``.
                                Evaluated every ``val_every`` epochs. Receives
                                the val dataloader iterable directly. Metrics
                                that accept a ``data_parallel`` keyword receive
                                the resolved DataParallelConfig.
                                If a metric returns a dict, its entries are
                                merged into epoch_metric_results directly.
        coupling:               Callable ``(x0_np, x1_np) -> x0_paired``.
        time_sampler:           Callable ``(key, batch_size) -> t``.
        path_sampler:           Callable ``(x0, x1, t, *, key) -> (x_t, u_t)``.
        num_epochs:             Total number of training epochs.
        num_steps_per_epoch:    Steps per epoch. ``0`` = full dataloader.
        p_uncond:               Probability of dropping the condition per sample.
        ema_decay:              EMA decay rate (typical: 0.9999).
        log_every:              Log metrics every this many epochs.
        val_every:              Run validation every this many epochs.
        checkpoint_every:       Save checkpoints every this many epochs.
        checkpoint_dir:         Directory for checkpoint files.
        num_train_eval_batches: Batches from train loader for batch metrics.
                                ``0`` = all.
        clearml_task:           ClearML Task for experiment tracking, or None
                                to disable all tracking (default).
        sample_fn:              Callable ``(model, key, num_samples) ->
                                np.ndarray (N, C, H, W)``, or None to skip
                                sample generation.
        sample_every:           Generate samples every this many epochs.
                                0 disables sample generation.
        num_samples:            Number of images per sampling event.
        samples_dir:            Root directory for saving sample .npy files.
        monitor (str): Name of the metric to monitor for best-model checkpointing
            and early stopping. Looked up in val_metrics first, then
            epoch_metric_results. Defaults to "flow_matching_loss".
        monitor_mode (str): "min" if lower metric values are better, "max" if
            higher values are better. Defaults to "min".
        early_stopping_patience (int | None): Number of consecutive validation
            cycles without improvement before training is halted. None disables
            early stopping. Defaults to None.
        grad_accum_steps (int): Number of gradient accumulation steps. When > 1,
            the optimizer is wrapped with ``optax.MultiSteps`` so that gradients
            are accumulated across that many mini-batches before an update is
            applied. Must be >= 1. Defaults to 1 (no accumulation).
        buffer_size: Number of prepared batches to prefetch.
        data_parallel: None, mapping, or DataParallelConfig controlling optional
            local data-parallel sharding.
        checkpoint_hash: Stable checkpoint compatibility hash used for full-state
            checkpoint saves and resume validation. Defaults to None.
        hash_payload: JSON-safe payload used to compute ``checkpoint_hash``.
            Defaults to None.
        latest_filename: Relative filename for the latest checkpoint pointer.
            Defaults to "latest.json".
        save_on_sigterm: Whether to install a SIGTERM handler that saves a
            full-state checkpoint before returning. Defaults to True.
        resume_checkpoint: Optional in-memory full-state checkpoint to restore.
            Defaults to None.
        resume_checkpoint_path: Optional serialized full-state checkpoint path to
            load before training. Defaults to None.
        resume_metadata: Optional metadata for resume validation and checkpoint
            example-tree construction. Defaults to None.
        source_checkpoint_path: Optional original checkpoint path recorded in new
            full-state checkpoint metadata. Defaults to None.
        sigterm_flag_factory: Context manager factory used to observe SIGTERM
            requests. Defaults to ``SigtermFlag``.
        time_loss_diagnostic: Keyword-only optional mapping or
            TimeLossDiagnosticConfig for logging flow-matching loss binned by
            sampled time ``t``. Defaults to None, which disables the diagnostic
            for direct train() calls.

    Returns:
        Trained EMA model when initialized, otherwise the live model.
    """
    if sample_fn is not None and sample_every > 0:
        if samples_dir is None and clearml_task is None:
            raise ValueError(
                "samples_dir must be provided when sample_fn and sample_every > 0 are set and clearml_task is None."
            )

        single_sample_fn = sample_fn
        batched_sample_fn = eqx.filter_jit(
            eqx.filter_vmap(
                lambda model, key: single_sample_fn(model=model, key=key),
                in_axes=(None, 0),
            )
        )

        ref_images = []
        for images_np, _ in val_dataloader:
            ref_images.append(np.asarray(images_np))
            if sum(len(b) for b in ref_images) >= num_samples:
                break
        ref_images = np.concatenate(ref_images, axis=0)[:num_samples].squeeze()
        if clearml_task is None:
            ref_samples_dir = os.path.join(samples_dir, f"reference")
            os.makedirs(ref_samples_dir, exist_ok=True)
            for i, img in enumerate(ref_images):
                np.save(os.path.join(ref_samples_dir, f"sample_{i:03d}.npy"), img)
        else:
            log_samples(
                task=clearml_task,
                images=ref_images,
                epoch=1,
                title="Reference Samples",
                share_clim=samples_share_clim,
                plot_method=samples_plot_method,
                arcsinh_percentile=samples_arcsinh_percentile,
            )

    if monitor_mode not in ("min", "max"):
        raise ValueError(f"monitor_mode must be 'min' or 'max', got {monitor_mode!r}")

    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

    if grad_accum_steps > 1:
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=grad_accum_steps)

    data_parallel_config = resolve_data_parallel_config(data_parallel)
    time_loss_config = resolve_time_loss_diagnostic_config(time_loss_diagnostic)

    state = make_train_state(model, optimizer)
    state = shard_train_state(state, data_parallel_config)

    train_step = make_train_step(
        optimizer,
        loss_fn,
        data_parallel=data_parallel_config,
    )
    batch_metric_step = make_batch_metric_step(
        batch_metrics,
        data_parallel=data_parallel_config,
    )
    time_binned_loss_step = None
    if time_loss_config.enabled:
        time_binned_loss_step = make_time_binned_loss_step(
            time_loss_config.num_bins,
            data_parallel=data_parallel_config,
        )
    prepare_jax = make_prepare_batch_jax(coupling, time_sampler, path_sampler, p_uncond)

    if num_steps_per_epoch == 0:
        microsteps_per_epoch = (len(dataloader) // grad_accum_steps) * grad_accum_steps
        steps_per_epoch = microsteps_per_epoch // grad_accum_steps
        if len(dataloader) % grad_accum_steps != 0:
            logger.warning(
                f"Dataloader length ({len(dataloader)}) is not divisible by "
                f"grad_accum_steps ({grad_accum_steps}). Dropping last "
                f"{len(dataloader) - microsteps_per_epoch} batches per epoch."
            )
    else:
        steps_per_epoch = num_steps_per_epoch
        microsteps_per_epoch = steps_per_epoch * grad_accum_steps

    total_epoch_time = 0.0
    avg_epoch_time = 0.0

    total_train_time = 0.0
    avg_train_time = 0.0

    total_val_time = 0.0
    avg_val_time = np.nan
    val_time = np.nan
    val_runs = 0

    val_metrics: dict = {}
    train_metrics: dict = {}
    epoch_metric_results: dict = {}
    time_loss_histories: dict[str, TimeBinnedLossHistory] = {}

    best_metric_value = float("inf") if monitor_mode == "min" else float("-inf")
    best_epoch = None
    patience_counter = 0

    ema_model = None
    ema_initialized = False

    sampling_key, key = jax.random.split(key)
    start_epoch = 0
    resume_completed_microsteps = 0
    resume_epoch_loss = 0.0

    if resume_checkpoint_path is not None and resume_metadata is None:
        resume_metadata_path = os.path.splitext(os.fspath(resume_checkpoint_path))[0]
        resume_metadata_path = f"{resume_metadata_path}.json"
        if not os.path.exists(resume_metadata_path):
            raise ValueError(
                "resume_metadata is required when resume_checkpoint_path has no "
                f"JSON sidecar: {resume_metadata_path}"
            )
        resume_metadata = load_json(resume_metadata_path)
        resume_metadata.setdefault("metadata_path", resume_metadata_path)

    if resume_checkpoint_path is not None:
        resume_ema_initialized = bool(
            resume_metadata and resume_metadata.get("ema_initialized")
        )
        like_checkpoint = TrainingCheckpoint(
            state=state,
            ema_model=model if resume_ema_initialized else None,
            ema_initialized=resume_ema_initialized,
            key=key,
            sampling_key=sampling_key,
            epoch=0,
            completed_microsteps=0,
            epoch_loss=0.0,
            best_metric_value=best_metric_value,
            best_epoch=None,
            patience_counter=0,
            total_epoch_time=0.0,
            total_train_time=0.0,
            total_val_time=0.0,
            val_runs=0,
            val_time=float("nan"),
            val_metrics={},
            train_metrics={},
            epoch_metric_results={},
        )
        resume_checkpoint = load_training_checkpoint(
            resume_checkpoint_path,
            like_checkpoint,
        )

    if resume_checkpoint is not None:
        if resume_metadata is not None and checkpoint_hash is not None:
            validate_checkpoint_metadata(
                resume_metadata,
                stable_hash=checkpoint_hash,
                monitor=monitor,
                monitor_mode=monitor_mode,
                microsteps_per_epoch=microsteps_per_epoch,
                allow_hash_override=False,
            )
        state = shard_train_state(resume_checkpoint.state, data_parallel_config)
        ema_model = resume_checkpoint.ema_model
        if ema_model is not None:
            ema_model = _copy_array_tree(ema_model)
            ema_model = shard_model(ema_model, data_parallel_config)
        ema_initialized = bool(resume_checkpoint.ema_initialized)
        key = resume_checkpoint.key
        sampling_key = resume_checkpoint.sampling_key
        start_epoch = int(resume_checkpoint.epoch)
        resume_completed_microsteps = int(resume_checkpoint.completed_microsteps)
        resume_epoch_loss = float(resume_checkpoint.epoch_loss)
        best_metric_value = float(resume_checkpoint.best_metric_value)
        best_epoch = resume_checkpoint.best_epoch
        patience_counter = int(resume_checkpoint.patience_counter)
        total_epoch_time = float(resume_checkpoint.total_epoch_time)
        total_train_time = float(resume_checkpoint.total_train_time)
        total_val_time = float(resume_checkpoint.total_val_time)
        val_runs = int(resume_checkpoint.val_runs)
        val_time = float(resume_checkpoint.val_time)
        val_metrics = dict(resume_checkpoint.val_metrics)
        train_metrics = dict(resume_checkpoint.train_metrics)
        epoch_metric_results = dict(resume_checkpoint.epoch_metric_results)
        if resume_completed_microsteps > 0:
            logger.warning(
                "Resuming epoch %s from %s completed microsteps; the dataloader "
                "prefix will be replayed with restored model state.",
                start_epoch + 1,
                resume_completed_microsteps,
            )

    def _clearml_task_id() -> str | None:
        """Return the active ClearML task id when one is available."""
        if clearml_task is None:
            return None
        return getattr(clearml_task, "id", None)

    def _float_metrics(metrics: dict) -> dict[str, float]:
        """Return a JSON-friendly float copy of a metric dictionary."""
        return {str(k): float(v) for k, v in metrics.items()}

    def _make_checkpoint(
        *,
        epoch_to_resume: int,
        completed_microsteps: int,
        current_epoch_loss: float,
    ) -> TrainingCheckpoint:
        """Build a full-state checkpoint from current trainer state."""
        return TrainingCheckpoint(
            state=state,
            ema_model=ema_model,
            ema_initialized=ema_initialized,
            key=key,
            sampling_key=sampling_key,
            epoch=int(epoch_to_resume),
            completed_microsteps=int(completed_microsteps),
            epoch_loss=float(current_epoch_loss),
            best_metric_value=float(best_metric_value),
            best_epoch=best_epoch,
            patience_counter=int(patience_counter),
            total_epoch_time=float(total_epoch_time),
            total_train_time=float(total_train_time),
            total_val_time=float(total_val_time),
            val_runs=int(val_runs),
            val_time=float(val_time),
            val_metrics=_float_metrics(val_metrics),
            train_metrics=_float_metrics(train_metrics),
            epoch_metric_results=_float_metrics(epoch_metric_results),
        )

    def _save_full_checkpoint(
        *,
        kind: str,
        epoch_to_resume: int,
        completed_microsteps: int,
        current_epoch_loss: float,
    ) -> dict:
        """Save the current full-state checkpoint and latest pointer."""
        if checkpoint_hash is None:
            raise ValueError("checkpoint_hash is required for full-state checkpoints.")
        checkpoint = _make_checkpoint(
            epoch_to_resume=epoch_to_resume,
            completed_microsteps=completed_microsteps,
            current_epoch_loss=current_epoch_loss,
        )
        metadata = save_training_checkpoint(
            run_dir=checkpoint_dir,
            checkpoint=checkpoint,
            stable_hash=checkpoint_hash,
            checkpoint_kind=kind,
            grad_accum_steps=grad_accum_steps,
            microsteps_per_epoch=microsteps_per_epoch,
            monitor=monitor,
            monitor_mode=monitor_mode,
            clearml_task_id=_clearml_task_id(),
            latest_filename=latest_filename,
            source_checkpoint_path=source_checkpoint_path or resume_checkpoint_path,
            hash_payload=hash_payload or {},
        )
        logger.info("Saved full-state %s checkpoint: %s", kind, metadata["payload_path"])
        return metadata

    with sigterm_flag_factory(
        save_on_sigterm and checkpoint_hash is not None
    ) as sigterm_flag:
        for epoch in range(start_epoch, num_epochs):
            is_resumed_epoch = (
                epoch == start_epoch and resume_completed_microsteps > 0
            )
            epoch_loss = jnp.float32(
                resume_epoch_loss if is_resumed_epoch else 0.0
            )
            epoch_loss_denominator = (
                resume_completed_microsteps if is_resumed_epoch else 0
            )
            epoch_start_time = time.perf_counter()

            prefetcher = BatchPrefetcher(
                dataloader=dataloader,
                num_items=microsteps_per_epoch,
                buffer_size=buffer_size,
            )

            try:
                with logging_redirect_tqdm():
                    pbar = tqdm(
                        range(microsteps_per_epoch),
                        desc=f"Epoch {epoch + 1}/{num_epochs}",
                        leave=False,
                        dynamic_ncols=True,
                    )
                    for microstep in pbar:
                        images_np, cond_np = next(prefetcher)
                        step_key, key = jax.random.split(key)
                        t, x_t, u_t, cond, cond_mask, dropout_keys = prepare_jax(
                            images_np, cond_np, step_key
                        )
                        x_t, u_t, t, cond, cond_mask, dropout_keys = shard_batch(
                            (x_t, u_t, t, cond, cond_mask, dropout_keys),
                            data_parallel_config,
                        )

                        state, loss = train_step(
                            state, x_t, u_t, t, cond, cond_mask, dropout_keys
                        )
                        if (microstep + 1) % grad_accum_steps == 0:
                            if not ema_initialized:
                                ema_model = _copy_array_tree(state.model)
                                ema_initialized = True
                            else:
                                ema_model = ema_update(
                                    ema_model,
                                    state.model,
                                    ema_decay,
                                )
                        epoch_loss = epoch_loss + loss
                        epoch_loss_denominator += 1

                        if sigterm_flag.requested:
                            partial_time = time.perf_counter() - epoch_start_time
                            total_train_time += partial_time
                            total_epoch_time += partial_time
                            if microstep + 1 >= microsteps_per_epoch:
                                checkpoint_epoch = epoch + 1
                                checkpoint_microsteps = 0
                                checkpoint_epoch_loss = 0.0
                            else:
                                checkpoint_epoch = epoch
                                max_mid_epoch_microsteps = max(
                                    microsteps_per_epoch - 1,
                                    0,
                                )
                                checkpoint_microsteps = min(
                                    epoch_loss_denominator,
                                    max_mid_epoch_microsteps,
                                )
                                checkpoint_epoch_loss = float(epoch_loss)
                                if (
                                    checkpoint_microsteps > 0
                                    and checkpoint_microsteps < epoch_loss_denominator
                                ):
                                    checkpoint_epoch_loss *= (
                                        checkpoint_microsteps
                                        / epoch_loss_denominator
                                    )
                            _save_full_checkpoint(
                                kind="sigterm",
                                epoch_to_resume=checkpoint_epoch,
                                completed_microsteps=checkpoint_microsteps,
                                current_epoch_loss=checkpoint_epoch_loss,
                            )
                            return ema_model if ema_model is not None else state.model
            finally:
                prefetcher.shutdown()

            epoch_loss = float(epoch_loss)
            if ema_model is not None:
                ema_model = eqx.nn.inference_mode(ema_model, value=True)
            train_loss_denominator = max(epoch_loss_denominator, 1)
            train_time = time.perf_counter() - epoch_start_time
            total_train_time += train_time
            avg_train_time = total_train_time / (epoch + 1)
            if sigterm_flag.requested:
                total_epoch_time += train_time
                _save_full_checkpoint(
                    kind="sigterm",
                    epoch_to_resume=epoch + 1,
                    completed_microsteps=0,
                    current_epoch_loss=0.0,
                )
                return ema_model if ema_model is not None else state.model

            if (epoch + 1) % val_every == 0:
                val_start_time = time.perf_counter()

                key, key_val, key_train, key_epoch, key_time_loss = jax.random.split(
                    key,
                    5,
                )

                eval_model = ema_model if ema_model is not None else state.model
                val_metrics = batch_metric_loop(
                    key=key_val,
                    ema_model=eval_model,
                    dataloader=val_dataloader,
                    step_fn=batch_metric_step,
                    prepare_jax=prepare_jax,
                    num_batches=num_train_eval_batches,
                    data_parallel=data_parallel_config,
                )

                train_metrics = batch_metric_loop(
                    key=key_train,
                    ema_model=eval_model,
                    dataloader=dataloader,
                    step_fn=batch_metric_step,
                    prepare_jax=prepare_jax,
                    num_batches=num_train_eval_batches,
                    data_parallel=data_parallel_config,
                )

                epoch_metric_results = {}
                if epoch_metrics:
                    for fn in epoch_metrics:
                        result = _call_epoch_metric(
                            fn,
                            eval_model,
                            val_dataloader,
                            key_epoch,
                            data_parallel_config,
                        )
                        if isinstance(result, dict):
                            epoch_metric_results.update(result)
                        else:
                            if isinstance(fn, functools.partial):
                                name = fn.func.__name__
                            elif hasattr(fn, "__name__"):
                                name = fn.__name__
                            else:
                                name = type(fn).__name__
                            epoch_metric_results[name] = result

                if time_loss_config.enabled and time_binned_loss_step is not None:
                    split_dataloaders = {
                        "val": val_dataloader,
                        "train": dataloader,
                    }
                    for split in _time_loss_diagnostic_splits(time_loss_config):
                        key_time_loss, split_key = jax.random.split(key_time_loss, 2)
                        time_loss_result = time_binned_loss_loop(
                            key=split_key,
                            model=eval_model,
                            dataloader=split_dataloaders[split],
                            step_fn=time_binned_loss_step,
                            prepare_jax=prepare_jax,
                            num_bins=time_loss_config.num_bins,
                            num_batches=time_loss_config.num_batches,
                            data_parallel=data_parallel_config,
                        )
                        history = time_loss_histories.get(split)
                        if history is None:
                            history = TimeBinnedLossHistory(
                                bin_edges=time_loss_result.bin_edges,
                            )
                            time_loss_histories[split] = history
                        history.append(epoch + 1, time_loss_result)
                        log_time_binned_loss(
                            task=clearml_task,
                            split=split,
                            epoch=epoch + 1,
                            result=time_loss_result,
                            history=history if time_loss_config.log_heatmap else None,
                        )

                val_time = time.perf_counter() - val_start_time
                total_val_time += val_time
                val_runs += 1
                avg_val_time = total_val_time / val_runs

                # --- Best-model checkpointing and early stopping ---
                current_monitor = val_metrics.get(monitor)
                if current_monitor is None:
                    current_monitor = epoch_metric_results.get(monitor)
                if current_monitor is None and (val_metrics or epoch_metric_results):
                    raise ValueError(
                        f"monitor metric '{monitor}' not found in val_metrics "
                        f"{list(val_metrics.keys())} or epoch_metric_results "
                        f"{list(epoch_metric_results.keys())}"
                    )
                if current_monitor is not None:
                    current_monitor = float(current_monitor)
                    is_improved = (
                        current_monitor < best_metric_value
                        if monitor_mode == "min"
                        else current_monitor > best_metric_value
                    )
                    if is_improved:
                        all_metrics = {monitor: current_monitor}
                        all_metrics.update(
                            {k: v for k, v in val_metrics.items() if k != monitor}
                        )
                        all_metrics.update(
                            {
                                k: v
                                for k, v in epoch_metric_results.items()
                                if k != monitor
                            }
                        )
                        metric_str = " | ".join(
                            f"{k} = {float(v):.4g}" for k, v in all_metrics.items()
                        )
                        logger.info(
                            f"New best model at epoch {epoch + 1}: {metric_str}"
                        )
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        best_raw_path = os.path.join(
                            checkpoint_dir, f"model_epoch{epoch + 1}_best_raw.eqx"
                        )
                        best_ema_path = os.path.join(
                            checkpoint_dir, f"model_epoch{epoch + 1}_best_ema.eqx"
                        )
                        eqx.tree_serialise_leaves(best_raw_path, state.model)
                        if ema_model is not None:
                            eqx.tree_serialise_leaves(best_ema_path, ema_model)
                            log_checkpoint(clearml_task, best_ema_path, epoch + 1)
                        best_metric_value = current_monitor
                        best_epoch = epoch + 1
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if sigterm_flag.requested:
                        _save_full_checkpoint(
                            kind="sigterm",
                            epoch_to_resume=epoch + 1,
                            completed_microsteps=0,
                            current_epoch_loss=0.0,
                        )
                        return ema_model if ema_model is not None else state.model

                    if (
                        early_stopping_patience is not None
                        and patience_counter >= early_stopping_patience
                    ):
                        logger.info(
                            f"Early stopping triggered at epoch {epoch + 1}: "
                            f"'{monitor}' did not improve for "
                            f"{early_stopping_patience} consecutive validation cycles."
                        )
                        break

            if (epoch + 1) % checkpoint_every == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                raw_path = os.path.join(
                    checkpoint_dir,
                    f"model_epoch{epoch + 1}_raw.eqx",
                )
                ema_path = os.path.join(
                    checkpoint_dir,
                    f"model_epoch{epoch + 1}_ema.eqx",
                )
                eqx.tree_serialise_leaves(raw_path, state.model)
                if ema_model is not None:
                    eqx.tree_serialise_leaves(ema_path, ema_model)
                    logger.info(f"Saved checkpoint: {ema_path}")
                    log_checkpoint(clearml_task, ema_path, epoch + 1)
                if checkpoint_hash is not None:
                    _save_full_checkpoint(
                        kind="periodic",
                        epoch_to_resume=epoch + 1,
                        completed_microsteps=0,
                        current_epoch_loss=0.0,
                    )

            if (
                sample_fn is not None
                and sample_every > 0
                and (epoch + 1) % sample_every == 0
            ):

                sample_model = ema_model if ema_model is not None else state.model
                sample_keys = jax.random.split(sampling_key, num_samples)
                images = batched_sample_fn(sample_model, sample_keys)
                images = np.asarray(images).squeeze()
                if clearml_task is None:
                    epoch_samples_dir = os.path.join(samples_dir, f"epoch_{epoch + 1}")
                    os.makedirs(epoch_samples_dir, exist_ok=True)
                    for i, img in enumerate(images):
                        np.save(
                            os.path.join(epoch_samples_dir, f"sample_{i:03d}.npy"),
                            img,
                        )
                else:
                    log_samples(
                        task=clearml_task,
                        images=images,
                        epoch=epoch + 1,
                        title="Model Samples",
                        share_clim=samples_share_clim,
                        plot_method=samples_plot_method,
                        arcsinh_percentile=samples_arcsinh_percentile,
                    )

            epoch_time = time.perf_counter() - epoch_start_time
            total_epoch_time += epoch_time
            avg_epoch_time = total_epoch_time / (epoch + 1)

            if (epoch + 1) % log_every == 0:
                train_loss = epoch_loss / train_loss_denominator
                scalars = {
                    "train/loss": train_loss,
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"epoch/{k}": v for k, v in epoch_metric_results.items()},
                }
                log_metrics(clearml_task, scalars, epoch + 1)
                all_metrics = {
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"epoch/{k}": v for k, v in epoch_metric_results.items()},
                }
                metric_str = (
                    " | ".join(f"{k}: {v:.4g}" for k, v in all_metrics.items())
                    if all_metrics
                    else "no metrics yet"
                )
                val_time_str = (
                    f"Val Time: {val_time:.2f}s (avg {avg_val_time:.2f}s)"
                    if val_runs > 0
                    else "Val Time: pending"
                )
                log_string = (
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    + f"Train Loss: {train_loss:.4g} | "
                    + metric_str
                    + " | "
                    + f"Epoch Time: {epoch_time:.2g}s (avg {avg_epoch_time:.2g}s) | "
                    + f"Train Time: {train_time:.2g}s (avg {avg_train_time:.2g}s) | "
                    + val_time_str
                )
                logger.info(log_string)

            if sigterm_flag.requested:
                _save_full_checkpoint(
                    kind="sigterm",
                    epoch_to_resume=epoch + 1,
                    completed_microsteps=0,
                    current_epoch_loss=0.0,
                )
                return ema_model if ema_model is not None else state.model

    return ema_model if ema_model is not None else state.model
