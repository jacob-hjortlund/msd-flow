"""Training loop and state management for flow matching.

Provides ``TrainState``, a JIT-compiled train step factory, and
the main training loop with periodic checkpointing.
"""

from typing import Any

import time
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
from msdflow.tracking import log_metrics, log_checkpoint, log_samples

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


def make_train_state(model, optimizer: optax.GradientTransformation) -> TrainState:
    """Initialise training state from a model and an Optax optimizer."""
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return TrainState(model=model, opt_state=opt_state)


def make_train_step(optimizer: optax.GradientTransformation, loss_fn: callable):
    """Return a JIT-compiled train step closed over the optimizer and loss function.

    The optimizer and loss_fn are static Python objects — closing over them
    avoids passing them as traced arguments to filter_jit.

    Args:
        optimizer: Optax GradientTransformation.
        loss_fn:   Differentiable loss callable with signature
                   ``(model, x_t, u_t, t, cond, cond_mask, key) -> scalar``.
    """

    @eqx.filter_jit
    def train_step(
        state: TrainState,
        x_t: jax.Array,
        u_t: jax.Array,
        t: jax.Array,
        cond: jax.Array,
        cond_mask: jax.Array,
        key: jax.Array,
    ) -> tuple[TrainState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            state.model, x_t, u_t, t, cond, cond_mask, key
        )
        updates, new_opt_state = optimizer.update(
            grads, state.opt_state, eqx.filter(state.model, eqx.is_array)
        )
        new_model = eqx.apply_updates(state.model, updates)
        return TrainState(model=new_model, opt_state=new_opt_state), loss

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


def prepare_batch(
    batch,
    key,
    coupling: callable,
    time_sampler: callable,
    path_sampler: callable,
    p_uncond: float,
):
    """Prepare a batch by sampling time steps, applying coupling, and masking conditions.

    Args:
        batch:          Tuple of (images, meta) from the dataloader.
        key:            JAX PRNG key for random operations.
        coupling:       Callable that takes (x0_np, x1_np) and returns coupled x0.
        time_sampler:   Callable for sampling time steps.
        path_sampler:   JIT-compiled ``sample_path`` partial.
        p_uncond:       Probability of dropping the condition per sample.

    Returns:
        Tuple of (t, x_t, u_t, cond, cond_mask, dropout_keys) ready for training step.
    """

    key_cpu, key_time, key_path, dropout_key = jax.random.split(key, 4)

    images, meta = batch
    x1_np = images.numpy()
    cond_np = meta.numpy()
    B = x1_np.shape[0]
    droupout_keys = jax.random.split(dropout_key, B)

    cpu_seed = int(jax.random.randint(key_cpu, shape=(), minval=0, maxval=2**31 - 1))
    rng = np.random.default_rng(cpu_seed)

    x0_np = rng.standard_normal(x1_np.shape).astype(np.float32)
    x0_paired = coupling(x0_np, x1_np)
    t = time_sampler(key_time, B)

    # CFG: randomly drop condition per sample with probability p_uncond
    cond_mask_np = (rng.random(B) >= p_uncond).astype(bool)

    x_t, u_t = path_sampler(jnp.array(x0_paired), jnp.array(x1_np), t, key=key_path)
    cond = jnp.array(cond_np)
    cond_mask = jnp.array(cond_mask_np)

    return t, x_t, u_t, cond, cond_mask, droupout_keys


def make_batch_metric_step(batch_metrics: list):
    """Return a JIT-compiled step that evaluates a list of batch metrics.

    Args:
        batch_metrics: List of callables, each with signature
                       ``(model, x_t, u_t, t, cond, cond_mask, key) -> scalar``.

    Returns:
        A ``filter_jit``-compiled callable with signature
        ``(model, x_t, u_t, t, cond, cond_mask) -> dict[str, jax.Array]``,
        keyed by the underlying function name for each metric.
    """

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

    @eqx.filter_jit
    def batch_metric_step(
        model,
        x_t: jax.Array,
        u_t: jax.Array,
        t: jax.Array,
        cond: jax.Array,
        cond_mask: jax.Array,
        key: jax.Array,
    ) -> dict:
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
    coupling: callable,
    time_sampler: callable,
    path_sampler: callable,
    p_uncond: float,
    num_batches: int = 0,
) -> dict:
    """Stream a dataloader through a batch metric step and return per-metric means.

    Args:
        key:          JAX PRNG key (consumed internally via splitting).
        ema_model:    EMA model used for inference.
        dataloader:   Iterable of ``(images, meta)`` batches.
        step_fn:      JIT-compiled step function from ``make_batch_metric_step()``.
        coupling:     Callable for batch coupling.
        time_sampler: Callable for sampling time steps.
        path_sampler: Callable for constructing the interpolant.
        p_uncond:     Probability of dropping the condition per sample.
        num_batches:  Maximum number of batches to process. ``0`` processes the
                      full dataloader.

    Returns:
        ``dict[str, float]`` mapping metric name to its mean value over all
        processed batches. Batches are weighted equally regardless of size;
        use uniform batch sizes for unbiased estimates. Returns an empty dict if the dataloader yields no batches.
    """
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
        t, x_t, u_t, cond, cond_mask, dropout_keys = prepare_batch(
            batch=batch,
            key=batch_key,
            coupling=coupling,
            time_sampler=time_sampler,
            path_sampler=path_sampler,
            p_uncond=p_uncond,
        )
        results = step_fn(ema_model, x_t, u_t, t, cond, cond_mask, dropout_keys)
        for k, v in results.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n_batches += 1

    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in totals.items()}


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
    samples_dir: str | None = None,
    monitor: str = "flow_matching_loss",
    monitor_mode: str = "min",
    early_stopping_patience: int | None = None,
    grad_accum_steps: int = 1,
    *args,
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
                                the val dataloader iterable directly. Extra
                                dependencies must be baked in via partial.
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

    Returns:
        Trained EMA model.
    """
    if sample_fn is not None and sample_every > 0 and samples_dir is None:
        raise ValueError(
            "samples_dir must be provided when sample_fn and sample_every > 0 are set"
        )

    if monitor_mode not in ("min", "max"):
        raise ValueError(f"monitor_mode must be 'min' or 'max', got {monitor_mode!r}")

    if grad_accum_steps < 1:
        raise ValueError(
            f"grad_accum_steps must be >= 1, got {grad_accum_steps}"
        )

    if grad_accum_steps > 1:
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=grad_accum_steps)

    state = make_train_state(model, optimizer)

    train_step = make_train_step(optimizer, loss_fn)
    batch_metric_step = make_batch_metric_step(batch_metrics)

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

    ema_model = model
    data_iter = iter(dataloader)

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

    best_metric_value = float("inf") if monitor_mode == "min" else float("-inf")
    patience_counter = 0
    best_epoch = None

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start_time = time.perf_counter()

        with logging_redirect_tqdm():
            pbar = tqdm(
                range(microsteps_per_epoch),
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=False,
                dynamic_ncols=True,
            )
            for microstep in pbar:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                batch_key, key = jax.random.split(key, 2)

                t, x_t, u_t, cond, cond_mask, dropout_keys = prepare_batch(
                    batch=batch,
                    key=batch_key,
                    coupling=coupling,
                    time_sampler=time_sampler,
                    path_sampler=path_sampler,
                    p_uncond=p_uncond,
                )

                state, loss = train_step(
                    state, x_t, u_t, t, cond, cond_mask, dropout_keys
                )
                if (microstep + 1) % grad_accum_steps == 0:
                    ema_model = ema_update(ema_model, state.model, ema_decay)
                epoch_loss += float(loss)

        ema_model = eqx.nn.inference_mode(ema_model, value=True)
        train_time = time.perf_counter() - epoch_start_time
        total_train_time += train_time
        avg_train_time = total_train_time / (epoch + 1)

        if (epoch + 1) % val_every == 0:
            val_start_time = time.perf_counter()

            key, key_val, key_train, key_epoch = jax.random.split(key, 4)

            val_metrics = batch_metric_loop(
                key=key_val,
                ema_model=ema_model,
                dataloader=val_dataloader,
                step_fn=batch_metric_step,
                coupling=coupling,
                time_sampler=time_sampler,
                path_sampler=path_sampler,
                p_uncond=p_uncond,
                num_batches=num_train_eval_batches,
            )

            train_metrics = batch_metric_loop(
                key=key_train,
                ema_model=ema_model,
                dataloader=dataloader,
                step_fn=batch_metric_step,
                coupling=coupling,
                time_sampler=time_sampler,
                path_sampler=path_sampler,
                p_uncond=p_uncond,
                num_batches=num_train_eval_batches,
            )

            epoch_metric_results = {}
            if epoch_metrics:
                for fn in epoch_metrics:
                    result = fn(ema_model, val_dataloader, key_epoch)
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
                        {k: v for k, v in epoch_metric_results.items() if k != monitor}
                    )
                    metric_str = " | ".join(
                        f"{k} = {float(v):.4g}" for k, v in all_metrics.items()
                    )
                    logger.info(f"New best model at epoch {epoch + 1}: {metric_str}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    best_raw_path = os.path.join(
                        checkpoint_dir, f"model_epoch{epoch + 1}_best_raw.eqx"
                    )
                    best_ema_path = os.path.join(
                        checkpoint_dir, f"model_epoch{epoch + 1}_best_ema.eqx"
                    )
                    eqx.tree_serialise_leaves(best_raw_path, state.model)
                    eqx.tree_serialise_leaves(best_ema_path, ema_model)
                    log_checkpoint(clearml_task, best_ema_path, epoch + 1)
                    best_metric_value = current_monitor
                    best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (
                    early_stopping_patience is not None
                    and patience_counter >= early_stopping_patience
                ):
                    logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}: '{monitor}' "
                        f"did not improve for {early_stopping_patience} consecutive "
                        f"validation cycles."
                    )
                    break

        if (epoch + 1) % checkpoint_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            raw_path = os.path.join(checkpoint_dir, f"model_epoch{epoch + 1}_raw.eqx")
            ema_path = os.path.join(checkpoint_dir, f"model_epoch{epoch + 1}_ema.eqx")
            eqx.tree_serialise_leaves(raw_path, state.model)
            eqx.tree_serialise_leaves(ema_path, ema_model)
            logger.info(f"Saved checkpoint: {ema_path}")
            log_checkpoint(clearml_task, ema_path, epoch + 1)

        if (
            sample_fn is not None
            and sample_every > 0
            and (epoch + 1) % sample_every == 0
        ):
            sample_key, key = jax.random.split(key)
            images = sample_fn(ema_model, sample_key, num_samples)
            epoch_samples_dir = os.path.join(samples_dir, f"epoch_{epoch + 1}")
            os.makedirs(epoch_samples_dir, exist_ok=True)
            for i, img in enumerate(images):
                np.save(os.path.join(epoch_samples_dir, f"sample_{i:03d}.npy"), img)
            log_samples(clearml_task, images, epoch + 1)

        epoch_time = time.perf_counter() - epoch_start_time
        total_epoch_time += epoch_time
        avg_epoch_time = total_epoch_time / (epoch + 1)

        if (epoch + 1) % log_every == 0:
            scalars = {
                "train/loss": epoch_loss / microsteps_per_epoch,
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
                + f"Train Loss: {epoch_loss / steps_per_epoch:.4g} | "
                + metric_str
                + " | "
                + f"Epoch Time: {epoch_time:.2g}s (avg {avg_epoch_time:.2g}s) | "
                + f"Train Time: {train_time:.2g}s (avg {avg_train_time:.2g}s) | "
                + val_time_str
            )
            logger.info(log_string)

    return ema_model
