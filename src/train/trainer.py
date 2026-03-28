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

import os
import logging
import numpy as np

from tqdm import tqdm
from src.flow.otfm import flow_matching_loss
from src.utils import register_all_resolvers
from tqdm.contrib.logging import logging_redirect_tqdm

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


def make_train_step(optimizer: optax.GradientTransformation):
    """Return a JIT-compiled train step closed over the optimizer.

    The optimizer is a static Python object — closing over it avoids
    passing it as a traced argument to filter_jit.
    """

    @eqx.filter_jit
    def train_step(
        state: TrainState,
        x_t: jax.Array,
        u_t: jax.Array,
        t: jax.Array,
        cond: jax.Array,
        cond_mask: jax.Array,
    ) -> tuple[TrainState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(flow_matching_loss)(
            state.model, x_t, u_t, t, cond, cond_mask
        )
        updates, new_opt_state = optimizer.update(
            grads, state.opt_state, eqx.filter(state.model, eqx.is_array)
        )
        new_model = eqx.apply_updates(state.model, updates)
        return TrainState(model=new_model, opt_state=new_opt_state), loss

    return train_step


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
        Tuple of (t, x_t, u_t, cond, cond_mask) ready for training step.
    """

    key_cpu, key_time, key_path = jax.random.split(key, 3)

    images, meta = batch
    x1_np = images.numpy()
    cond_np = meta.numpy()
    B = x1_np.shape[0]

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

    return t, x_t, u_t, cond, cond_mask


def make_val_step():
    """Return a JIT-compiled validation step.

    Computes flow matching loss without gradient. Mirrors the structure of
    ``make_train_step`` but does not update any state.

    Returns:
        A ``filter_jit``-compiled callable with signature
        ``(model, x_t, u_t, t, cond, cond_mask) -> scalar_loss``.
    """

    @eqx.filter_jit
    def val_step(
        model,
        x_t: jax.Array,
        u_t: jax.Array,
        t: jax.Array,
        cond: jax.Array,
        cond_mask: jax.Array,
    ) -> jax.Array:
        return flow_matching_loss(model, x_t, u_t, t, cond, cond_mask)

    return val_step


def validation_loop(
    key: jax.Array,
    ema_model,
    dataloader,
    step_fn: callable,
    coupling: callable,
    time_sampler: callable,
    path_sampler: callable,
    p_uncond: float,
) -> float:
    """Run a full pass over val_dataloader and return mean flow matching loss.

    Args:
        key:             JAX PRNG key (consumed internally via splitting).
        ema_model:       EMA model used for inference.
        dataloader:      Iterable of ``(images, meta)`` batches.
        step_fn:         JIT-compiled step function from ``make_val_step()``.
        coupling:        Callable for batch coupling.
        time_sampler:    Callable for sampling time steps.
        path_sampler:    Callable for constructing the interpolant.
        p_uncond:        Probability of dropping the condition per sample.

    Returns:
        Mean validation loss over all batches.
    """
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch_key, key = jax.random.split(key)
        t, x_t, u_t, cond, cond_mask = prepare_batch(
            batch=batch,
            key=batch_key,
            coupling=coupling,
            time_sampler=time_sampler,
            path_sampler=path_sampler,
            p_uncond=p_uncond,
        )
        total_loss += float(step_fn(ema_model, x_t, u_t, t, cond, cond_mask))
        n_batches += 1

    return total_loss / n_batches


def train(
    key,
    model,
    dataloader,
    val_dataloader,
    optimizer: optax.GradientTransformation,
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
):
    """Main training loop with EMA and periodic validation.

    Args:
        key:                JAX PRNG key.
        model:              Velocity-field network to train.
        dataloader:         PyTorch DataLoader yielding ``(images, meta)`` tuples
                            where images is ``(B, C, H, W)`` and meta is
                            ``(B, cond_dim)`` or ``(B, 0)`` if unconditional.
        val_dataloader:     DataLoader for the validation split, same format as
                            ``dataloader``. Used for periodic EMA model evaluation.
        optimizer:          Optax GradientTransformation.
        coupling:           Callable ``(x0_np, x1_np) -> x0_paired`` for batch
                            coupling (e.g. ``independent_coupling`` or
                            ``ot_coupling``).
        time_sampler:       Callable ``(key, batch_size) -> t`` for sampling
                            per-sample times.
        path_sampler:       Callable ``(x0, x1, t, *, key) -> (x_t, u_t)``
                            for constructing the interpolant.
        num_epochs:         Total number of training epochs.
        num_steps_per_epoch: Steps per epoch. Use ``0`` to consume the full
                            dataloader each epoch.
        p_uncond:           Probability of dropping the condition per sample
                            (classifier-free guidance training).
        ema_decay:          EMA decay rate (typical: 0.9999).
        log_every:          Log metrics every this many epochs.
        val_every:          Run validation every this many epochs.
        checkpoint_every:   Save checkpoints every this many epochs.
        checkpoint_dir:     Directory for checkpoint files.

    Returns:
        Trained EMA model.
    """
    state = make_train_state(model, optimizer)
    train_step = make_train_step(optimizer)
    val_step = make_val_step()

    steps_per_epoch = (
        len(dataloader) if num_steps_per_epoch == 0 else num_steps_per_epoch
    )

    ema_model = model
    data_iter = iter(dataloader)

    total_epoch_time = 0.0
    avg_epoch_time = 0.0

    total_train_time = 0.0
    avg_train_time = 0.0

    total_val_time = 0.0
    avg_val_time = np.nan
    val_loss = np.nan
    val_time = np.nan
    val_runs = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start_time = time.perf_counter()

        with logging_redirect_tqdm():
            pbar = tqdm(
                range(steps_per_epoch),
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=False,
                dynamic_ncols=True,
            )
            for _ in pbar:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                batch_key, key = jax.random.split(key)

                t, x_t, u_t, cond, cond_mask = prepare_batch(
                    batch=batch,
                    key=batch_key,
                    coupling=coupling,
                    time_sampler=time_sampler,
                    path_sampler=path_sampler,
                    p_uncond=p_uncond,
                )

                state, loss = train_step(state, x_t, u_t, t, cond, cond_mask)
                ema_model = ema_update(ema_model, state.model, ema_decay)
                epoch_loss += float(loss)

        train_time = time.perf_counter() - epoch_start_time
        total_train_time += train_time
        avg_train_time = total_train_time / (epoch + 1)

        if (epoch + 1) % val_every == 0:
            val_start_time = time.perf_counter()

            key, key_val = jax.random.split(key)
            val_loss = validation_loop(
                key=key_val,
                ema_model=ema_model,
                dataloader=val_dataloader,
                step_fn=val_step,
                coupling=coupling,
                time_sampler=time_sampler,
                path_sampler=path_sampler,
                p_uncond=p_uncond,
            )
            val_time = time.perf_counter() - val_start_time
            total_val_time += val_time
            val_runs += 1
            avg_val_time = total_val_time / val_runs

        if (epoch + 1) % checkpoint_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            raw_path = os.path.join(checkpoint_dir, f"model_epoch{epoch + 1}_raw.eqx")
            ema_path = os.path.join(checkpoint_dir, f"model_epoch{epoch + 1}_ema.eqx")
            eqx.tree_serialise_leaves(raw_path, state.model)
            eqx.tree_serialise_leaves(ema_path, ema_model)
            logger.info(f"Saved checkpoint: {ema_path}")

        epoch_time = time.perf_counter() - epoch_start_time
        total_epoch_time += epoch_time
        avg_epoch_time = total_epoch_time / (epoch + 1)

        if (epoch + 1) % log_every == 0:
            log_string = (
                f"Epoch {epoch + 1}/{num_epochs} | "
                + f"Train Loss: {epoch_loss / steps_per_epoch:.4g} | "
                + f"Val Loss: {val_loss:.4g} | "
                + f"Epoch Time: {epoch_time:.2g}s (avg {avg_epoch_time:.2g}s) | "
                + f"Train Time: {train_time:.2g}s (avg {avg_train_time:.2g}s) | "
                + f"Val Time: {val_time:.2f}s (avg {avg_val_time:.2f}s)"
            )
            logger.info(log_string)

    return ema_model
