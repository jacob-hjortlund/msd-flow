"""Training loop and state management for flow matching.

Provides ``TrainState``, a JIT-compiled train step factory, and
the main training loop with periodic checkpointing.
"""

from typing import Any

import functools

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import os
import logging
import numpy as np

from src.flow.otfm import (
    flow_matching_loss,
    sample_path,
    sample_time_uniform,
    sample_time_logit_normal,
)
from src.flow.coupling import ot_coupling

logger = logging.getLogger(__name__)


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


def _run_validation(
    ema_model,
    val_dataloader,
    val_step,
    sample_path_fn,
    key: jax.Array,
    time_sampling: str,
    t_min: float,
    t_max: float,
    p_uncond: float,
) -> float:
    """Run a full pass over val_dataloader and return mean flow matching loss.

    Args:
        ema_model:       EMA model used for inference.
        val_dataloader:  Iterable of ``(images, meta)`` batches.
        val_step:        JIT-compiled val step from ``make_val_step()``.
        sample_path_fn:  JIT-compiled ``sample_path`` partial.
        key:             JAX PRNG key (consumed internally via splitting).
        time_sampling:   ``"uniform"`` or ``"logit_normal"``.
        t_min:           Lower time bound (uniform sampling only).
        t_max:           Upper time bound (uniform sampling only).
        p_uncond:        Probability of dropping the condition per sample.

    Returns:
        Mean validation loss over all batches.
    """
    total_loss = 0.0
    n_batches = 0

    for batch in val_dataloader:
        images, meta = batch
        x1_np = images.numpy()
        cond_np = meta.numpy()
        B = x1_np.shape[0]

        key, key_cpu, key_time, key_path = jax.random.split(key, 4)
        cpu_seed = int(jax.random.randint(key_cpu, shape=(), minval=0, maxval=2**31 - 1))
        rng = np.random.default_rng(cpu_seed)

        x0_np = rng.standard_normal(x1_np.shape).astype(np.float32)
        x0_paired = ot_coupling(x0_np, x1_np)

        if time_sampling == "uniform":
            t = sample_time_uniform(key_time, B, t_min, t_max)
        elif time_sampling == "logit_normal":
            t = sample_time_logit_normal(key_time, B)
        else:
            raise ValueError(
                f"Unknown time_sampling={time_sampling!r}; "
                "choose 'uniform' or 'logit_normal'."
            )

        cond_mask_np = (rng.random(B) >= p_uncond).astype(bool)

        x_t, u_t = sample_path_fn(
            jnp.array(x0_paired), jnp.array(x1_np), t, key=key_path
        )
        cond = jnp.array(cond_np)
        cond_mask = jnp.array(cond_mask_np)

        total_loss += float(val_step(ema_model, x_t, u_t, t, cond, cond_mask))
        n_batches += 1

    return total_loss / n_batches


def train(cfg, model, dataloader, val_dataloader, optimizer: optax.GradientTransformation):
    """Main training loop with EMA and periodic validation.

    Args:
        cfg:            Hydra DictConfig with cfg.seed, cfg.train.*, cfg.flow.otfm.*
        model:          Velocity-field network to train.
        dataloader:     PyTorch DataLoader yielding ``(images, meta)`` tuples
                        where images is ``(B, C, H, W)`` and meta is
                        ``(B, cond_dim)`` or ``(B, 0)`` if unconditional.
        val_dataloader: DataLoader for the validation split, same format as
                        ``dataloader``. Used for periodic EMA model evaluation.
        optimizer:      Optax GradientTransformation (construct via
                        hydra.utils.instantiate(cfg.train.optimizer) before calling).

    Returns:
        Trained EMA model.
    """
    state = make_train_state(model, optimizer)
    train_step = make_train_step(optimizer)
    val_step = make_val_step()
    key = jax.random.PRNGKey(cfg.seed)

    t_min = float(cfg.flow.otfm.t_min)
    t_max = float(cfg.flow.otfm.t_max)
    sigma_0 = float(cfg.flow.otfm.get("sigma_0", 0.0))
    sigma_1 = float(cfg.flow.otfm.get("sigma_1", 0.0))
    time_sampling = cfg.flow.otfm.get("time_sampling", "uniform")
    num_epochs = int(cfg.train.num_epochs)
    num_steps_per_epoch = int(cfg.train.num_steps_per_epoch)
    log_every = int(cfg.train.log_every)
    ckpt_every = int(cfg.train.checkpoint_every)
    ckpt_dir = cfg.train.checkpoint_dir
    p_uncond = float(cfg.train.get("p_uncond", 0.0))
    ema_decay = float(cfg.train.get("ema_decay", 0.9999))
    val_every = int(cfg.train.val_every)

    steps_per_epoch = (
        len(dataloader) if num_steps_per_epoch == 0 else num_steps_per_epoch
    )

    _sample_path = jax.jit(
        functools.partial(sample_path, sigma_0=sigma_0, sigma_1=sigma_1)
    )

    ema_model = model
    data_iter = iter(dataloader)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            images, meta = batch
            x1_np = images.numpy()
            cond_np = meta.numpy()
            B = x1_np.shape[0]

            key, key_cpu, key_time, key_path = jax.random.split(key, 4)
            cpu_seed = int(
                jax.random.randint(key_cpu, shape=(), minval=0, maxval=2**31 - 1)
            )
            rng = np.random.default_rng(cpu_seed)

            x0_np = rng.standard_normal(x1_np.shape).astype(np.float32)
            x0_paired = ot_coupling(x0_np, x1_np)

            if time_sampling == "uniform":
                t = sample_time_uniform(key_time, B, t_min, t_max)
            elif time_sampling == "logit_normal":
                # logit_normal samples from (0, 1) by design; t_min/t_max are
                # not applied because the sigmoid maps to the full open interval.
                t = sample_time_logit_normal(key_time, B)
            else:
                raise ValueError(
                    f"Unknown time_sampling={time_sampling!r}; "
                    "choose 'uniform' or 'logit_normal'."
                )

            # CFG: randomly drop condition per sample with probability p_uncond
            cond_mask_np = (rng.random(B) >= p_uncond).astype(bool)

            x_t, u_t = _sample_path(
                jnp.array(x0_paired), jnp.array(x1_np), t, key=key_path
            )
            cond = jnp.array(cond_np)
            cond_mask = jnp.array(cond_mask_np)

            state, loss = train_step(state, x_t, u_t, t, cond, cond_mask)
            ema_model = ema_update(ema_model, state.model, ema_decay)
            epoch_loss += float(loss)

        if (epoch + 1) % log_every == 0:
            logger.info(
                f"epoch={epoch + 1}  loss={epoch_loss / steps_per_epoch:.6f}"
            )

        if (epoch + 1) % val_every == 0:
            key, key_val = jax.random.split(key)
            val_loss = _run_validation(
                ema_model,
                val_dataloader,
                val_step,
                _sample_path,
                key_val,
                time_sampling,
                t_min,
                t_max,
                p_uncond,
            )
            logger.info(f"epoch={epoch + 1}  val_loss={val_loss:.6f}")

        if (epoch + 1) % ckpt_every == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            path = os.path.join(ckpt_dir, f"model_epoch{epoch + 1}.eqx")
            eqx.tree_serialise_leaves(path, ema_model)
            logger.info(f"Saved checkpoint: {path}")

    return ema_model
