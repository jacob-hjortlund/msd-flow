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

from src.flow.otfm import flow_matching_loss, sample_path
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


def train(cfg, model, dataloader, optimizer: optax.GradientTransformation):
    """Main training loop.

    Args:
        cfg:        Hydra DictConfig with cfg.seed, cfg.train.*, cfg.flow.otfm.*
        model:      Velocity-field network to train.
        dataloader: PyTorch DataLoader yielding ``(images, meta)`` tuples
                    where images is ``(B, C, H, W)`` and meta is
                    ``(B, cond_dim)`` or ``(B, 0)`` if unconditional.
        optimizer:  Optax GradientTransformation (construct via
                    hydra.utils.instantiate(cfg.train.optimizer) before calling).

    Returns:
        Trained model.
    """
    state = make_train_state(model, optimizer)
    train_step = make_train_step(optimizer)
    key = jax.random.PRNGKey(cfg.seed)

    t_min = float(cfg.flow.otfm.t_min)
    t_max = float(cfg.flow.otfm.t_max)
    sigma_0 = float(cfg.flow.otfm.get("sigma_0", 0.0))
    sigma_1 = float(cfg.flow.otfm.get("sigma_1", 0.0))
    num_steps = int(cfg.train.num_steps)
    log_every = int(cfg.train.log_every)
    ckpt_every = int(cfg.train.checkpoint_every)
    ckpt_dir = cfg.train.checkpoint_dir
    p_uncond = float(cfg.train.get("p_uncond", 0.0))

    _sample_path = jax.jit(
        functools.partial(sample_path, sigma_0=sigma_0, sigma_1=sigma_1)
    )

    data_iter = iter(dataloader)
    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images, meta = batch
        x1_np = images.numpy()
        cond_np = meta.numpy()
        B = x1_np.shape[0]

        key, key_cpu, key_path = jax.random.split(key, 3)
        cpu_seed = int(jax.random.randint(key_cpu, shape=(), minval=0, maxval=2**31 - 1))
        rng = np.random.default_rng(cpu_seed)

        x0_np = rng.standard_normal(x1_np.shape).astype(np.float32)
        t_np = rng.uniform(t_min, t_max, size=(B,)).astype(np.float32)
        x0_paired = ot_coupling(x0_np, x1_np)

        # CFG: randomly drop condition per sample with probability p_uncond
        cond_mask_np = (rng.random(B) >= p_uncond).astype(bool)

        x_t, u_t = _sample_path(
            jnp.array(x0_paired), jnp.array(x1_np), jnp.array(t_np), key=key_path
        )
        t = jnp.array(t_np)
        cond = jnp.array(cond_np)
        cond_mask = jnp.array(cond_mask_np)

        state, loss = train_step(state, x_t, u_t, t, cond, cond_mask)

        if step % log_every == 0:
            logger.info(f"step={step}  loss={float(loss):.6f}")

        if step > 0 and step % ckpt_every == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            path = os.path.join(ckpt_dir, f"model_step{step}.eqx")
            eqx.tree_serialise_leaves(path, state.model)
            logger.info(f"Saved checkpoint: {path}")

    return state.model
