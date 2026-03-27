"""Training loop and state management for flow matching.

Provides ``TrainState``, a JIT-compiled train step factory, and
the main training loop with periodic checkpointing.
"""

from typing import Any

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import os
import logging
import numpy as np

from src.flow.otfm import flow_matching_loss, minibatch_ot_coupling

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
        x0: jax.Array,
        x1: jax.Array,
        t: jax.Array,
    ) -> tuple[TrainState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(flow_matching_loss)(
            state.model, x0, x1, t
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
        model:      UNet to train
        dataloader: PyTorch DataLoader yielding ``(images, meta)`` tuples
                    where images is a ``(B, C, H, W)`` tensor.
        optimizer:  Optax GradientTransformation (construct via
                    hydra.utils.instantiate(cfg.train.optimizer) before calling)

    Returns:
        Trained model.
    """
    state = make_train_state(model, optimizer)
    train_step = make_train_step(optimizer)
    key = jax.random.PRNGKey(cfg.seed)

    t_min = float(cfg.flow.otfm.t_min)
    t_max = float(cfg.flow.otfm.t_max)
    num_steps = int(cfg.train.num_steps)
    log_every = int(cfg.train.log_every)
    ckpt_every = int(cfg.train.checkpoint_every)
    ckpt_dir = cfg.train.checkpoint_dir

    data_iter = iter(dataloader)
    for step in range(num_steps):
        # Fetch next batch (cycle through dataloader)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images, _meta = batch
        x1_np = images.numpy()
        B = x1_np.shape[0]

        # Split key each step for reproducible, non-repeated noise draws.
        key, subkey = jax.random.split(key)
        # Derive a CPU seed from subkey so NumPy randomness is JAX-seed-controlled.
        cpu_seed = int(jax.random.randint(subkey, shape=(), minval=0, maxval=2**31 - 1))
        rng = np.random.default_rng(cpu_seed)

        # CPU-side: OT coupling and time sampling
        x0_np = rng.standard_normal(x1_np.shape).astype(np.float32)
        t_np = rng.uniform(t_min, t_max, size=(B,)).astype(np.float32)
        x0_paired = minibatch_ot_coupling(x0_np, x1_np)

        # Convert to JAX arrays
        x0 = jnp.array(x0_paired)
        x1 = jnp.array(x1_np)
        t = jnp.array(t_np)

        state, loss = train_step(state, x0, x1, t)

        if step % log_every == 0:
            logger.info(f"step={step}  loss={float(loss):.6f}")

        if step > 0 and step % ckpt_every == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            path = os.path.join(ckpt_dir, f"model_step{step}.eqx")
            eqx.tree_serialise_leaves(path, state.model)
            logger.info(f"Saved checkpoint: {path}")

    return state.model
