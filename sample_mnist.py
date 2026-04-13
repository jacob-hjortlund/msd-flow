"""MCMC / flow-sampling script for a trained MNIST model.

Loads a checkpoint produced by ``train_mnist.py`` and generates samples.
Sampling uses the ODE solver already wired up in ``msdflow.flow.sample``
(Euler integrator by default; override via ``flow.sample.*`` config keys).

Run from the ``msd-flow/`` directory:

    python sample_mnist.py checkpoint=checkpoints/mnist/model_epoch10_best_ema.eqx
    python sample_mnist.py checkpoint=checkpoints/mnist/model_epoch10_best_ema.eqx \\
        num_samples=16 flow.sample.dt0=0.005
"""

import importlib
import os
import logging

import hydra
import jax.random as jr
import equinox as eqx
import numpy as np

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from msdflow.utils import register_all_resolvers
from msdflow.flow.sample import sample as ode_sample


def _import_class(dotted_path: str):
    """Import a class from a fully-qualified dotted string, e.g. 'diffrax.Euler'."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


register_all_resolvers()
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config_mnist")
def main(cfg: DictConfig):

    checkpoint: str = cfg.get("checkpoint", None)
    num_samples: int = cfg.get("num_samples", 16)
    output_dir: str = cfg.get("output_dir", os.path.join(cfg.work_dir, "samples_mcmc"))

    if checkpoint is None:
        raise ValueError(
            "Provide a checkpoint path via 'checkpoint=<path>'. "
            "Example: python sample_mnist.py checkpoint=checkpoints/model_epoch10_best_ema.eqx"
        )

    # 1. Reconstruct model structure from config, then load weights
    log.info(f"--- Loading checkpoint: {checkpoint} ---")
    rng_key = jr.PRNGKey(cfg.seed)
    model_key, rng_key = jr.split(rng_key)
    model_like = instantiate(cfg.model)(key=model_key)
    model = eqx.tree_deserialise_leaves(checkpoint, model_like)
    model = eqx.nn.inference_mode(model, value=True)

    # 2. Build the sample config from flow.sample
    sample_cfg = OmegaConf.to_container(cfg.flow.sample, resolve=True)
    solver_cls = _import_class(sample_cfg["solver"])
    controller_cls = _import_class(sample_cfg["stepsize_controller"])

    image_shape = (1, cfg.image_size, cfg.image_size)

    # 3. Generate samples via ODE integration (flow matching sampler)
    log.info(f"--- Generating {num_samples} samples ---")
    sample_key, rng_key = jr.split(rng_key)
    sample_keys = jr.split(sample_key, num_samples)

    images = []
    for i, key_i in enumerate(sample_keys):
        img = ode_sample(
            model=model,
            shape=image_shape,
            key=key_i,
            solver=solver_cls,
            dt0=sample_cfg["dt0"],
            t0=sample_cfg["t0"],
            t1=sample_cfg["t1"],
            stepsize_controller=controller_cls,
            stepsize_controller_cfg=sample_cfg.get("stepsize_controller_cfg", {}),
            cond=None,
            guidance_scale=sample_cfg.get("guidance_scale", 1.0),
        )
        images.append(np.array(img))
        if (i + 1) % 4 == 0:
            log.info(f"  Sampled {i + 1}/{num_samples}")

    images = np.stack(images)  # (N, 1, 28, 28)

    # 4. Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "samples.npy")
    np.save(out_path, images)
    log.info(f"Saved {num_samples} samples to {out_path}  (shape {images.shape})")


if __name__ == "__main__":
    main()
