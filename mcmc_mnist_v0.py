"""MCMC posterior sampling for a blurred MNIST image using a trained FM model.

Formulation
-----------
Forward model:  Y = A(FM(Z)) + ε,   ε ~ N(0, σ²I)
Prior:          Z ~ N(0, I)
Posterior:      log p(Z|Y) = -‖Y - A(FM(Z))‖² / (2σ²) - ‖Z‖² / 2

Usage
-----
    uv run python mcmc_mnist_v0.py checkpoint=checkpoints/mnist/model_epoch10_best_ema.eqx
"""

import os
import logging
import functools

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
import blackjax
import matplotlib.pyplot as plt

from hydra.utils import instantiate
from omegaconf import DictConfig

from msdflow.utils import register_all_resolvers
from msdflow.flow.sample import integrate_from_z
from msdflow.data.blur import gaussian_blur
from msdflow.data.mnist_inverse import BlurredMNIST


register_all_resolvers()
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config_mnist")
def main(cfg: DictConfig):

    checkpoint = cfg.checkpoint
    if checkpoint is None:
        raise ValueError(
            "Provide a checkpoint path, e.g.:\n"
            "  python mcmc_mnist_v0.py checkpoint=checkpoints/mnist/model_epoch10_best_ema.eqx"
        )

    mcmc_cfg = cfg.mcmc
    sigma_blur: float = mcmc_cfg.sigma_blur
    sigma_noise: float = mcmc_cfg.sigma_noise
    sample_idx: int = mcmc_cfg.sample_idx
    step_size: float = mcmc_cfg.step_size
    num_warmup: int = mcmc_cfg.num_warmup
    num_samples: int = mcmc_cfg.num_samples
    output_dir: str = mcmc_cfg.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load trained FM model
    # ------------------------------------------------------------------
    log.info("Loading checkpoint: %s", checkpoint)
    model_key = jr.PRNGKey(cfg.seed)
    model = instantiate(cfg.model)(key=model_key)
    model = eqx.tree_deserialise_leaves(checkpoint, model)
    model = eqx.nn.inference_mode(model, value=True)

    # ------------------------------------------------------------------
    # 2. Load target observation Y
    # ------------------------------------------------------------------
    log.info(
        "Loading MNIST sample %d (sigma_blur=%.2f, sigma_noise=%.3f)",
        sample_idx, sigma_blur, sigma_noise,
    )
    dataset = BlurredMNIST(
        root=os.path.join(cfg.work_dir, "data", "mnist"),
        train=False,
        sigma_blur=sigma_blur,
        sigma_noise=sigma_noise,
        seed=cfg.seed,
    )
    y_torch, x_true_torch = dataset[sample_idx]
    y_obs = jnp.array(y_torch.numpy())        # (1, 28, 28)
    x_true = jnp.array(x_true_torch.numpy())  # (1, 28, 28)

    np.save(os.path.join(output_dir, "y_obs.npy"), np.array(y_obs))
    np.save(os.path.join(output_dir, "x_true.npy"), np.array(x_true))

    # ------------------------------------------------------------------
    # 3. Log-posterior
    # ------------------------------------------------------------------
    blur_fn = functools.partial(gaussian_blur, sigma=sigma_blur)
    n_dims = cfg.image_size * cfg.image_size  # 784

    @jax.jit
    def log_posterior(z_flat: jax.Array) -> jax.Array:
        z = z_flat.reshape(1, cfg.image_size, cfg.image_size)
        x_gen = integrate_from_z(z, model)
        y_pred = blur_fn(x_gen)
        log_likelihood = -jnp.sum((y_obs - y_pred) ** 2) / (2.0 * sigma_noise ** 2)
        log_prior = -0.5 * jnp.sum(z_flat ** 2)
        return log_likelihood + log_prior

    # ------------------------------------------------------------------
    # 4. Initialise NUTS
    # ------------------------------------------------------------------
    nuts = blackjax.nuts(
        log_posterior,
        step_size=step_size,
        inverse_mass_matrix=jnp.ones(n_dims),
    )

    rng_key = jr.PRNGKey(cfg.seed + 1)
    rng_key, init_key = jr.split(rng_key)
    z_init = jr.normal(init_key, (n_dims,))
    state = nuts.init(z_init)
    step_fn = jax.jit(nuts.step)

    # ------------------------------------------------------------------
    # 5. Warm-up
    # ------------------------------------------------------------------
    log.info("Warm-up: %d steps", num_warmup)
    for i in range(num_warmup):
        rng_key, subkey = jr.split(rng_key)
        state, info = step_fn(subkey, state)
        if (i + 1) % 50 == 0:
            log.info(
                "  warm-up %d/%d  acceptance=%.3f",
                i + 1, num_warmup, float(info.acceptance_rate),
            )

    # ------------------------------------------------------------------
    # 6. Sampling
    # ------------------------------------------------------------------
    log.info("Sampling: %d steps", num_samples)
    all_z = np.zeros((num_samples, n_dims), dtype=np.float32)

    for i in range(num_samples):
        rng_key, subkey = jr.split(rng_key)
        state, info = step_fn(subkey, state)
        all_z[i] = np.array(state.position)

        if (i + 1) % 50 == 0:
            log.info(
                "  step %d/%d  acceptance=%.3f  divergent=%s",
                i + 1, num_samples,
                float(info.acceptance_rate),
                bool(info.is_divergent),
            )

    # ------------------------------------------------------------------
    # 7. Decode all posterior Z samples → images
    # ------------------------------------------------------------------
    log.info("Decoding posterior samples")
    decode = jax.jit(lambda z: integrate_from_z(
        z.reshape(1, cfg.image_size, cfg.image_size), model
    ))
    x_samples = np.stack([np.array(decode(jnp.array(z))) for z in all_z])
    # (num_samples, 1, H, W)

    np.save(os.path.join(output_dir, "z_samples.npy"), all_z)
    np.save(os.path.join(output_dir, "x_samples.npy"), x_samples)

    # ------------------------------------------------------------------
    # 8. Quick summary figure
    # ------------------------------------------------------------------
    n_show = min(8, num_samples)
    fig, axes = plt.subplots(2, n_show + 1, figsize=(2.5 * (n_show + 1), 5))

    def show(ax, img_chw, title):
        arr = np.clip((np.array(img_chw[0]) + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    show(axes[0, 0], np.array(y_obs), "y_obs")
    show(axes[1, 0], np.array(x_true), "x_true")
    for j in range(n_show):
        idx = num_samples // n_show * j
        show(axes[0, j + 1], x_samples[idx], f"x[{idx}]")
        axes[1, j + 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "posterior_samples.png"), dpi=120)
    plt.close(fig)

    log.info("Done. Outputs in %s", output_dir)
    log.info("  z_samples: %s", all_z.shape)
    log.info("  x_samples: %s", x_samples.shape)


if __name__ == "__main__":
    main()
