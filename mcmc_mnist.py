"""MCMC posterior sampling for a blurred MNIST image using a trained FM model.

Formulation
-----------
Forward model:  Y = A(FM(Z)) + ε,   ε ~ N(0, σ²I)
Prior:          Z ~ N(0, I)
Posterior:      log p(Z|Y) = -‖Y - A(FM(Z))‖² / (2σ²) - ‖Z‖² / 2

Multiple independent chains are run in parallel via ``jax.vmap``. This
enables the Gelman-Rubin R-hat diagnostic and makes multi-modal posteriors
less likely to be missed. Progress images are saved every ``save_every``
steps so you can visually inspect convergence.

Usage
-----
    uv run python mcmc_mnist.py checkpoint=checkpoints/mnist/model_epoch10_best_ema.eqx
    uv run python mcmc_mnist.py checkpoint=... mcmc.n_chains=4 mcmc.num_samples=500
"""

import csv
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
import matplotlib.gridspec as gridspec

from hydra.utils import instantiate
from omegaconf import DictConfig

from msdflow.utils import register_all_resolvers
from msdflow.flow.sample import integrate_from_z
from msdflow.data.blur import gaussian_blur
from msdflow.data.mnist_inverse import BlurredMNIST


register_all_resolvers()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def gelman_rubin(chains: np.ndarray) -> np.ndarray:
    """Compute per-dimension R-hat (Gelman-Rubin) from multiple chains.

    Values close to 1.0 indicate convergence. A common threshold is R-hat < 1.1.

    Args:
        chains: Array of shape ``(n_chains, n_samples, n_dims)``.

    Returns:
        Array of shape ``(n_dims,)`` with per-dimension R-hat values.
    """
    n_chains, n_samples, n_dims = chains.shape
    chain_means = chains.mean(axis=1)        # (n_chains, n_dims)
    overall_mean = chain_means.mean(axis=0)  # (n_dims,)

    # Between-chain variance
    B = n_samples * ((chain_means - overall_mean) ** 2).sum(axis=0) / (n_chains - 1)

    # Within-chain variance
    W = ((chains - chain_means[:, None, :]) ** 2).sum(axis=(0, 1)) / (
        n_chains * (n_samples - 1)
    )

    var_hat = (n_samples - 1) / n_samples * W + B / n_samples
    return np.sqrt(var_hat / np.maximum(W, 1e-10))


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a (H, W) array in [-1, 1] to a uint8 array in [0, 255]."""
    return np.clip((img + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)


def save_progress_image(
    step: int,
    y_obs: np.ndarray,
    x_true: np.ndarray,
    chain_z: np.ndarray,
    model,
    image_size: int,
    progress_dir: str,
) -> None:
    """Save a grid image comparing reference images with current chain states.

    Layout: [y_obs | x_true | chain_0 | chain_1 | ...]

    Args:
        step:         Current sampling step number.
        y_obs:        Blurred observation ``(1, H, W)``.
        x_true:       Ground-truth clean image ``(1, H, W)``.
        chain_z:      Current chain positions ``(n_chains, n_dims)``.
        model:        FM model (inference mode).
        image_size:   Spatial size of images.
        progress_dir: Directory for saved PNGs.
    """
    n_chains = chain_z.shape[0]
    n_cols = 2 + n_chains  # y_obs, x_true, one per chain

    fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols, 2.5))

    def show(ax, img_chw, title):
        ax.imshow(_to_uint8(img_chw[0]), cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    show(axes[0], np.array(y_obs), "y_obs\n(blurred)")
    show(axes[1], np.array(x_true), "x_true\n(clean)")

    for c in range(n_chains):
        z = jnp.array(chain_z[c]).reshape(1, image_size, image_size)
        x_gen = np.array(integrate_from_z(z, model))
        show(axes[2 + c], x_gen, f"chain {c}\nstep {step}")

    plt.tight_layout()
    path = os.path.join(progress_dir, f"step_{step:06d}.png")
    plt.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./configs", config_name="config_mnist")
def main(cfg: DictConfig):

    checkpoint = cfg.checkpoint
    if checkpoint is None:
        raise ValueError(
            "Provide a checkpoint path, e.g.:\n"
            "  python mcmc_mnist.py checkpoint=checkpoints/mnist/model_epoch10_best_ema.eqx"
        )

    mcmc_cfg = cfg.mcmc
    sigma_blur: float = mcmc_cfg.sigma_blur
    sigma_noise: float = mcmc_cfg.sigma_noise
    sample_idx: int = mcmc_cfg.sample_idx
    step_size: float = mcmc_cfg.step_size
    num_warmup: int = mcmc_cfg.num_warmup
    num_samples: int = mcmc_cfg.num_samples
    n_chains: int = mcmc_cfg.n_chains
    save_every: int = mcmc_cfg.save_every
    output_dir: str = mcmc_cfg.output_dir
    progress_dir: str = os.path.join(output_dir, "progress")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)

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
    #    log p(Z|Y) = -‖Y - A(FM(Z))‖² / (2σ²) - ‖Z‖² / 2
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
    # 4. Initialise n_chains in parallel
    #    Each chain starts from an independent draw from the prior N(0,I)
    # ------------------------------------------------------------------
    nuts = blackjax.nuts(
        log_posterior,
        step_size=step_size,
        inverse_mass_matrix=jnp.ones(n_dims),
    )

    rng_key = jr.PRNGKey(cfg.seed + 1)
    rng_key, init_key = jr.split(rng_key)
    z_inits = jr.normal(init_key, (n_chains, n_dims))  # (n_chains, 784)

    init_fn = jax.vmap(nuts.init)
    step_fn = jax.jit(jax.vmap(nuts.step))
    states = init_fn(z_inits)

    # ------------------------------------------------------------------
    # 5. Warm-up
    # ------------------------------------------------------------------
    log.info("Warm-up: %d steps x %d chains", num_warmup, n_chains)
    for i in range(num_warmup):
        rng_key, subkey = jr.split(rng_key)
        keys = jr.split(subkey, n_chains)
        states, infos = step_fn(keys, states)
        if (i + 1) % 50 == 0:
            mean_acc = float(infos.acceptance_rate.mean())
            log.info("  warm-up %d/%d  mean_acceptance=%.3f", i + 1, num_warmup, mean_acc)

    # ------------------------------------------------------------------
    # 6. Sampling — collect chains, save progress images and stats CSV
    # ------------------------------------------------------------------
    log.info("Sampling: %d steps x %d chains", num_samples, n_chains)

    # chains[chain, step, dim]
    all_z = np.zeros((n_chains, num_samples, n_dims), dtype=np.float32)

    stats_path = os.path.join(output_dir, "chain_stats.csv")
    with open(stats_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "chain", "acceptance_rate", "is_divergent", "tree_depth"])

        for i in range(num_samples):
            rng_key, subkey = jr.split(rng_key)
            keys = jr.split(subkey, n_chains)
            states, infos = step_fn(keys, states)

            positions = np.array(states.position)  # (n_chains, 784)
            all_z[:, i, :] = positions

            # Write per-chain stats
            for c in range(n_chains):
                writer.writerow([
                    i,
                    c,
                    float(infos.acceptance_rate[c]),
                    bool(infos.is_divergent[c]),
                    int(infos.num_integration_steps[c]),
                ])

            # Periodic progress image
            if (i + 1) % save_every == 0:
                log.info(
                    "  step %d/%d  mean_acceptance=%.3f  divergent=%d",
                    i + 1, num_samples,
                    float(infos.acceptance_rate.mean()),
                    int(infos.is_divergent.sum()),
                )
                save_progress_image(
                    step=i + 1,
                    y_obs=np.array(y_obs),
                    x_true=np.array(x_true),
                    chain_z=positions,
                    model=model,
                    image_size=cfg.image_size,
                    progress_dir=progress_dir,
                )

    # ------------------------------------------------------------------
    # 7. Gelman-Rubin R-hat
    # ------------------------------------------------------------------
    r_hat = gelman_rubin(all_z)  # (784,)
    log.info(
        "Gelman-Rubin R-hat — mean: %.4f  max: %.4f  fraction > 1.1: %.2f%%",
        float(r_hat.mean()),
        float(r_hat.max()),
        float((r_hat > 1.1).mean()) * 100,
    )
    np.save(os.path.join(output_dir, "r_hat.npy"), r_hat)

    # Save R-hat reshaped as a 28×28 heatmap so you can see which pixels
    # of the latent space converged well
    r_hat_img = r_hat.reshape(cfg.image_size, cfg.image_size)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(r_hat_img, vmin=1.0, vmax=1.5, cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax, label="R-hat")
    ax.set_title("Gelman-Rubin R-hat (per latent pixel)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "r_hat_heatmap.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8. Decode all posterior Z samples → images
    # ------------------------------------------------------------------
    log.info("Decoding posterior samples")
    # Flatten chains: (n_chains * n_samples, 784)
    z_flat_all = all_z.reshape(-1, n_dims)
    decode = jax.jit(lambda z: integrate_from_z(
        z.reshape(1, cfg.image_size, cfg.image_size), model
    ))
    x_samples = np.stack([np.array(decode(jnp.array(z))) for z in z_flat_all])
    # Restore chain dimension: (n_chains, n_samples, 1, H, W)
    x_samples = x_samples.reshape(n_chains, num_samples, 1, cfg.image_size, cfg.image_size)

    np.save(os.path.join(output_dir, "z_samples.npy"), all_z)
    np.save(os.path.join(output_dir, "x_samples.npy"), x_samples)

    log.info("Done. Outputs in %s", output_dir)
    log.info("  z_samples:  %s  (chains × steps × dims)", all_z.shape)
    log.info("  x_samples:  %s  (chains × steps × C × H × W)", x_samples.shape)
    log.info("  chain_stats: %s", stats_path)


if __name__ == "__main__":
    main()
