import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from scipy.linalg import sqrtm

# ---------------------------------------------------------------------------
# Metric signatures
# ---------------------------------------------------------------------------
# All metrics are plain callables configured via Hydra ``_target_``. Two
# signatures are expected, depending on when the metric is evaluated:
#
#   Batch metric:  (model, x_t, u_t, t, cond, cond_mask, key) -> scalar
#     Evaluated per-batch during validation. Receives prepared interpolant
#     tensors. Must return a scalar JAX array. Used for logging and
#     overfitting detection (train vs. val comparison).
#
#   Epoch metric:  (model, val_dataloader, key) -> scalar
#     Evaluated once per validation cycle. Receives the val dataloader
#     iterable directly and streams through it (no pre-collection).
#     Any additional dependencies (solver, n_samples, etc.) should be
#     baked in via Hydra ``_partial_: true``. Used for generation-based
#     metrics (e.g. FID) and early stopping.
# ---------------------------------------------------------------------------


def _to_velocity(
    pred: jnp.ndarray,
    x_t: jnp.ndarray,
    t: jnp.ndarray,
    prediction_type: str,
) -> jnp.ndarray:
    """Convert a model prediction to a velocity field.

    Args:
        pred:            shape (B, C, H, W) — raw model output.
        x_t:             shape (B, C, H, W) — interpolated samples at time t.
        t:               shape (B,) — per-sample times in [0, 1).
        prediction_type: ``"velocity"`` returns ``pred`` unchanged;
            ``"image"`` applies ``(pred - x_t) / (1 - t)``.
            Must be a Python string constant (not a traced JAX value) when
            this function is called inside ``jax.jit`` or ``eqx.filter_jit``.

    Returns:
        Velocity field of shape (B, C, H, W).
    """
    if prediction_type == "image":
        t_ = t[:, None, None, None]
        return (pred - x_t) / (1.0 - t_)
    return pred


def flow_matching_loss(
    model,
    x_t: jnp.ndarray,
    u_t: jnp.ndarray,
    t: jnp.ndarray,
    cond: jnp.ndarray,
    cond_mask: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Compute the flow matching MSE loss.

    Supports velocity-predicting and image-predicting models. The loss is
    always computed in velocity space; image-space predictions are converted
    via ``v_t = (x_t_pred - x_t) / (1 - t)`` before the MSE is evaluated.

    Args:
        model: Network accepting ``(t, x_t, cond, cond_mask)``. Must have a
            ``prediction_type`` attribute of ``"velocity"`` (default) or
            ``"image"``.
        x_t:   shape (B, C, H, W) — interpolated samples at time t.
        u_t:   shape (B, C, H, W) — target velocities (x1 - x0).
        t:     shape (B,) — per-sample times in [0, 1).
        cond:  shape (B, cond_dim) — conditioning vectors. Pass
            ``jnp.empty((B, 0))`` when the model is unconditional.
        cond_mask: shape (B,) bool — per-sample mask. ``True`` = use
            the real condition; ``False`` = use the null embedding.

    Returns:
        Scalar mean squared error between predicted and target velocities.
    """
    pred = eqx.filter_vmap(model)(t, x_t, cond, cond_mask, key)
    v_t = _to_velocity(pred, x_t, t, model.prediction_type)
    return jnp.mean((v_t - u_t) ** 2)


def _frechet_distance(
    mu_real: np.ndarray,
    sigma_real: np.ndarray,
    mu_fake: np.ndarray,
    sigma_fake: np.ndarray,
) -> float:
    """Compute the Fréchet distance between two multivariate Gaussians.

    Args:
        mu_real:    Mean of real distribution, shape (D,).
        sigma_real: Covariance of real distribution, shape (D, D).
        mu_fake:    Mean of fake distribution, shape (D,).
        sigma_fake: Covariance of fake distribution, shape (D, D).

    Returns:
        Fréchet distance (scalar float).
    """
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real @ sigma_fake)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean))


@eqx.filter_jit
def _extract_batch(encoder, images):
    """Encode a batch of images into feature vectors.

    Args:
        encoder: Callable mapping a single image (C, H, W) to features (D,).
        images:  Batch of images, shape (B, C, H, W).

    Returns:
        Feature matrix of shape (B, D).
    """
    return jax.vmap(encoder)(images)


class FIDAccumulator:
    """Accumulates streaming statistics for one encoder.

    Maintains running sums for mean and covariance computation with O(D²)
    memory. Does not store images or feature vectors beyond one batch.

    Args:
        encoder: Callable mapping a single image (C, H, W) to features (D,).
            Must be JAX-vmappable.
    """

    def __init__(self, encoder: callable):
        self.encoder = encoder
        self._sum_features = None  # np.ndarray (D,)
        self._sum_outer = None  # np.ndarray (D, D)
        self._n = 0
        self._cached_real = None  # set by compute_fid_metrics

    def update(self, images: jax.Array) -> None:
        """Encode a batch and update running accumulators.

        Args:
            images: Batch of images, shape (B, C, H, W).
        """
        features = np.asarray(_extract_batch(self.encoder, images))  # (B, D)
        if self._sum_features is None:
            D = features.shape[1]
            self._sum_features = np.zeros(D, dtype=np.float64)
            self._sum_outer = np.zeros((D, D), dtype=np.float64)
        f64 = features.astype(np.float64)
        self._sum_features += f64.sum(axis=0)
        self._sum_outer += f64.T @ f64
        self._n += features.shape[0]

    def statistics(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Compute mean, covariance, and count from accumulated sums.

        Returns:
            Tuple of (mu, sigma, n) where mu has shape (D,), sigma has
            shape (D, D), and n is the total image count.
        """
        if self._n == 0:
            empty = np.array([])
            return empty, np.array([[]]), 0
        mu = self._sum_features / self._n
        sigma = (self._sum_outer / self._n) - np.outer(mu, mu)
        return mu, sigma, self._n

    def reset(self) -> None:
        """Zero streaming accumulators for reuse across epochs.

        Does not clear cached real-image statistics (``_cached_real``).
        """
        self._sum_features = None
        self._sum_outer = None
        self._n = 0


def compute_fid_metrics(
    accumulators: dict[str, "FIDAccumulator"],
    model,
    val_dataloader,
    generate_fn: callable,
    n_samples: int | None,
    gen_batch_size: int,
    key: jax.Array,
    n_real: int | None = None,
) -> dict[str, float]:
    """Compute FID scores for one or more encoders.

    Iterates the validation dataloader once (or skips if cached) and
    generates fake images in chunks, dispatching each batch to all
    accumulators. Returns one FID score per accumulator.

    Args:
        accumulators:    Named accumulators, one per encoder. Keys become
            the output metric names.
        model:           The generative model passed to ``generate_fn``.
        val_dataloader:  Iterable yielding ``(images, meta)`` tuples.
        generate_fn:     ``(model, key=...) -> jax.Array`` of shape ``(C, H, W)``.
            One unconditional sample. Solver args baked in via partial.
            Called as ``generate_fn(model, key=k)``.
        n_samples:       Number of fake images. ``None`` matches real count.
        gen_batch_size:  Images generated and encoded per chunk.
        key:             PRNG key for generation.
        n_real:          Maximum number of real images to use from
            ``val_dataloader``. ``None`` (default) uses the full dataset.

    Returns:
        Dict mapping accumulator names to FID scores.
    """
    # --- Real-image pass (skip if all accumulators have cached stats) ---
    all_cached = all(acc._cached_real is not None for acc in accumulators.values())
    if not all_cached:
        for acc in accumulators.values():
            acc.reset()
        n_real_seen = 0
        for images, _meta in val_dataloader:
            images = images.numpy()
            images = jnp.asarray(images)
            if n_real is not None:
                remaining = n_real - n_real_seen
                if remaining <= 0:
                    break
                if images.shape[0] > remaining:
                    images = images[:remaining]
            for acc in accumulators.values():
                acc.update(images)
            n_real_seen += images.shape[0]
        for acc in accumulators.values():
            mu, sigma, n = acc.statistics()
            acc._cached_real = (mu, sigma, n)
            acc.reset()

    # --- Determine n_samples ---
    if n_samples is None:
        n_samples = max(acc._cached_real[2] for acc in accumulators.values())

    # --- Fake-image pass ---
    for acc in accumulators.values():
        acc.reset()

    n_generated = 0

    def _generate_fn(key):
        return generate_fn(model, key=key)

    _generate_fn = eqx.filter_jit(jax.vmap(_generate_fn))

    while n_generated < n_samples:
        chunk_size = min(gen_batch_size, n_samples - n_generated)
        all_keys = jax.random.split(key, chunk_size + 1)
        key = all_keys[0]
        sub_keys = all_keys[1:]
        fake_images = _generate_fn(sub_keys)
        for acc in accumulators.values():
            acc.update(fake_images)
        n_generated += chunk_size

    # --- Compute FID per accumulator ---
    results = {}
    for name, acc in accumulators.items():
        mu_real, sigma_real, _ = acc._cached_real
        mu_fake, sigma_fake, _ = acc.statistics()
        results[name] = _frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    return results


class FIDMetric:
    """Epoch metric wrapper that adapts ``compute_fid_metrics`` to the
    ``(model, val_dataloader, key)`` signature expected by the trainer.

    Holds persistent ``FIDAccumulator`` instances so real-image statistics
    are cached across epochs. All generation and scoring logic is delegated
    to ``compute_fid_metrics``.

    Args:
        accumulators:   Named accumulators, one per encoder. Keys become
            the output metric names.
        generate_fn:    ``(model, key=...) -> jax.Array`` of shape ``(C, H, W)``.
            One unconditional sample. Solver args baked in via partial.
            Called as ``generate_fn(model, key=k)``.
        n_samples:      Number of fake images. ``None`` matches real count.
        gen_batch_size: Images generated and encoded per chunk.
        n_real:         Maximum real images from val_dataloader. ``None``
            uses the full dataset.
    """

    def __init__(
        self,
        accumulators: dict[str, "FIDAccumulator"],
        generate_fn: callable,
        n_samples: int | None = None,
        gen_batch_size: int = 64,
        n_real: int | None = None,
    ):
        self.accumulators = accumulators
        self.generate_fn = generate_fn
        self.n_samples = n_samples
        self.gen_batch_size = gen_batch_size
        self.n_real = n_real

    def __call__(self, model, val_dataloader, key: jax.Array) -> dict[str, float]:
        """Compute FID scores for all accumulators.

        Args:
            model:          Generative model passed to ``generate_fn``.
            val_dataloader: Iterable yielding ``(images, meta)`` tuples.
            key:            PRNG key for generation.

        Returns:
            Dict mapping accumulator names to FID scores.
        """
        return compute_fid_metrics(
            accumulators=self.accumulators,
            model=model,
            val_dataloader=val_dataloader,
            generate_fn=self.generate_fn,
            n_samples=self.n_samples,
            gen_batch_size=self.gen_batch_size,
            key=key,
            n_real=self.n_real,
        )
