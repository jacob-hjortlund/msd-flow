import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from scipy.linalg import sqrtm
from jax.scipy.ndimage import map_coordinates
from tqdm import tqdm

from msdflow.train.parallel import DataParallelConfig
from msdflow.train.parallel import make_data_parallel_config
from msdflow.train.parallel import resolve_data_parallel_config


logger = logging.getLogger(__name__)
_FID_PARALLEL_AXIS_NAME = "fid_sample"


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


def _flow_matching_per_sample_loss_values(
    model,
    x_t: jnp.ndarray,
    u_t: jnp.ndarray,
    t: jnp.ndarray,
    cond: jnp.ndarray,
    cond_mask: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Compute per-sample flow-matching losses from one model prediction call.

    Args:
        model: Network accepting ``(t, x_t, cond, cond_mask, key)`` for one
            sample. Must expose ``prediction_type`` as ``"velocity"`` or
            ``"image"``.
        x_t: Interpolated samples with shape ``(B, C, H, W)``.
        u_t: Target velocity fields with shape ``(B, C, H, W)``.
        t: Per-sample times with shape ``(B,)``.
        cond: Conditioning vectors with shape ``(B, cond_dim)``.
        cond_mask: Per-sample condition mask with shape ``(B,)``.
        key: Per-sample PRNG keys with leading shape ``(B,)``.

    Returns:
        Mean squared velocity error for each sample, shape ``(B,)``.
    """
    pred = eqx.filter_vmap(model)(t, x_t, cond, cond_mask, key)
    v_t = _to_velocity(pred, x_t, t, model.prediction_type)
    squared_error = (v_t - u_t) ** 2
    reduce_axes = tuple(range(1, squared_error.ndim))
    return jnp.mean(squared_error, axis=reduce_axes)


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
    return jnp.mean(
        _flow_matching_per_sample_loss_values(
            model,
            x_t,
            u_t,
            t,
            cond,
            cond_mask,
            key,
        )
    )


def flow_matching_per_sample_loss(
    model,
    x_t: jnp.ndarray,
    u_t: jnp.ndarray,
    t: jnp.ndarray,
    cond: jnp.ndarray,
    cond_mask: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Compute one flow-matching MSE loss per batch element.

    Args:
        model: Network accepting ``(t, x_t, cond, cond_mask, key)`` for one
            sample. Must expose ``prediction_type`` as ``"velocity"`` or
            ``"image"``.
        x_t: Interpolated samples with shape ``(B, C, H, W)``.
        u_t: Target velocity fields with shape ``(B, C, H, W)``.
        t: Per-sample times with shape ``(B,)``.
        cond: Conditioning vectors with shape ``(B, cond_dim)``.
        cond_mask: Per-sample condition mask with shape ``(B,)``.
        key: Per-sample PRNG keys with leading shape ``(B,)``.

    Returns:
        Mean squared velocity error for each sample, shape ``(B,)``.
    """
    return _flow_matching_per_sample_loss_values(
        model,
        x_t,
        u_t,
        t,
        cond,
        cond_mask,
        key,
    )


def bin_time_losses(
    t: jnp.ndarray,
    losses: jnp.ndarray,
    num_bins: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Aggregate per-sample losses into equal-width time bins.

    Args:
        t: Per-sample times with shape ``(B,)``.
        losses: Per-sample losses with shape ``(B,)``.
        num_bins: Number of equal-width bins over ``[0, 1]``.

    Returns:
        Tuple of ``(loss_sums, counts)`` where both arrays have shape
        ``(num_bins,)``. ``loss_sums`` uses the same dtype as ``losses`` and
        ``counts`` uses ``int32``.

    Raises:
        ValueError: If ``num_bins`` is less than one.
    """
    num_bins = int(num_bins)
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")

    clipped_t = jnp.clip(t, 0.0, 1.0)
    bin_indices = jnp.floor(clipped_t * num_bins).astype(jnp.int32)
    bin_indices = jnp.minimum(bin_indices, num_bins - 1)

    loss_sums = jnp.zeros((num_bins,), dtype=losses.dtype)
    counts = jnp.zeros((num_bins,), dtype=jnp.int32)
    loss_sums = loss_sums.at[bin_indices].add(losses)
    counts = counts.at[bin_indices].add(1)
    return loss_sums, counts


@dataclass
class TimeBinnedLossResult:
    """Host-side accumulated loss statistics for time bins.

    Attributes:
        bin_edges: Bin edges over ``[0, 1]`` with shape ``(num_bins + 1,)``.
        loss_sums: Loss sums per bin with shape ``(num_bins,)``.
        counts: Sample counts per bin with shape ``(num_bins,)``.
    """

    bin_edges: np.ndarray
    loss_sums: np.ndarray
    counts: np.ndarray

    @classmethod
    def empty(cls, num_bins: int) -> "TimeBinnedLossResult":
        """Create an empty accumulator with equally spaced time bins.

        Args:
            num_bins: Number of equal-width bins over ``[0, 1]``.

        Returns:
            Empty result accumulator.

        Raises:
            ValueError: If ``num_bins`` is less than one.
        """
        num_bins = int(num_bins)
        if num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {num_bins}")
        return cls(
            bin_edges=np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64),
            loss_sums=np.zeros((num_bins,), dtype=np.float64),
            counts=np.zeros((num_bins,), dtype=np.int64),
        )

    @property
    def mean_loss(self) -> np.ndarray:
        """Return mean loss per bin, using NaN for empty bins."""
        return np.divide(
            self.loss_sums,
            self.counts,
            out=np.full_like(self.loss_sums, np.nan, dtype=np.float64),
            where=self.counts > 0,
        )

    def add_batch(self, loss_sums: np.ndarray, counts: np.ndarray) -> None:
        """Add one batch of bin sums and counts.

        Args:
            loss_sums: Batch loss sums per bin.
            counts: Batch sample counts per bin.

        Raises:
            ValueError: If input arrays do not match the accumulator shape.
        """
        loss_sums = np.asarray(loss_sums, dtype=np.float64)
        counts = np.asarray(counts, dtype=np.int64)
        if loss_sums.shape != self.loss_sums.shape:
            raise ValueError(
                "loss_sums shape must match time bins; "
                f"got {loss_sums.shape}, expected {self.loss_sums.shape}"
            )
        if counts.shape != self.counts.shape:
            raise ValueError(
                "counts shape must match time bins; "
                f"got {counts.shape}, expected {self.counts.shape}"
            )
        self.loss_sums += loss_sums
        self.counts += counts


@dataclass
class TimeBinnedLossHistory:
    """In-memory history for cumulative time-binned loss heatmaps.

    Attributes:
        bin_edges: Time-bin edges shared by all appended results.
        epochs: Epoch numbers in append order.
        mean_losses: Mean-loss arrays in append order.
        counts: Count arrays in append order.
    """

    bin_edges: np.ndarray
    epochs: list[int] = field(default_factory=list)
    mean_losses: list[np.ndarray] = field(default_factory=list)
    counts: list[np.ndarray] = field(default_factory=list)

    def append(self, epoch: int, result: TimeBinnedLossResult) -> None:
        """Append one epoch result to the history.

        Args:
            epoch: One-indexed epoch number.
            result: Time-binned loss result for this epoch.

        Raises:
            ValueError: If the result bin edges differ from the history edges.
        """
        if not np.allclose(np.asarray(result.bin_edges), np.asarray(self.bin_edges)):
            raise ValueError("result bin_edges must match history bin_edges")
        self.epochs.append(int(epoch))
        self.mean_losses.append(np.asarray(result.mean_loss, dtype=np.float64).copy())
        self.counts.append(np.asarray(result.counts, dtype=np.int64).copy())


def make_time_binned_loss_step(
    num_bins: int,
    data_parallel: DataParallelConfig | None = None,
):
    """Return a JIT-compiled step for time-binned flow-matching loss.

    Args:
        num_bins: Number of equal-width bins over ``[0, 1]``.
        data_parallel: Optional data-parallel runtime configuration.

    Returns:
        A callable with the same batch arguments as batch metrics. It returns
        ``(loss_sums, counts)`` arrays with shape ``(num_bins,)``.

    Raises:
        ValueError: If ``num_bins`` is less than one.
    """
    num_bins = int(num_bins)
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")
    data_parallel = resolve_data_parallel_config(data_parallel)

    @eqx.filter_jit(donate="all-except-first")
    def time_binned_loss_step(
        model,
        x_t: jax.Array,
        u_t: jax.Array,
        t: jax.Array,
        cond: jax.Array,
        cond_mask: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        if data_parallel.enabled:
            model = eqx.filter_shard(model, data_parallel.model_sharding)
            x_t, u_t, t, cond, cond_mask, key = eqx.filter_shard(
                (x_t, u_t, t, cond, cond_mask, key),
                data_parallel.data_sharding,
            )
        losses = flow_matching_per_sample_loss(
            model,
            x_t,
            u_t,
            t,
            cond,
            cond_mask,
            key,
        )
        return bin_time_losses(t, losses, num_bins)

    return time_binned_loss_step


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


def _parse_parallel_generation_enabled(value: Any) -> bool:
    """Parse a FID parallel-generation enabled flag.

    Args:
        value: Boolean-like value from direct mappings or Hydra config.

    Returns:
        Parsed boolean.

    Raises:
        ValueError: If value is not a supported boolean representation.
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
        "fid_metric.parallel_generation.enabled must be a boolean or one of "
        "'true', 'false', '1', '0', 'yes', 'no', 'on', or 'off'; "
        f"got {value!r}"
    )


def _resolve_fid_parallel_generation_config(
    parallel_generation: Any = None,
    data_parallel: DataParallelConfig | None = None,
) -> DataParallelConfig:
    """Resolve FID fake-image parallel generation settings.

    Args:
        parallel_generation: None or a mapping-like object with enabled and
            min_devices keys.
        data_parallel: Resolved trainer data-parallel config used for defaults.

    Returns:
        DataParallelConfig using the internal FID sample-axis name.

    Raises:
        TypeError: If parallel_generation is not a supported config object.
        ValueError: If enabled settings are invalid.
    """
    inherited = resolve_data_parallel_config(data_parallel)
    enabled = inherited.enabled
    min_devices = inherited.min_devices
    raw_min_devices = min_devices

    if parallel_generation is not None:
        if not (
            isinstance(parallel_generation, Mapping)
            or hasattr(parallel_generation, "get")
        ):
            raise TypeError(
                "fid_metric.parallel_generation must be None or mapping-like; "
                f"got {type(parallel_generation).__name__}"
            )
    try:
        if parallel_generation is not None:
            enabled = _parse_parallel_generation_enabled(
                parallel_generation.get("enabled", enabled)
            )
            raw_min_devices = parallel_generation.get("min_devices", min_devices)
            min_devices = int(raw_min_devices)
        if enabled and len(jax.local_devices()) < min_devices:
            return make_data_parallel_config(
                enabled=False,
                axis_name=_FID_PARALLEL_AXIS_NAME,
                min_devices=min_devices,
            )
        return make_data_parallel_config(
            enabled=enabled,
            axis_name=_FID_PARALLEL_AXIS_NAME,
            min_devices=min_devices,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fid_metric.parallel_generation is invalid: "
            f"enabled={enabled}, min_devices={raw_min_devices!r}: {exc}"
        ) from exc


def _effective_parallel_gen_batch_size(
    gen_batch_size: int,
    num_devices: int,
) -> int:
    """Return a device-divisible global generation chunk size.

    Args:
        gen_batch_size: Configured global FID generation chunk size.
        num_devices: Number of devices in the FID sample mesh.

    Returns:
        Effective global chunk size divisible by num_devices.

    Raises:
        ValueError: If gen_batch_size or num_devices is less than one.
    """
    gen_batch_size = int(gen_batch_size)
    num_devices = int(num_devices)
    if gen_batch_size < 1:
        raise ValueError(
            "fid_metric.gen_batch_size must be >= 1; "
            f"got gen_batch_size={gen_batch_size}"
        )
    if num_devices < 1:
        raise ValueError(
            "fid_metric.parallel_generation requires num_devices >= 1; "
            f"got num_devices={num_devices}"
        )
    return ((gen_batch_size + num_devices - 1) // num_devices) * num_devices


def _log_parallel_gen_batch_size_adjustment(
    gen_batch_size: int,
    effective_gen_batch_size: int,
    num_devices: int,
) -> None:
    """Warn when FID parallel generation adjusts the global chunk size.

    Args:
        gen_batch_size: Configured global generation chunk size.
        effective_gen_batch_size: Device-divisible chunk size that will run.
        num_devices: Number of local devices in the FID sample mesh.
    """
    if int(effective_gen_batch_size) == int(gen_batch_size):
        return
    logger.warning(
        "fid_metric.parallel_generation adjusted gen_batch_size "
        "from %s to %s for %s local device(s)",
        gen_batch_size,
        effective_gen_batch_size,
        num_devices,
    )


@eqx.filter_jit(donate="all-except-first")
def _batched_generate(model, keys, generate_fn):
    return jax.vmap(lambda key: generate_fn(model, key=key))(keys)


@eqx.filter_jit
def _parallel_batched_generate(model, keys, generate_fn):
    """Generate a sharded fake-image batch without donating the model.

    Args:
        model: Generative model or model-like pytree passed to generate_fn.
        keys: Device-sharded PRNG keys with leading sample dimension.
        generate_fn: Callable ``(model, key=...) -> image``.

    Returns:
        Generated image batch.
    """
    return jax.vmap(lambda key: generate_fn(model, key=key))(keys)


def _device_put_array_leaves(pytree: Any, sharding: Any | None) -> Any:
    """Place every array leaf in a pytree onto a target sharding.

    Args:
        pytree: Model or model-like pytree that may contain JAX array leaves.
        sharding: Target sharding for array leaves. ``None`` returns the input
            unchanged.

    Returns:
        Pytree with array leaves explicitly placed on ``sharding``.
    """
    if sharding is None:
        return pytree
    return jax.tree_util.tree_map(
        lambda leaf: jax.device_put(leaf, sharding) if eqx.is_array(leaf) else leaf,
        pytree,
    )


def _generate_fake_images(
    model,
    keys: jax.Array,
    generate_fn: callable,
    fid_parallel: DataParallelConfig,
) -> jax.Array:
    """Generate fake images through the selected FID generation path.

    Args:
        model: Generative model passed to generate_fn.
        keys: PRNG keys for fake-image generation.
        generate_fn: Callable ``(model, key=...) -> image``.
        fid_parallel: Resolved FID parallel-generation config.

    Returns:
        Generated image batch as a normal JAX array.
    """
    if not fid_parallel.enabled:
        return _batched_generate(model, keys, generate_fn)

    sharded_keys = jax.device_put(keys, fid_parallel.data_sharding)
    fake_images = _parallel_batched_generate(model, sharded_keys, generate_fn)
    return jnp.asarray(jax.device_get(fake_images))


def compute_fid_metrics(
    accumulators: dict[str, "FIDAccumulator"],
    model,
    val_dataloader,
    generate_fn: callable,
    n_samples: int | None,
    gen_batch_size: int,
    key: jax.Array,
    n_real: int | None = None,
    parallel_generation: Any = None,
    data_parallel: DataParallelConfig | None = None,
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
        parallel_generation: Optional FID-specific parallel generation config.
        data_parallel: Optional resolved trainer data-parallel config used for
            parallel_generation defaults.

    Returns:
        Dict mapping accumulator names to FID scores.
    """
    fid_parallel = _resolve_fid_parallel_generation_config(
        parallel_generation=parallel_generation,
        data_parallel=data_parallel,
    )
    effective_gen_batch_size = int(gen_batch_size)
    if fid_parallel.enabled:
        effective_gen_batch_size = _effective_parallel_gen_batch_size(
            gen_batch_size=gen_batch_size,
            num_devices=fid_parallel.num_devices,
        )
        if effective_gen_batch_size != int(gen_batch_size):
            _log_parallel_gen_batch_size_adjustment(
                gen_batch_size,
                effective_gen_batch_size,
                fid_parallel.num_devices,
            )
    elif int(gen_batch_size) < 1:
        raise ValueError(
            "fid_metric.gen_batch_size must be >= 1; "
            f"got gen_batch_size={gen_batch_size}"
        )

    # --- Real-image pass (skip if all accumulators have cached stats) ---
    if (n_real == 0) or (n_real is None):
        n_real = len(val_dataloader.dataset)

    all_cached = all(acc._cached_real is not None for acc in accumulators.values())
    if not all_cached:
        for acc in accumulators.values():
            acc.reset()
        n_real_seen = 0
        pbar = tqdm(
            total=n_real, desc="FID real", leave=False, dynamic_ncols=True, unit="img"
        )
        n_real_seen = 0

        for images, _meta in val_dataloader:
            images = images.numpy()
            images = jnp.asarray(images)

            remaining = n_real - n_real_seen
            if remaining <= 0:
                break

            batch_n = min(images.shape[0], remaining)
            images = images[:batch_n]

            for acc in accumulators.values():
                acc.update(images)

            n_real_seen += batch_n
            pbar.update(batch_n)

        pbar.close()
        for acc in accumulators.values():
            mu, sigma, n = acc.statistics()
            acc._cached_real = (mu, sigma, n)
            acc.reset()

    # --- Determine n_samples ---
    if (n_samples == 0) or (n_samples == None):
        n_samples = max(acc._cached_real[2] for acc in accumulators.values())

    # --- Fake-image pass ---
    for acc in accumulators.values():
        acc.reset()

    generation_model = model
    if fid_parallel.enabled:
        generation_model = _device_put_array_leaves(model, fid_parallel.model_sharding)

    n_generated = 0

    pbar = tqdm(total=n_samples, desc="FID fake", leave=False, dynamic_ncols=True)
    while n_generated < n_samples:
        remaining = n_samples - n_generated
        chunk_size = (
            effective_gen_batch_size
            if fid_parallel.enabled
            else min(effective_gen_batch_size, remaining)
        )
        all_keys = jax.random.split(key, chunk_size + 1)
        key = all_keys[0]
        sub_keys = all_keys[1:]
        fake_images = _generate_fake_images(
            model=generation_model,
            keys=sub_keys,
            generate_fn=generate_fn,
            fid_parallel=fid_parallel,
        )
        consume_n = min(remaining, fake_images.shape[0])
        fake_images = fake_images[:consume_n]
        for acc in accumulators.values():
            acc.update(fake_images)
        n_generated += consume_n
        pbar.update(consume_n)
    pbar.close()

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
        parallel_generation: Optional FID-specific parallel generation config.
    """

    def __init__(
        self,
        accumulators: dict[str, "FIDAccumulator"],
        generate_fn: callable,
        n_samples: int | None = None,
        gen_batch_size: int = 64,
        n_real: int | None = None,
        parallel_generation: Any = None,
    ):
        self.accumulators = accumulators
        self.generate_fn = generate_fn
        self.n_samples = n_samples
        self.gen_batch_size = gen_batch_size
        self.n_real = n_real
        self.parallel_generation = parallel_generation

    def __call__(
        self,
        model,
        val_dataloader,
        key: jax.Array,
        data_parallel: DataParallelConfig | None = None,
    ) -> dict[str, float]:
        """Compute FID scores for all accumulators.

        Args:
            model:          Generative model passed to ``generate_fn``.
            val_dataloader: Iterable yielding ``(images, meta)`` tuples.
            key:            PRNG key for generation.
            data_parallel:  Optional resolved trainer data-parallel config used
                for parallel_generation defaults.

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
            parallel_generation=self.parallel_generation,
            data_parallel=data_parallel,
        )


def _safe_divide(num, den, eps=1e-12):
    return num / jnp.maximum(den, eps)


def _prepare_image(img):
    """
    img: array of shape (1, N, N)
    Returns:
        J: non-negative 2D image of shape (N, N)
    """
    assert img.ndim == 3 and img.shape[0] == 1, "Expected image of shape (1, N, N)"
    # J = jnp.maximum(img[0], 0.0)
    J = (img[0] + 1.0) / 2.0
    return J


def _coordinate_grids(N):
    """
    Returns X, Y coordinate grids of shape (N, N).
    X is horizontal (column index), Y is vertical (row index).
    """
    y = jnp.arange(N)
    x = jnp.arange(N)
    Y, X = jnp.meshgrid(y, x, indexing="ij")
    return X, Y


def centroid(img, eps=1e-12):
    """
    Intensity-weighted centroid.

    img: shape (1, N, N)
    Returns:
        xc, yc
    """
    # J = _prepare_image(img)
    # N = J.shape[0]
    # X, Y = _coordinate_grids(N)
    # total = jnp.sum(J)
    # xc = _safe_divide(jnp.sum(J * X), total, eps)
    # yc = _safe_divide(jnp.sum(J * Y), total, eps)
    # return xc, yc
    return (256, 256)


def _radii_and_sorted_intensity(J, xc, yc):
    """
    Flattened radii and intensities sorted by radius.
    """
    N = J.shape[0]
    X, Y = _coordinate_grids(N)
    r = jnp.sqrt((X - xc) ** 2 + (Y - yc) ** 2)

    r_flat = r.reshape(-1)
    J_flat = J.reshape(-1)

    order = jnp.argsort(r_flat)
    r_sorted = r_flat[order]
    J_sorted = J_flat[order]
    return r_sorted, J_sorted


def radius_at_fraction(img, frac, eps=1e-12):
    """
    Radius enclosing a given fraction of total non-negative intensity.

    img: shape (1, N, N)
    frac: scalar in [0, 1]
    """
    J = _prepare_image(img)
    # xc, yc = centroid(img, eps=eps)
    mid_idx = img.shape[0] // 2
    xc, yc = (mid_idx, mid_idx)
    r_sorted, J_sorted = _radii_and_sorted_intensity(J, xc, yc)

    cumulative = jnp.cumsum(J_sorted)
    total = jnp.sum(J_sorted)
    target = frac * total

    idx = jnp.searchsorted(cumulative, target, side="left")
    idx = jnp.clip(idx, 0, r_sorted.shape[0] - 1)
    return r_sorted[idx]


def half_light_radius(img, eps=1e-12):
    return radius_at_fraction(img, 0.5, eps=eps)


def concentration(img, eps=1e-12):
    """
    C = 5 log10(r80 / r20)
    """
    r20 = radius_at_fraction(img, 0.2, eps=eps)
    r80 = radius_at_fraction(img, 0.8, eps=eps)
    C = 5.0 * jnp.log10(_safe_divide(r80, r20, eps))
    return C, r20, r80


def second_moments(img, eps=1e-12):
    """
    Intensity-weighted second central moments.

    Returns:
        Mxx, Myy, Mxy
    """
    J = _prepare_image(img)
    N = J.shape[0]
    X, Y = _coordinate_grids(N)
    # xc, yc = centroid(img, eps=eps)
    mid_idx = img.shape[0] // 2
    xc, yc = (mid_idx, mid_idx)

    total = jnp.sum(J)

    dx = X - xc
    dy = Y - yc

    Mxx = _safe_divide(jnp.sum(J * dx * dx), total, eps)
    Myy = _safe_divide(jnp.sum(J * dy * dy), total, eps)
    Mxy = _safe_divide(jnp.sum(J * dx * dy), total, eps)

    return Mxx, Myy, Mxy


def shape_metrics(img, eps=1e-12):
    """
    Axis ratio, ellipticity, and position angle from second moments.

    Returns:
        q : axis ratio b/a
        e : ellipticity = 1 - q
        theta : position angle in radians
        a, b : sqrt(eigenvalues)
    """
    Mxx, Myy, Mxy = second_moments(img, eps=eps)

    cov = jnp.array([[Mxx, Mxy], [Mxy, Myy]])
    eigvals, eigvecs = jnp.linalg.eigh(cov)

    # eigh returns ascending eigenvalues
    lam2, lam1 = eigvals[0], eigvals[1]  # lam1 >= lam2
    a = jnp.sqrt(jnp.maximum(lam1, 0.0))
    b = jnp.sqrt(jnp.maximum(lam2, 0.0))

    q = _safe_divide(b, a, eps)
    e = 1.0 - q

    # Position angle of major axis
    # Equivalent formula:
    # theta = 0.5 * arctan2(2 Mxy, Mxx - Myy)
    theta = 0.5 * jnp.arctan2(2.0 * Mxy, Mxx - Myy)

    return q, e, theta, a, b


def _rotate_180_about_center(J, xc, yc, order=1):
    """
    Rotate a 2D image J by 180 degrees about (xc, yc), using interpolation.

    Returns rotated image of same shape.
    """
    N = J.shape[0]
    X, Y = _coordinate_grids(N)

    # 180-degree rotation about (xc, yc):
    # x' = 2 xc - x
    # y' = 2 yc - y
    X_src = 2.0 * xc - X
    Y_src = 2.0 * yc - Y

    coords = jnp.stack([Y_src, X_src], axis=0)  # map_coordinates expects (row, col)
    J_rot = map_coordinates(J, coords, order=order, mode="constant", cval=0.0)
    return J_rot


def asymmetry(img, eps=1e-12):
    """
    A = sum |J - J_180| / sum |J|

    Uses 180-degree rotation about the intensity-weighted centroid.
    """
    J = _prepare_image(img)
    mid_idx = img.shape[0] // 2
    xc, yc = (mid_idx, mid_idx)
    J_rot = _rotate_180_about_center(J, xc, yc, order=1)

    num = jnp.sum(jnp.abs(J - J_rot))
    den = jnp.sum(jnp.abs(J))
    A = _safe_divide(num, den, eps)
    return A


def morphology_metrics(img, eps=1e-12):
    """
    Compute a set of morphology metrics for a single image of shape (1, N, N).

    Returns a dict of JAX scalars.
    """
    mid_idx = img.shape[0] // 2
    xc, yc = (mid_idx, mid_idx)
    r50 = half_light_radius(img, eps=eps)
    C, r20, r80 = concentration(img, eps=eps)
    q, e, theta, a, b = shape_metrics(img, eps=eps)
    A = asymmetry(img, eps=eps)

    return {
        "xc": xc,
        "yc": yc,
        "r20": r20,
        "r50": r50,
        "r80": r80,
        "concentration": C,
        "axis_ratio": q,
        "ellipticity": e,
        "position_angle": theta,
        "a": a,
        "b": b,
        "asymmetry": A,
    }
