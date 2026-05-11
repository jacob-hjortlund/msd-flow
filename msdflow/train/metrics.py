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

from msdflow.flow.clr import project_channel_mean_zero
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
        return (pred - x_t) / jnp.clip(1.0 - t_, min=0.05)
    return pred


def _flow_matching_per_sample_loss_values(
    model,
    x_t: jnp.ndarray,
    u_t: jnp.ndarray,
    t: jnp.ndarray,
    cond: jnp.ndarray,
    cond_mask: jnp.ndarray,
    key: jax.Array,
    project_velocity: bool = False,
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
        project_velocity: If ``True``, project predicted velocities to zero
            spatial mean independently per sample and channel before computing
            the loss. Target velocities are not projected. When used under
            JAX JIT or Equinox JIT, this must be a Python/static bool rather
            than a traced JAX value.

    Returns:
        Mean squared velocity error for each sample, shape ``(B,)``.
    """
    pred = eqx.filter_vmap(model)(t, x_t, cond, cond_mask, key)
    v_t = _to_velocity(pred, x_t, t, model.prediction_type)
    if project_velocity:
        v_t = project_channel_mean_zero(v_t)
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
    project_velocity: bool = False,
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
        key: Per-sample PRNG keys with leading shape ``(B,)``.
        project_velocity: If ``True``, project predicted velocities to zero
            spatial mean independently per sample and channel before computing
            the loss. Target velocities are not projected. When used under
            JAX JIT or Equinox JIT, this must be a Python/static bool rather
            than a traced JAX value.

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
            project_velocity=project_velocity,
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
    project_velocity: bool = False,
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
        project_velocity: If ``True``, project predicted velocities to zero
            spatial mean independently per sample and channel before computing
            the loss. Target velocities are not projected. When used under
            JAX JIT or Equinox JIT, this must be a Python/static bool rather
            than a traced JAX value.

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
        project_velocity=project_velocity,
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
        """Return mean loss per time bin.

        Returns:
            Mean loss per bin, with ``NaN`` for empty bins.
        """
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
    project_velocity: bool = False,
):
    """Return a JIT-compiled step for time-binned flow-matching loss.

    Args:
        num_bins: Number of equal-width bins over ``[0, 1]``.
        data_parallel: Optional data-parallel runtime configuration.
        project_velocity: If ``True``, project predicted velocities to zero
            spatial mean independently per sample and channel before binning
            losses. Target velocities are not projected. This is captured by
            the returned Equinox JIT step and must be a Python/static bool,
            not a traced JAX value.

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
    @eqx.debug.assert_max_traces(max_traces=1)
    def time_binned_loss_step(
        model,
        x_t: jax.Array,
        u_t: jax.Array,
        t: jax.Array,
        cond: jax.Array,
        cond_mask: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Aggregate one prepared batch into time-binned loss statistics.

        Args:
            model: Flow-matching model evaluated on the prepared batch.
            x_t: Interpolated samples.
            u_t: Target velocity fields.
            t: Per-sample times.
            cond: Conditioning vectors.
            cond_mask: Per-sample conditioning mask.
            key: Per-sample dropout keys.

        Returns:
            Tuple of ``(loss_sums, counts)`` arrays with shape ``(num_bins,)``.
            Losses use the factory's static ``project_velocity`` setting.
        """
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
            project_velocity=project_velocity,
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


def _extract_batch_impl(encoder, images):
    """Encode a batch of images into feature vectors.

    Args:
        encoder: Callable mapping a single image (C, H, W) to features (D,).
        images:  Batch of images, shape (B, C, H, W).

    Returns:
        Feature matrix of shape (B, D).
    """
    return jax.vmap(encoder)(images)


def _make_extract_batch_step():
    """Return a per-accumulator JIT guard for feature extraction.

    Returns:
        JIT-compiled feature extraction callable guarded against repeated
        retracing for one accumulator.
    """

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=2)
    def extract_batch(encoder, images):
        return _extract_batch_impl(encoder, images)

    return extract_batch


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
        self._extract_batch = _make_extract_batch_step()
        self._sum_features = None  # np.ndarray (D,)
        self._sum_outer = None  # np.ndarray (D, D)
        self._n = 0
        self._cached_real = None  # set by compute_fid_metrics

    def update(self, images: jax.Array, n_images: int | None = None) -> None:
        """Encode a batch and update running accumulators.

        Args:
            images: Batch of images, shape (B, C, H, W).
            n_images: Optional number of leading encoded images to accumulate.
                ``None`` accumulates the whole batch.
        """
        features = np.asarray(self._extract_batch(self.encoder, images))  # (B, D)
        if n_images is not None:
            features = features[: int(n_images)]

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

        Does not clear cached real-image statistics (``_cached_real``) or
        replace the per-accumulator feature-extraction trace guard.
        """
        self._sum_features = None
        self._sum_outer = None
        self._n = 0


@dataclass
class FIDConditionCache:
    """Cached validation conditions shared by all FID accumulators.

    Attributes:
        conditions: Cached validation metadata with shape ``(N, cond_dim)``.
        n_real: Number of real validation rows represented by ``conditions``.
    """

    conditions: np.ndarray | None = None
    n_real: int | None = None

    def matches(self, n_real: int) -> bool:
        """Return whether the cache contains conditions for ``n_real`` rows.

        Args:
            n_real: Required real validation row count.

        Returns:
            ``True`` when this cache can supply conditions for the requested
            real-image population.
        """
        return (
            self.conditions is not None
            and self.n_real == int(n_real)
            and self.conditions.shape[0] >= int(n_real)
        )


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


def _generate_fn_guidance_scale(generate_fn: callable) -> float:
    """Return the configured guidance scale for a generation callable.

    Args:
        generate_fn: Generation function or ``functools.partial``.

    Returns:
        Configured guidance scale, defaulting to ``1.0`` when not present.
    """
    keywords = getattr(generate_fn, "keywords", None)
    if isinstance(keywords, Mapping) and "guidance_scale" in keywords:
        return float(keywords["guidance_scale"])
    return float(getattr(generate_fn, "guidance_scale", 1.0))


def _fid_requires_validation_conditions(generate_fn: callable) -> bool:
    """Return whether FID generation must pass validation conditions.

    Args:
        generate_fn: Generation function or ``functools.partial``.

    Returns:
        ``True`` when classifier-free guidance is enabled for FID generation.
    """
    return _generate_fn_guidance_scale(generate_fn) != 1.0


def _batched_generate_impl(model, keys, conditions, generate_fn):
    """Generate a fake-image batch with optional validation conditions."""
    if conditions is None:
        return jax.vmap(lambda key: generate_fn(model, key=key))(keys)
    return jax.vmap(lambda key, cond: generate_fn(model, key=key, cond=cond))(
        keys,
        conditions,
    )


def _make_batched_generate_step():
    """Return a per-FID-call JIT guard for serial generation.

    Returns:
        JIT-compiled generation callable guarded against retracing within one
        FID metric computation.
    """

    @eqx.filter_jit(donate="all-except-first")
    @eqx.debug.assert_max_traces(max_traces=1)
    def batched_generate(model, keys, conditions, generate_fn):
        return _batched_generate_impl(model, keys, conditions, generate_fn)

    return batched_generate


def _parallel_batched_generate_impl(model, keys, conditions, generate_fn):
    """Generate a fake-image batch from sharded keys and optional conditions."""
    if conditions is None:
        return jax.vmap(lambda key: generate_fn(model, key=key))(keys)
    return jax.vmap(lambda key, cond: generate_fn(model, key=key, cond=cond))(
        keys,
        conditions,
    )


def _make_parallel_batched_generate_step():
    """Return a per-FID-call JIT guard for parallel generation.

    Returns:
        JIT-compiled parallel generation callable guarded against retracing
        within one FID metric computation.
    """

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def parallel_batched_generate(model, keys, conditions, generate_fn):
        return _parallel_batched_generate_impl(model, keys, conditions, generate_fn)

    return parallel_batched_generate


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
    conditions: jax.Array | None,
    generate_fn: callable,
    fid_parallel: DataParallelConfig,
    batched_generate: callable,
    parallel_batched_generate: callable,
) -> jax.Array:
    """Generate fake images through the selected FID generation path.

    Args:
        model: Generative model passed to generate_fn.
        keys: PRNG keys for fake-image generation.
        conditions: Optional validation conditions for CFG generation.
        generate_fn: Callable ``(model, key=..., cond=...) -> image`` when
            conditions are supplied, otherwise ``(model, key=...) -> image``.
        fid_parallel: Resolved FID parallel-generation config.
        batched_generate: Per-call serial generation step.
        parallel_batched_generate: Per-call parallel generation step.

    Returns:
        Generated image batch as a normal JAX array.
    """
    if not fid_parallel.enabled:
        return batched_generate(model, keys, conditions, generate_fn)

    sharded_keys = jax.device_put(keys, fid_parallel.data_sharding)
    sharded_conditions = None
    if conditions is not None:
        sharded_conditions = jax.device_put(conditions, fid_parallel.data_sharding)
    fake_images = parallel_batched_generate(
        model,
        sharded_keys,
        sharded_conditions,
        generate_fn,
    )
    return jnp.asarray(jax.device_get(fake_images))


def _numpy_batch(batch: Any) -> np.ndarray:
    """Convert a tensor-like batch to a NumPy array.

    Args:
        batch: PyTorch tensor, NumPy array, or array-like object.

    Returns:
        NumPy representation of ``batch``.
    """
    if hasattr(batch, "numpy"):
        return batch.numpy()
    return np.asarray(batch)


def _validate_condition_batch(meta: np.ndarray) -> np.ndarray:
    """Validate and normalize a validation metadata batch for CFG FID.

    Args:
        meta: Metadata batch from the validation dataloader.

    Returns:
        Metadata array with shape ``(B, cond_dim)``.

    Raises:
        ValueError: If metadata is empty.
    """
    if meta.ndim == 1:
        meta = meta[:, None]
    if meta.shape[-1] == 0:
        raise ValueError("CFG FID requires validation metadata, but meta has width 0")
    return meta


def _limited_validation_chunks(
    val_dataloader,
    n_images: int,
    chunk_size: int,
    include_conditions: bool = False,
):
    """Yield validation images and optional conditions in stable-size chunks.

    Args:
        val_dataloader: Iterable yielding ``(images, meta)`` tuples.
        n_images: Maximum number of leading images to consume.
        chunk_size: Leading dimension for yielded arrays.
        include_conditions: Whether to collect validation metadata.

    Yields:
        Tuples of ``(images, conditions, n_valid)``. ``conditions`` is ``None``
        unless ``include_conditions`` is true. The final chunk is zero-padded.

    Raises:
        ValueError: If ``chunk_size`` is less than one or CFG metadata is empty.
    """
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    remaining = int(n_images)
    image_buffer = None
    condition_buffer = None
    for images, meta in val_dataloader:
        if remaining <= 0:
            break

        image_batch = _numpy_batch(images)
        take = min(image_batch.shape[0], remaining)
        if take <= 0:
            continue

        image_batch = image_batch[:take]
        remaining -= take
        if image_buffer is None:
            image_buffer = image_batch
        else:
            image_buffer = np.concatenate((image_buffer, image_batch), axis=0)

        if include_conditions:
            condition_batch = _validate_condition_batch(_numpy_batch(meta))[:take]
            if condition_buffer is None:
                condition_buffer = condition_batch
            else:
                condition_buffer = np.concatenate(
                    (condition_buffer, condition_batch),
                    axis=0,
                )

        while image_buffer is not None and image_buffer.shape[0] >= chunk_size:
            if include_conditions:
                conditions = jnp.asarray(condition_buffer[:chunk_size])
                condition_buffer = condition_buffer[chunk_size:]
                if condition_buffer.shape[0] == 0:
                    condition_buffer = None
            else:
                conditions = None
            yield jnp.asarray(image_buffer[:chunk_size]), conditions, chunk_size
            image_buffer = image_buffer[chunk_size:]
            if image_buffer.shape[0] == 0:
                image_buffer = None

    if image_buffer is None or image_buffer.shape[0] == 0:
        return

    n_valid = int(image_buffer.shape[0])
    pad_width = [(0, chunk_size - n_valid)]
    pad_width.extend((0, 0) for _ in range(image_buffer.ndim - 1))
    padded_images = np.pad(image_buffer, pad_width, mode="constant")

    if include_conditions:
        condition_pad_width = [(0, chunk_size - n_valid)]
        condition_pad_width.extend((0, 0) for _ in range(condition_buffer.ndim - 1))
        conditions = jnp.asarray(
            np.pad(condition_buffer, condition_pad_width, mode="constant")
        )
    else:
        conditions = None
    yield jnp.asarray(padded_images), conditions, n_valid


def _limited_image_chunks(
    val_dataloader,
    n_images: int,
    chunk_size: int,
):
    """Yield validation images in stable-size chunks.

    Args:
        val_dataloader: Iterable yielding ``(images, meta)`` tuples.
        n_images: Maximum number of leading images to consume.
        chunk_size: Leading dimension for yielded arrays.

    Yields:
        Tuples of ``(images, n_valid)`` where ``images`` has leading size
        ``chunk_size`` and ``n_valid`` is the number of rows that should
        contribute to statistics.
    """
    for images, _conditions, n_valid in _limited_validation_chunks(
        val_dataloader,
        n_images=n_images,
        chunk_size=chunk_size,
        include_conditions=False,
    ):
        yield images, n_valid


def _condition_chunk(
    conditions: np.ndarray,
    start: int,
    n_samples: int,
    chunk_size: int,
) -> jax.Array:
    """Return a stable-size condition chunk from cached validation conditions.

    Args:
        conditions: Cached validation metadata with shape ``(N, cond_dim)``.
        start: Starting row for this chunk.
        n_samples: Total number of fake samples requested.
        chunk_size: Stable generation chunk size.

    Returns:
        Condition chunk with leading dimension ``chunk_size``.

    Raises:
        ValueError: If not enough conditions are cached.
    """
    valid_n = min(int(chunk_size), int(n_samples) - int(start))
    if conditions.shape[0] < start + valid_n:
        raise ValueError(
            "CFG FID requires cached validation conditions for "
            f"n_samples={n_samples}, but only {conditions.shape[0]} are available"
        )
    chunk = conditions[start : start + valid_n]
    if valid_n == chunk_size:
        return jnp.asarray(chunk)
    pad_width = [(0, chunk_size - valid_n)]
    pad_width.extend((0, 0) for _ in range(chunk.ndim - 1))
    return jnp.asarray(np.pad(chunk, pad_width, mode="constant"))


def compute_fid_metrics(
    accumulators: dict[str, "FIDAccumulator"],
    model,
    val_dataloader,
    generate_fn: callable,
    batched_generate_wrapper: callable,
    parallel_batched_generate_wrapper: callable,
    n_samples: int | None,
    gen_batch_size: int,
    key: jax.Array,
    n_real: int | None = None,
    parallel_generation: Any = None,
    data_parallel: DataParallelConfig | None = None,
    condition_cache: FIDConditionCache | None = None,
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
        generate_fn:     ``(model, key=...) -> jax.Array`` of shape ``(C, H, W)``
            for unconditional generation, or ``(model, key=..., cond=...)``
            when classifier-free guidance conditions are required.
        n_samples:       Number of fake images. ``None`` matches real count.
        gen_batch_size:  Images generated and encoded per chunk.
        key:             PRNG key for generation.
        n_real:          Maximum number of real images to use from
            ``val_dataloader``. ``None`` (default) uses the full dataset.
        parallel_generation: Optional FID-specific parallel generation config.
        data_parallel: Optional resolved trainer data-parallel config used for
            parallel_generation defaults.
        condition_cache: Optional local validation-condition cache for direct
            callers that reuse cached real statistics.

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

    requires_conditions = _fid_requires_validation_conditions(generate_fn)
    if condition_cache is None:
        condition_cache = FIDConditionCache()

    # --- Real-image pass (skip if all accumulators have cached stats) ---
    all_cached = all(acc._cached_real is not None for acc in accumulators.values())
    if (n_real == 0) or (n_real is None):
        if all_cached:
            n_real = max(acc._cached_real[2] for acc in accumulators.values())
        else:
            n_real = len(val_dataloader.dataset)
    n_real = int(n_real)

    condition_chunks = []
    if not all_cached:
        for acc in accumulators.values():
            acc.reset()
        pbar = tqdm(
            total=n_real, desc="FID real", leave=False, dynamic_ncols=True, unit="img"
        )
        n_real_seen = 0

        for images, conditions, batch_n in _limited_validation_chunks(
            val_dataloader,
            n_images=n_real,
            chunk_size=effective_gen_batch_size,
            include_conditions=requires_conditions,
        ):
            for acc in accumulators.values():
                acc.update(images, n_images=batch_n)
            if requires_conditions:
                condition_chunks.append(np.asarray(conditions)[:batch_n])

            n_real_seen += batch_n
            pbar.update(batch_n)

        pbar.close()
        for acc in accumulators.values():
            mu, sigma, n = acc.statistics()
            acc._cached_real = (mu, sigma, n)
            acc.reset()
        if requires_conditions:
            if n_real_seen < n_real:
                raise ValueError(
                    "CFG FID requires validation conditions for "
                    f"n_real={n_real}, but only {n_real_seen} rows were available"
                )
            condition_cache.conditions = np.concatenate(condition_chunks, axis=0)
            condition_cache.n_real = n_real

    # --- Determine n_samples ---
    if (n_samples == 0) or (n_samples is None):
        n_samples = max(acc._cached_real[2] for acc in accumulators.values())
    n_samples = int(n_samples)
    if requires_conditions:
        if n_samples > n_real:
            raise ValueError(
                "CFG FID requires n_samples <= n_real when using validation "
                f"conditions, got n_samples={n_samples}, n_real={n_real}"
            )
        if not condition_cache.matches(n_real):
            condition_chunks = []
            n_condition_seen = 0
            for _images, conditions, batch_n in _limited_validation_chunks(
                val_dataloader,
                n_images=n_real,
                chunk_size=effective_gen_batch_size,
                include_conditions=True,
            ):
                condition_chunks.append(np.asarray(conditions)[:batch_n])
                n_condition_seen += batch_n
            if n_condition_seen < n_real:
                raise ValueError(
                    "CFG FID requires validation conditions for "
                    f"n_real={n_real}, but only {n_condition_seen} rows were available"
                )
            condition_cache.conditions = np.concatenate(condition_chunks, axis=0)
            condition_cache.n_real = n_real

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
        chunk_size = effective_gen_batch_size
        all_keys = jax.random.split(key, chunk_size + 1)
        key = all_keys[0]
        sub_keys = all_keys[1:]
        condition_chunk = None
        if requires_conditions:
            condition_chunk = _condition_chunk(
                condition_cache.conditions,
                start=n_generated,
                n_samples=n_samples,
                chunk_size=chunk_size,
            )
        fake_images = _generate_fake_images(
            model=generation_model,
            keys=sub_keys,
            conditions=condition_chunk,
            generate_fn=generate_fn,
            fid_parallel=fid_parallel,
            batched_generate=batched_generate_wrapper,
            parallel_batched_generate=parallel_batched_generate_wrapper,
        )
        consume_n = min(remaining, fake_images.shape[0])
        for acc in accumulators.values():
            acc.update(fake_images, n_images=consume_n)
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
        self.batched_generate_wrapper = _make_batched_generate_step()
        self.parallel_batched_generate_wrapper = _make_parallel_batched_generate_step()
        self.condition_cache = FIDConditionCache()

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
            batched_generate_wrapper=self.batched_generate_wrapper,
            parallel_batched_generate_wrapper=self.parallel_batched_generate_wrapper,
            n_samples=self.n_samples,
            gen_batch_size=self.gen_batch_size,
            key=key,
            n_real=self.n_real,
            parallel_generation=self.parallel_generation,
            data_parallel=data_parallel,
            condition_cache=self.condition_cache,
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
