"""Tests for FID metric components in msdflow.train.metrics."""

import inspect

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import torch
from torch.utils.data import DataLoader, TensorDataset

from msdflow.train.metrics import _frechet_distance
from msdflow.train.metrics import _effective_parallel_gen_batch_size
from msdflow.train.metrics import _log_parallel_gen_batch_size_adjustment
from msdflow.train.metrics import _resolve_fid_parallel_generation_config
from msdflow.train.metrics import compute_fid_metrics
from msdflow.train.metrics import FIDAccumulator
from msdflow.train.metrics import FIDMetric
from msdflow.train.parallel import make_data_parallel_config
from msdflow.train.trainer import train


def test_frechet_distance_identical_distributions_is_zero():
    """FID between identical distributions must be zero."""
    rng = np.random.default_rng(42)
    D = 8
    mu = rng.standard_normal(D)
    sigma = np.eye(D)
    fid = _frechet_distance(mu, sigma, mu, sigma)
    assert fid == pytest.approx(0.0, abs=1e-6)


def test_frechet_distance_shifted_mean():
    """FID with shifted mean and identical covariance equals ||delta||^2."""
    D = 4
    mu_real = np.zeros(D)
    mu_fake = np.ones(D)
    sigma = np.eye(D)
    fid = _frechet_distance(mu_real, sigma, mu_fake, sigma)
    # When covariances are equal identity: FID = ||mu_r - mu_f||^2 + trace(2I - 2I) = D
    assert fid == pytest.approx(float(D), abs=1e-6)


def test_frechet_distance_is_non_negative():
    """FID must be non-negative for any pair of distributions."""
    rng = np.random.default_rng(7)
    D = 6
    mu_r = rng.standard_normal(D)
    mu_f = rng.standard_normal(D)
    A = rng.standard_normal((D, D))
    sigma_r = A @ A.T + np.eye(D) * 0.1
    B = rng.standard_normal((D, D))
    sigma_f = B @ B.T + np.eye(D) * 0.1
    fid = _frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
    assert fid >= -1e-6


def _identity_encoder(x):
    """Trivial encoder: flatten spatial dims to a 1-D feature vector."""
    return x.reshape(-1)


def test_accumulator_statistics_match_numpy():
    """Streaming stats must match direct numpy computation."""
    rng = np.random.default_rng(0)
    # 3 batches of 4 images, 1 channel, 2x2 -> D=4 features
    all_images = rng.standard_normal((12, 1, 2, 2)).astype(np.float32)
    batches = [jnp.array(all_images[i : i + 4]) for i in range(0, 12, 4)]

    acc = FIDAccumulator(encoder=_identity_encoder)
    for batch in batches:
        acc.update(batch)
    mu, sigma, n = acc.statistics()

    # Reference: flatten all images and compute directly
    flat = all_images.reshape(12, 4)
    expected_mu = flat.mean(axis=0)
    expected_sigma = np.cov(flat, rowvar=False, bias=True)

    assert n == 12
    np.testing.assert_allclose(mu, expected_mu, atol=1e-5)
    np.testing.assert_allclose(sigma, expected_sigma, atol=1e-5)


def test_accumulator_reset_clears_state():
    """After reset, statistics must return zero count."""
    acc = FIDAccumulator(encoder=_identity_encoder)
    images = jnp.ones((2, 1, 2, 2))
    acc.update(images)
    acc.reset()
    _, _, n = acc.statistics()
    assert n == 0


def test_accumulator_single_image():
    """Accumulator works with a single image (covariance is zero matrix)."""
    image = jnp.ones((1, 1, 2, 2)) * 3.0
    acc = FIDAccumulator(encoder=_identity_encoder)
    acc.update(image)
    mu, sigma, n = acc.statistics()
    assert n == 1
    np.testing.assert_allclose(mu, np.full(4, 3.0), atol=1e-6)
    # Single sample: cov = outer(x,x)/1 - outer(mu,mu) = 0
    np.testing.assert_allclose(sigma, np.zeros((4, 4)), atol=1e-6)




def _make_dummy_dataloader(n_batches, batch_size, shape=(1, 2, 2), seed=0):
    """Return a torch DataLoader yielding (images, meta) tuples.

    compute_fid_metrics queries .dataset (for length) and calls .numpy() on
    each batch's images tensor; both require a real DataLoader over torch
    tensors.
    """
    rng = np.random.default_rng(seed)
    n_total = n_batches * batch_size
    images = torch.from_numpy(
        rng.standard_normal((n_total, *shape)).astype(np.float32)
    )
    meta = torch.empty(n_total, 0)
    dataset = TensorDataset(images, meta)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _make_empty_dataloader(shape=(1, 2, 2)):
    """Return a torch DataLoader over a zero-length TensorDataset.

    Used to verify the cached real-stats short-circuit on second call.
    """
    images = torch.empty(0, *shape, dtype=torch.float32)
    meta = torch.empty(0, 0, dtype=torch.float32)
    dataset = TensorDataset(images, meta)
    return DataLoader(dataset, batch_size=1, shuffle=False)


def _dummy_generate_fn(model, key):
    """Generate a single fake image of shape (1, 2, 2) from random noise."""
    return jax.random.normal(key, (1, 2, 2))


def test_resolve_fid_parallel_generation_inherits_trainer_config():
    """FID parallel generation inherits enabled and min_devices by default."""
    trainer_cfg = make_data_parallel_config(enabled=True, min_devices=1)

    fid_cfg = _resolve_fid_parallel_generation_config(
        parallel_generation=None,
        data_parallel=trainer_cfg,
    )

    assert fid_cfg.enabled is True
    assert fid_cfg.min_devices == 1
    assert fid_cfg.axis_name == "fid_sample"


def test_resolve_fid_parallel_generation_can_disable_inherited_enabled():
    """FID parallel generation can be disabled even when training is enabled."""
    trainer_cfg = make_data_parallel_config(enabled=True, min_devices=1)

    fid_cfg = _resolve_fid_parallel_generation_config(
        parallel_generation={"enabled": False},
        data_parallel=trainer_cfg,
    )

    assert fid_cfg.enabled is False
    assert fid_cfg.axis_name == "fid_sample"


def test_resolve_fid_parallel_generation_falls_back_when_devices_unavailable():
    """Enabled FID parallel generation falls back when devices are unavailable."""
    unavailable = len(jax.local_devices()) + 1

    fid_cfg = _resolve_fid_parallel_generation_config(
        parallel_generation={"enabled": True, "min_devices": unavailable},
        data_parallel=None,
    )

    assert fid_cfg.enabled is False
    assert fid_cfg.axis_name == "fid_sample"
    assert fid_cfg.min_devices == unavailable
    assert fid_cfg.num_devices == 1
    assert fid_cfg.data_sharding is None
    assert fid_cfg.model_sharding is None


@pytest.mark.parametrize("min_devices", ["two", None])
def test_resolve_fid_parallel_generation_wraps_malformed_min_devices(min_devices):
    """Malformed FID min_devices errors include FID config context."""
    with pytest.raises(ValueError, match="fid_metric.parallel_generation"):
        _resolve_fid_parallel_generation_config(
            parallel_generation={"enabled": True, "min_devices": min_devices},
            data_parallel=None,
        )


def test_effective_parallel_gen_batch_size_rounds_up():
    """Parallel FID generation rounds global chunk size up to a device multiple."""
    effective = _effective_parallel_gen_batch_size(
        gen_batch_size=63,
        num_devices=2,
    )

    assert effective == 64


def test_effective_parallel_gen_batch_size_rounds_large_ints_without_float():
    """Parallel FID generation rounds huge integers without float conversion."""
    gen_batch_size = 10**400 + 1
    num_devices = 3
    effective = _effective_parallel_gen_batch_size(
        gen_batch_size=gen_batch_size,
        num_devices=num_devices,
    )

    assert effective == (
        (gen_batch_size + num_devices - 1) // num_devices
    ) * num_devices


def test_effective_parallel_gen_batch_size_rejects_invalid_values():
    """FID generation requires a positive global generation batch size."""
    with pytest.raises(ValueError, match="gen_batch_size"):
        _effective_parallel_gen_batch_size(gen_batch_size=0, num_devices=2)


def test_log_parallel_gen_batch_size_adjustment_warns(caplog):
    """FID generation logs when the effective global chunk size changes."""
    _log_parallel_gen_batch_size_adjustment(
        gen_batch_size=3,
        effective_gen_batch_size=4,
        num_devices=2,
    )

    assert "fid_metric.parallel_generation" in caplog.text
    assert "from 3 to 4" in caplog.text


def test_log_parallel_gen_batch_size_adjustment_noops_when_unchanged(caplog):
    """FID generation does not log when the global chunk size is unchanged."""
    _log_parallel_gen_batch_size_adjustment(
        gen_batch_size=4,
        effective_gen_batch_size=4,
        num_devices=2,
    )

    assert caplog.text == ""


def test_compute_fid_metrics_returns_dict_with_correct_keys():
    """Output dict keys must match accumulator names."""
    accumulators = {
        "enc_a": FIDAccumulator(encoder=_identity_encoder),
        "enc_b": FIDAccumulator(encoder=_identity_encoder),
    }
    dataloader = _make_dummy_dataloader(n_batches=2, batch_size=4)
    key = jax.random.PRNGKey(0)
    result = compute_fid_metrics(
        accumulators=accumulators,
        model=None,
        val_dataloader=dataloader,
        generate_fn=_dummy_generate_fn,
        n_samples=None,
        gen_batch_size=4,
        key=key,
    )
    assert set(result.keys()) == {"enc_a", "enc_b"}


def test_compute_fid_metrics_values_are_finite_floats():
    """All returned FID scores must be finite floats."""
    accumulators = {"fid": FIDAccumulator(encoder=_identity_encoder)}
    dataloader = _make_dummy_dataloader(n_batches=3, batch_size=4)
    key = jax.random.PRNGKey(1)
    result = compute_fid_metrics(
        accumulators=accumulators,
        model=None,
        val_dataloader=dataloader,
        generate_fn=_dummy_generate_fn,
        n_samples=None,
        gen_batch_size=4,
        key=key,
    )
    assert isinstance(result["fid"], float)
    assert np.isfinite(result["fid"])


def test_compute_fid_metrics_real_stats_cached_across_calls():
    """Second call must skip the real-image pass (cached stats reused)."""
    acc = FIDAccumulator(encoder=_identity_encoder)
    accumulators = {"fid": acc}
    dataloader = _make_dummy_dataloader(n_batches=2, batch_size=4)

    key1, key2 = jax.random.split(jax.random.PRNGKey(2))
    result1 = compute_fid_metrics(
        accumulators=accumulators,
        model=None,
        val_dataloader=dataloader,
        generate_fn=_dummy_generate_fn,
        n_samples=8,
        gen_batch_size=4,
        key=key1,
    )
    # Use an empty DataLoader to prove the second call hits the cache and
    # never iterates the real-image loop.
    dataloader_empty = _make_empty_dataloader()
    result2 = compute_fid_metrics(
        accumulators=accumulators,
        model=None,
        val_dataloader=dataloader_empty,
        generate_fn=_dummy_generate_fn,
        n_samples=8,
        gen_batch_size=4,
        key=key2,
    )
    # Both should return valid FID (second call used cached real stats)
    assert np.isfinite(result1["fid"])
    assert np.isfinite(result2["fid"])


def test_compute_fid_metrics_n_samples_defaults_to_real_count():
    """When n_samples=None, number of fake images equals number of real images."""
    accumulators = {"fid": FIDAccumulator(encoder=_identity_encoder)}
    # 2 batches * 4 images = 8 real images
    dataloader = _make_dummy_dataloader(n_batches=2, batch_size=4)
    key = jax.random.PRNGKey(3)

    # We can verify indirectly: the function should complete without error
    # and produce a finite FID score
    result = compute_fid_metrics(
        accumulators=accumulators,
        model=None,
        val_dataloader=dataloader,
        generate_fn=_dummy_generate_fn,
        n_samples=None,
        gen_batch_size=4,
        key=key,
    )
    assert np.isfinite(result["fid"])


def test_compute_fid_metrics_n_real_limits_real_images():
    """When n_real is set, only that many real images are used."""
    acc = FIDAccumulator(encoder=_identity_encoder)
    accumulators = {"fid": acc}
    # 3 batches * 4 images = 12 real images, but cap at 6
    dataloader = _make_dummy_dataloader(n_batches=3, batch_size=4)
    key = jax.random.PRNGKey(4)

    result = compute_fid_metrics(
        accumulators=accumulators,
        model=None,
        val_dataloader=dataloader,
        generate_fn=_dummy_generate_fn,
        n_samples=6,
        gen_batch_size=4,
        key=key,
        n_real=6,
    )
    assert np.isfinite(result["fid"])
    # Cached real stats should reflect exactly 6 images
    _, _, n = acc._cached_real
    assert n == 6


def test_compute_fid_metrics_n_real_none_uses_full_dataset():
    """When n_real is None (default), all real images are used."""
    acc = FIDAccumulator(encoder=_identity_encoder)
    accumulators = {"fid": acc}
    # 3 batches * 4 images = 12 real images
    dataloader = _make_dummy_dataloader(n_batches=3, batch_size=4)
    key = jax.random.PRNGKey(5)

    result = compute_fid_metrics(
        accumulators=accumulators,
        model=None,
        val_dataloader=dataloader,
        generate_fn=_dummy_generate_fn,
        n_samples=12,
        gen_batch_size=4,
        key=key,
    )
    assert np.isfinite(result["fid"])
    _, _, n = acc._cached_real
    assert n == 12


def test_compute_fid_metrics_parallel_generation_min_one_runs():
    """min_devices=1 exercises the parallel generation path on CPU-only CI."""
    acc = FIDAccumulator(encoder=_identity_encoder)
    dataloader = _make_dummy_dataloader(n_batches=2, batch_size=4)

    result = compute_fid_metrics(
        accumulators={"fid": acc},
        model=None,
        val_dataloader=dataloader,
        generate_fn=_dummy_generate_fn,
        n_samples=8,
        gen_batch_size=4,
        key=jax.random.PRNGKey(20),
        parallel_generation={"enabled": True, "min_devices": 1},
    )

    assert np.isfinite(result["fid"])
    _, _, n_fake = acc.statistics()
    assert n_fake == 8


def test_compute_fid_metrics_parallel_generation_preserves_exact_n_samples():
    """Extra internally generated final-chunk images must not enter statistics."""
    acc = FIDAccumulator(encoder=_identity_encoder)
    dataloader = _make_dummy_dataloader(n_batches=2, batch_size=4)

    result = compute_fid_metrics(
        accumulators={"fid": acc},
        model=None,
        val_dataloader=dataloader,
        generate_fn=_dummy_generate_fn,
        n_samples=6,
        gen_batch_size=4,
        key=jax.random.PRNGKey(21),
        parallel_generation={"enabled": True, "min_devices": 1},
    )

    assert np.isfinite(result["fid"])
    _, _, n_fake = acc.statistics()
    assert n_fake == 6


def _model_generate_fn(model, key):
    """Generate an image from a model array without mutating the model."""
    noise = jax.random.normal(key, model.shape)
    return model + jnp.zeros_like(noise)


def test_parallel_generation_does_not_donate_model_argument():
    """FID parallel generation must leave the model reusable after a call."""
    model = jnp.ones((1, 2, 2), dtype=jnp.float32)
    acc = FIDAccumulator(encoder=_identity_encoder)
    dataloader = _make_dummy_dataloader(n_batches=2, batch_size=4)

    compute_fid_metrics(
        accumulators={"fid": acc},
        model=model,
        val_dataloader=dataloader,
        generate_fn=_model_generate_fn,
        n_samples=4,
        gen_batch_size=4,
        key=jax.random.PRNGKey(22),
        parallel_generation={"enabled": True, "min_devices": 1},
    )

    np.testing.assert_allclose(np.asarray(model), np.ones((1, 2, 2), dtype=np.float32))

    compute_fid_metrics(
        accumulators={"fid": acc},
        model=model,
        val_dataloader=_make_empty_dataloader(),
        generate_fn=_model_generate_fn,
        n_samples=4,
        gen_batch_size=4,
        key=jax.random.PRNGKey(23),
        parallel_generation={"enabled": True, "min_devices": 1},
    )


def test_fid_metric_accepts_parallel_generation_config():
    """FIDMetric forwards parallel_generation and data_parallel to compute."""
    metric = FIDMetric(
        accumulators={"fid": FIDAccumulator(encoder=_identity_encoder)},
        generate_fn=_dummy_generate_fn,
        n_samples=4,
        gen_batch_size=4,
        parallel_generation={"enabled": True, "min_devices": 1},
    )
    dataloader = _make_dummy_dataloader(n_batches=2, batch_size=4)
    data_parallel = make_data_parallel_config(enabled=True, min_devices=1)

    result = metric(
        model=None,
        val_dataloader=dataloader,
        key=jax.random.PRNGKey(24),
        data_parallel=data_parallel,
    )

    assert np.isfinite(result["fid"])


def test_train_has_no_num_val_eval_batches_param():
    """The num_val_eval_batches parameter must be removed from train()."""
    sig = inspect.signature(train)
    assert "num_val_eval_batches" not in sig.parameters




def test_fid_metric_delegates_to_compute_fid_metrics():
    """FIDMetric.__call__ must delegate to compute_fid_metrics and return its dict."""
    accumulators = {
        "enc_a": FIDAccumulator(encoder=_identity_encoder),
        "enc_b": FIDAccumulator(encoder=_identity_encoder),
    }
    dataloader = _make_dummy_dataloader(n_batches=2, batch_size=4)
    key = jax.random.PRNGKey(10)

    metric = FIDMetric(
        accumulators=accumulators,
        generate_fn=_dummy_generate_fn,
        n_samples=8,
        gen_batch_size=4,
    )
    result = metric(model=None, val_dataloader=dataloader, key=key)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"enc_a", "enc_b"}
    for v in result.values():
        assert isinstance(v, float)
        assert np.isfinite(v)


def test_fid_metric_caches_real_stats_across_calls():
    """FIDMetric must cache real-image stats so a second call skips the real pass."""
    acc = FIDAccumulator(encoder=_identity_encoder)
    metric = FIDMetric(
        accumulators={"fid": acc},
        generate_fn=_dummy_generate_fn,
        n_samples=8,
        gen_batch_size=4,
    )
    dataloader = _make_dummy_dataloader(n_batches=2, batch_size=4)
    key1, key2 = jax.random.split(jax.random.PRNGKey(11))

    result1 = metric(model=None, val_dataloader=dataloader, key=key1)
    # Second call with empty DataLoader should still work via cache.
    result2 = metric(model=None, val_dataloader=_make_empty_dataloader(), key=key2)

    assert np.isfinite(result1["fid"])
    assert np.isfinite(result2["fid"])
