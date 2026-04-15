"""Tests for FID metric components in msdflow.train.metrics."""

import inspect

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from msdflow.train.metrics import _frechet_distance
from msdflow.train.metrics import FIDAccumulator
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


from msdflow.train.metrics import compute_fid_metrics


def _make_dummy_dataloader(n_batches, batch_size, shape=(1, 2, 2), seed=0):
    """Return a list of (images_tensor, meta_tensor) tuples usable as a dataloader."""
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(n_batches):
        images = jnp.array(rng.standard_normal((batch_size, *shape)).astype(np.float32))
        meta = jnp.empty((batch_size, 0))
        batches.append((images, meta))
    return batches


def _dummy_generate_fn(model, key):
    """Generate a single fake image of shape (1, 2, 2) from random noise."""
    return jax.random.normal(key, (1, 2, 2))


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
    # Mutate dataloader to prove it's not re-read on second call
    dataloader_empty = []
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


def test_train_has_no_num_val_eval_batches_param():
    """The num_val_eval_batches parameter must be removed from train()."""
    sig = inspect.signature(train)
    assert "num_val_eval_batches" not in sig.parameters
