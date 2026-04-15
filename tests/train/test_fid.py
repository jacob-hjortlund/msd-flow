"""Tests for FID metric components in msdflow.train.metrics."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from msdflow.train.metrics import _frechet_distance
from msdflow.train.metrics import FIDAccumulator


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
