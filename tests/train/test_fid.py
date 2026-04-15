"""Tests for FID metric components in msdflow.train.metrics."""

import numpy as np
import pytest

from msdflow.train.metrics import _frechet_distance


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
