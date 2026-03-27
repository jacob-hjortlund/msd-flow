"""Tests for src.flow.coupling."""

import numpy as np
import pytest
from src.flow.coupling import independent_coupling, ot_coupling


def test_independent_coupling_returns_x0_unchanged():
    """independent_coupling must return the exact x0 array."""
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    x1 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    result = independent_coupling(x0, x1)
    np.testing.assert_array_equal(result, x0)


def test_independent_coupling_output_shape():
    """independent_coupling output shape matches input."""
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    x1 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    assert independent_coupling(x0, x1).shape == x0.shape


def test_ot_coupling_output_shape():
    """ot_coupling output shape matches input shape."""
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    x1 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    x0_paired = ot_coupling(x0, x1)
    assert x0_paired.shape == x0.shape


def test_ot_coupling_is_permutation():
    """ot_coupling returns a permutation of the source rows."""
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    x1 = rng.standard_normal((4, 1, 8, 8)).astype(np.float32)
    x0_paired = ot_coupling(x0, x1)
    x0_flat = x0.reshape(4, -1)
    x0p_flat = x0_paired.reshape(4, -1)
    for row in x0p_flat:
        assert any(np.allclose(row, x0_row) for x0_row in x0_flat)
