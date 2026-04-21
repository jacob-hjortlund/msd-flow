"""Tests for msdflow.flow.coupling."""

import jax.numpy as jnp
import numpy as np
import pytest
from msdflow.flow.coupling import independent_coupling, ot_coupling


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
    # Every row of x0_paired must appear exactly once in x0
    matches = [
        np.where([np.allclose(row, x0_row) for x0_row in x0_flat])[0]
        for row in x0p_flat
    ]
    # Each result row must match exactly one source row
    assert all(len(m) == 1 for m in matches), "each output row must match exactly one input row"
    # All matched indices must be distinct (no source row used twice)
    matched_indices = [m[0] for m in matches]
    assert len(set(matched_indices)) == len(matched_indices), "output rows must be a permutation (no duplicates)"


def test_independent_coupling_accepts_jax_arrays():
    """independent_coupling must work with JAX arrays."""
    x0 = jnp.ones((2, 1, 4, 4))
    x1 = jnp.zeros((2, 1, 4, 4))
    result = independent_coupling(x0, x1)
    assert isinstance(result, jnp.ndarray)
    assert jnp.array_equal(result, x0)
