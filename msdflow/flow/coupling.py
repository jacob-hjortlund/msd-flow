"""Minibatch coupling strategies for flow matching.

Both functions share the interface (x0, x1) -> x0_paired, where x0_paired
is the permuted (or identity) source array to be paired with x1 during training.

``independent_coupling`` is type-agnostic and works inside JAX JIT.
``ot_coupling`` uses scipy and requires NumPy arrays (CPU only, outside JIT).
"""

from typing import Union

import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

ArrayLike = Union[np.ndarray, jnp.ndarray]


def independent_coupling(x0: ArrayLike, x1: ArrayLike) -> ArrayLike:
    """Return x0 unchanged — the independent (no-pairing) coupling.

    x0 and x1 are treated as independently drawn samples with no attempt to
    match them. This is the baseline coupling for vanilla flow matching.

    Accepts both NumPy and JAX arrays and is compatible with JAX JIT.

    Args:
        x0: shape (B, C, H, W) — noise samples (NumPy or JAX array).
        x1: shape (B, C, H, W) — data samples (unused; NumPy or JAX array).

    Returns:
        x0 unchanged, shape (B, C, H, W).
    """
    return x0


def ot_coupling(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Pair x0 ~ N(0,I) with x1 ~ p_data via minibatch optimal transport.

    Runs the Hungarian algorithm on the pairwise squared L2 cost matrix to
    return a permutation of x0 that minimises total transport cost to x1.
    Both inputs shape (B, C, H, W).

    Args:
        x0: shape (B, C, H, W) — noise samples.
        x1: shape (B, C, H, W) — data samples.

    Returns:
        Permutation of x0 that minimises squared L2 cost to x1,
        shape (B, C, H, W).
    """
    B = x0.shape[0]
    x0_flat = x0.reshape(B, -1)
    x1_flat = x1.reshape(B, -1)
    cost = np.sum((x0_flat[:, None, :] - x1_flat[None, :, :]) ** 2, axis=-1)
    _, col_ind = linear_sum_assignment(cost)
    return x0[col_ind]
