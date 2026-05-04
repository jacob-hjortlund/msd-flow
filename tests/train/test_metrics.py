"""Tests for msdflow.train.metrics."""

import jax
import pytest

import numpy as np
import equinox as eqx
import jax.numpy as jnp

from msdflow.model.unet import UNet
from msdflow.flow.interpolate import sample_path
from msdflow.train.metrics import (
    TimeBinnedLossHistory,
    TimeBinnedLossResult,
    bin_time_losses,
    flow_matching_loss,
    flow_matching_per_sample_loss,
    make_time_binned_loss_step,
    _to_velocity,
)

KEY = jax.random.PRNGKey(0)

SMALL_MODEL = UNet(
    in_channels=1,
    out_channels=1,
    base_channels=4,
    channel_multipliers=[1, 2],
    num_res_blocks=1,
    num_heads=1,
    num_groups=2,
    activation=jax.nn.silu,
    key=KEY,
)

SMALL_MODEL_COND = UNet(
    in_channels=1,
    out_channels=1,
    base_channels=4,
    channel_multipliers=[1, 2],
    num_res_blocks=1,
    num_heads=1,
    num_groups=2,
    activation=jax.nn.silu,
    cond_dim=1,
    key=KEY,
)


class ZeroVelocityModel(eqx.Module):
    """Model that predicts zero velocity for deterministic metric tests."""

    prediction_type: str = "velocity"

    def __call__(self, t, x_t, cond, cond_mask, key):
        """Return a zero velocity field with the same shape as the input image."""
        return jnp.zeros_like(x_t)


class ConstantVelocityModel(eqx.Module):
    """Model that predicts a constant velocity bias."""

    prediction_type: str = "velocity"
    value: float = 1.0

    def __call__(self, t, x_t, cond, cond_mask, key):
        """Return a constant velocity field with the same shape as input."""
        return jnp.ones_like(x_t) * self.value


def test_flow_matching_loss_is_scalar():
    """Verify flow matching loss is a scalar."""
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss = flow_matching_loss(SMALL_MODEL, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2))
    assert loss.shape == ()


def test_flow_matching_per_sample_loss_returns_one_loss_per_example():
    """Per-sample loss should return one finite value for each batch element."""
    model = ZeroVelocityModel()
    batch_size = 3
    x_t = jnp.zeros((batch_size, 1, 4, 4))
    u_t = jnp.ones((batch_size, 1, 4, 4))
    t = jnp.array([0.0, 0.5, 1.0])
    cond = jnp.empty((batch_size, 0))
    cond_mask = jnp.zeros((batch_size,), dtype=bool)
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

    losses = flow_matching_per_sample_loss(model, x_t, u_t, t, cond, cond_mask, keys)

    assert losses.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(losses))
    assert jnp.allclose(losses, jnp.ones((batch_size,)))


def test_flow_matching_loss_project_velocity_removes_constant_bias():
    """Projecting velocity should remove spatially constant channel bias."""
    model = ConstantVelocityModel(value=2.0)
    batch_size = 3
    x_t = jnp.zeros((batch_size, 2, 4, 4))
    u_t = jnp.zeros((batch_size, 2, 4, 4))
    t = jnp.array([0.0, 0.5, 0.9])
    cond = jnp.empty((batch_size, 0))
    cond_mask = jnp.zeros((batch_size,), dtype=bool)
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

    unprojected = flow_matching_loss(
        model,
        x_t,
        u_t,
        t,
        cond,
        cond_mask,
        keys,
        project_velocity=False,
    )
    projected = flow_matching_loss(
        model,
        x_t,
        u_t,
        t,
        cond,
        cond_mask,
        keys,
        project_velocity=True,
    )

    assert jnp.allclose(unprojected, 4.0)
    assert jnp.allclose(projected, 0.0)


def test_flow_matching_per_sample_loss_project_velocity_removes_constant_bias():
    """Per-sample projected velocity losses should satisfy the same constraint."""
    model = ConstantVelocityModel(value=3.0)
    batch_size = 2
    x_t = jnp.zeros((batch_size, 1, 4, 4))
    u_t = jnp.zeros((batch_size, 1, 4, 4))
    t = jnp.array([0.25, 0.75])
    cond = jnp.empty((batch_size, 0))
    cond_mask = jnp.zeros((batch_size,), dtype=bool)
    keys = jax.random.split(jax.random.PRNGKey(1), batch_size)

    losses = flow_matching_per_sample_loss(
        model,
        x_t,
        u_t,
        t,
        cond,
        cond_mask,
        keys,
        project_velocity=True,
    )

    assert losses.shape == (batch_size,)
    assert jnp.allclose(losses, jnp.zeros((batch_size,)))


def test_bin_time_losses_assigns_t_one_to_final_bin():
    """Time binning should include t == 1.0 in the final bin."""
    t = jnp.array([0.0, 0.24, 0.25, 0.99, 1.0])
    losses = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    loss_sums, counts = bin_time_losses(t, losses, num_bins=4)

    assert np.allclose(np.asarray(loss_sums), np.array([3.0, 3.0, 0.0, 9.0]))
    assert np.array_equal(np.asarray(counts), np.array([2, 1, 0, 2]))


def test_bin_time_losses_rejects_nonpositive_num_bins():
    """Time binning should reject nonpositive bin counts."""
    t = jnp.array([0.5])
    losses = jnp.array([1.0])

    with pytest.raises(ValueError, match="num_bins must be >= 1"):
        bin_time_losses(t, losses, num_bins=0)


def test_time_binned_loss_result_mean_uses_nan_for_empty_bins():
    """Empty time bins should have NaN mean loss and zero counts."""
    result = TimeBinnedLossResult.empty(num_bins=3)
    result.add_batch(
        loss_sums=np.array([2.0, 0.0, 6.0]),
        counts=np.array([1, 0, 3]),
    )

    assert np.allclose(result.mean_loss[[0, 2]], np.array([2.0, 2.0]))
    assert np.isnan(result.mean_loss[1])
    assert np.array_equal(result.counts, np.array([1, 0, 3]))


def test_time_binned_loss_result_empty_rejects_nonpositive_num_bins():
    """Time-binned result creation should reject nonpositive bin counts."""
    with pytest.raises(ValueError, match="num_bins must be >= 1"):
        TimeBinnedLossResult.empty(num_bins=0)


def test_time_binned_loss_history_records_epochs_and_means():
    """History should keep epoch, mean loss, and counts for heatmap plotting."""
    result = TimeBinnedLossResult.empty(num_bins=2)
    result.add_batch(
        loss_sums=np.array([1.0, 4.0]),
        counts=np.array([1, 2]),
    )
    history = TimeBinnedLossHistory(bin_edges=result.bin_edges)

    history.append(epoch=5, result=result)

    assert history.epochs == [5]
    assert np.allclose(history.mean_losses[0], np.array([1.0, 2.0]))
    assert np.array_equal(history.counts[0], np.array([1, 2]))


def test_time_binned_loss_history_snapshots_appended_results():
    """History should not change when a previously appended result mutates."""
    result = TimeBinnedLossResult.empty(num_bins=2)
    result.add_batch(
        loss_sums=np.array([1.0, 4.0]),
        counts=np.array([1, 2]),
    )
    history = TimeBinnedLossHistory(bin_edges=result.bin_edges)

    history.append(epoch=5, result=result)
    result.add_batch(
        loss_sums=np.array([10.0, 10.0]),
        counts=np.array([10, 10]),
    )

    assert np.allclose(history.mean_losses[0], np.array([1.0, 2.0]))
    assert np.array_equal(history.counts[0], np.array([1, 2]))


def test_make_time_binned_loss_step_returns_bin_sums_and_counts():
    """Diagnostic step should return one sum and count per configured time bin."""
    model = ZeroVelocityModel()
    batch_size = 4
    x_t = jnp.zeros((batch_size, 1, 4, 4))
    u_t = jnp.ones((batch_size, 1, 4, 4))
    t = jnp.array([0.1, 0.2, 0.7, 1.0])
    cond = jnp.empty((batch_size, 0))
    cond_mask = jnp.zeros((batch_size,), dtype=bool)
    keys = jax.random.split(jax.random.PRNGKey(1), batch_size)
    step = make_time_binned_loss_step(num_bins=4)

    loss_sums, counts = step(model, x_t, u_t, t, cond, cond_mask, keys)

    assert loss_sums.shape == (4,)
    assert counts.shape == (4,)
    assert np.allclose(np.asarray(loss_sums), np.array([2.0, 0.0, 1.0, 1.0]))
    assert np.array_equal(np.asarray(counts), np.array([2, 0, 1, 1]))


def test_make_time_binned_loss_step_project_velocity_removes_constant_bias():
    """Projected diagnostic losses should remove spatially constant bias."""
    model = ConstantVelocityModel(value=2.0)
    batch_size = 3

    def make_batch():
        """Create fresh arrays because the diagnostic step donates inputs."""
        return (
            jnp.zeros((batch_size, 1, 4, 4)),
            jnp.zeros((batch_size, 1, 4, 4)),
            jnp.array([0.1, 0.3, 0.8]),
            jnp.empty((batch_size, 0)),
            jnp.zeros((batch_size,), dtype=bool),
            jax.random.split(jax.random.PRNGKey(2), batch_size),
        )

    unprojected_step = make_time_binned_loss_step(
        num_bins=2,
        project_velocity=False,
    )
    unprojected_sums, unprojected_counts = unprojected_step(
        model,
        *make_batch(),
    )

    projected_step = make_time_binned_loss_step(
        num_bins=2,
        project_velocity=True,
    )
    projected_sums, projected_counts = projected_step(
        model,
        *make_batch(),
    )

    assert np.allclose(np.asarray(unprojected_sums), np.array([8.0, 4.0]))
    assert np.array_equal(np.asarray(unprojected_counts), np.array([2, 1]))
    assert np.allclose(np.asarray(projected_sums), np.array([0.0, 0.0]))
    assert np.array_equal(np.asarray(projected_counts), np.array([2, 1]))


def test_make_time_binned_loss_step_rejects_nonpositive_num_bins():
    """Diagnostic step creation should reject nonpositive bin counts."""
    with pytest.raises(ValueError, match="num_bins must be >= 1"):
        make_time_binned_loss_step(num_bins=0)


def test_flow_matching_loss_is_positive():
    """Verify flow matching loss is non-negative."""
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss = flow_matching_loss(SMALL_MODEL, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2))
    assert loss >= 0.0


def test_flow_matching_loss_has_gradient():
    """Verify at least one gradient leaf is non-zero."""
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss, grads = eqx.filter_value_and_grad(flow_matching_loss)(
        SMALL_MODEL, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2)
    )
    grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(g != 0.0) for g in grad_leaves)


def test_flow_matching_loss_with_cond():
    """Verify flow matching loss works with conditioning."""
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.array([[0.4], [0.8]])
    cond_mask = jnp.ones(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss = flow_matching_loss(SMALL_MODEL_COND, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2))
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_flow_matching_loss_with_cond_mask_false():
    """Verify loss works when all conditions are masked (unconditional)."""
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.array([[0.4], [0.8]])
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss = flow_matching_loss(SMALL_MODEL_COND, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2))
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_flow_matching_loss_gradient_with_cond():
    """Verify gradients flow through conditional loss."""
    B = 2
    k1, k2 = jax.random.split(KEY)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.array([[0.4], [0.8]])
    cond_mask = jnp.array([True, False])
    x_t, u_t = sample_path(x0, x1, t)
    loss, grads = eqx.filter_value_and_grad(flow_matching_loss)(
        SMALL_MODEL_COND, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2)
    )
    grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(g != 0.0) for g in grad_leaves)


def test_sample_path_stochastic_x_t_differs_from_deterministic():
    """With nonzero sigma, x_t must differ from the noiseless interpolant."""
    key = jax.random.PRNGKey(42)
    x0 = jnp.ones((2, 1, 4, 4)) * 2.0
    x1 = jnp.ones((2, 1, 4, 4)) * 5.0
    t = jnp.array([0.3, 0.7])
    x_t_det, _ = sample_path(x0, x1, t)
    x_t_stoch, _ = sample_path(x0, x1, t, sigma_0=0.1, sigma_1=0.1, key=key)
    assert not jnp.allclose(x_t_det, x_t_stoch)


def test_sample_path_stochastic_velocity_unchanged():
    """Velocity u_t must equal x1 - x0 regardless of sigma values."""
    key = jax.random.PRNGKey(7)
    x0 = jnp.ones((2, 1, 4, 4)) * 2.0
    x1 = jnp.ones((2, 1, 4, 4)) * 5.0
    t = jnp.array([0.3, 0.7])
    _, u_t = sample_path(x0, x1, t, sigma_0=0.5, sigma_1=0.2, key=key)
    assert jnp.allclose(u_t, x1 - x0)


def test_sample_path_zero_sigma_matches_deterministic():
    """sigma_0=0, sigma_1=0 with a key provided must give the same result as no key."""
    key = jax.random.PRNGKey(0)
    x0 = jnp.ones((2, 1, 4, 4)) * 2.0
    x1 = jnp.ones((2, 1, 4, 4)) * 5.0
    t = jnp.array([0.3, 0.7])
    x_t_a, u_t_a = sample_path(x0, x1, t)
    x_t_b, u_t_b = sample_path(x0, x1, t, sigma_0=0.0, sigma_1=0.0, key=key)
    assert jnp.allclose(x_t_a, x_t_b)
    assert jnp.allclose(u_t_a, u_t_b)


# --- _to_velocity ---


def test_to_velocity_velocity_mode_returns_pred():
    """In velocity mode, _to_velocity is an identity on pred."""
    pred = jnp.ones((2, 1, 4, 4)) * 3.0
    x_t = jnp.ones((2, 1, 4, 4)) * 1.0
    t = jnp.array([0.4, 0.6])
    v = _to_velocity(pred, x_t, t, "velocity")
    assert jnp.allclose(v, pred)


def test_to_velocity_image_mode_formula():
    """In image mode, _to_velocity applies (pred - x_t) / (1 - t)."""
    pred = jnp.ones((2, 1, 4, 4)) * 2.0
    x_t = jnp.ones((2, 1, 4, 4)) * 0.5
    t = jnp.array([0.5, 0.5])
    v = _to_velocity(pred, x_t, t, "image")
    expected = (pred - x_t) / (1.0 - t[:, None, None, None])
    assert jnp.allclose(v, expected)


def test_to_velocity_image_mode_shape():
    """Output shape matches input shape."""
    B, C, H, W = 3, 2, 8, 8
    pred = jax.random.normal(KEY, (B, C, H, W))
    x_t = jax.random.normal(KEY, (B, C, H, W))
    t = jnp.array([0.1, 0.5, 0.9])
    v = _to_velocity(pred, x_t, t, "image")
    assert v.shape == (B, C, H, W)


# --- Image-mode prediction tests ---

_KEY2 = jax.random.PRNGKey(99)
_SMALL_IMG_MODEL = UNet(
    in_channels=1,
    out_channels=1,
    base_channels=4,
    channel_multipliers=[1, 2],
    num_res_blocks=1,
    num_heads=1,
    num_groups=2,
    activation=jax.nn.silu,
    key=_KEY2,
    prediction_type="image",
)


def test_flow_matching_loss_image_mode_is_scalar():
    """Loss with an image-prediction model is a scalar."""
    B = 2
    k1, k2 = jax.random.split(_KEY2)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss = flow_matching_loss(_SMALL_IMG_MODEL, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2))
    assert loss.shape == ()


def test_flow_matching_loss_image_mode_is_finite():
    """Loss with an image-prediction model is finite."""
    B = 2
    k1, k2 = jax.random.split(_KEY2)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    loss = flow_matching_loss(_SMALL_IMG_MODEL, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2))
    assert jnp.isfinite(loss)


def test_flow_matching_loss_image_mode_has_gradient():
    """Gradients flow through image-mode loss."""
    B = 2
    k1, k2 = jax.random.split(_KEY2)
    x0 = jax.random.normal(k1, (B, 1, 8, 8))
    x1 = jax.random.normal(k2, (B, 1, 8, 8))
    t = jnp.array([0.3, 0.7])
    cond = jnp.empty((B, 0))
    cond_mask = jnp.zeros(B, dtype=bool)
    x_t, u_t = sample_path(x0, x1, t)
    _, grads = eqx.filter_value_and_grad(flow_matching_loss)(
        _SMALL_IMG_MODEL, x_t, u_t, t, cond, cond_mask, jax.random.split(jax.random.PRNGKey(0), 2)
    )
    grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(g != 0.0) for g in grad_leaves)
