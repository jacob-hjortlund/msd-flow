"""ODE-based sampling from a trained flow matching model.

Integrates the learned velocity field from noise (t=0) to data (t=1)
using Diffrax solvers.
"""

import jax
import diffrax

import jax.numpy as jnp

from msdflow.train.metrics import _to_velocity

# TODO: Move to inference


def sample(
    model,
    shape: tuple,
    key: jax.Array,
    solver,
    dt0: float,
    t0: float,
    t1: float,
    stepsize_controller,
    stepsize_controller_cfg: dict,
    cond: jax.Array | None = None,
    guidance_scale: float = 1.0,
) -> jax.Array:
    """Draw one sample by integrating the learned ODE from t0 to t1.

    Args:
        model:                Network accepting ``(t, x_t, cond, cond_mask)``.
            Must have a ``prediction_type`` attribute of ``"velocity"`` or
            ``"image"``.
        shape:                Shape of a single sample, e.g. (C, H, W)
        key:                  JAX PRNG key for initial noise
        solver:               Diffrax solver class (e.g. ``diffrax.Euler``).
        dt0:                  Initial step size.
        t0:                   Start time (0.0 = noise).
        t1:                   End time (1.0 = data).
        stepsize_controller:  Diffrax step-size controller class.
        stepsize_controller_cfg: Keyword arguments forwarded to the step-size
            controller constructor.
        cond:                 Conditioning vector of shape ``(D,)``. Pass
            ``None`` for unconditional sampling (the model's null embedding
            is used).
        guidance_scale:       Classifier-free guidance scale. ``1.0`` performs
            a single conditional forward pass; values ``> 1.0`` blend the
            conditional and unconditional predictions via
            ``v_uncond + guidance_scale * (v_cond - v_uncond)``.

    Returns:
        Sample array of shape `shape`.
    """

    if cond is None and guidance_scale != 1.0:
        raise ValueError(
            "guidance_scale != 1.0 requires an explicit cond; "
            "for unconditional sampling, leave cond=None (the default)."
        )

    solver = solver()
    stepsize_controller = (
        stepsize_controller()
    )  # TODO: Fix when controller has args / kwargs

    mask_true = jnp.array(True)
    mask_false = jnp.array(False)
    # When cond is None we are doing unconditional sampling: use the null
    # embedding (mask_false) and pass a dummy zeros vector so that any
    # cond[0] access inside the model is safe even though jnp.where evaluates
    # both branches eagerly.
    if cond is None:
        _cond = jnp.zeros(1)
        _mask = mask_false
    else:
        _cond = cond
        _mask = mask_true

    # Split off a key for model calls during integration (inference; key is
    # unused by UNet at present but required by the signature).
    key, model_key = jax.random.split(key)
    # model_key is a single static key shared across all ODE solver steps.
    # This is intentional: the UNet key is currently unused at inference
    # (dropout is disabled via inference_mode). If a future model uses the
    # key at inference, this will need to be re-split per step.

    def drift(t, y, args):
        # Python-level branch: evaluated at trace time, not a JAX conditional.
        # guidance_scale must remain a Python float (never a jax.Array).
        # t is a JAX scalar; _to_velocity expects shape (B,), so we
        # temporarily add/remove a batch dimension.
        t_batch = jnp.reshape(t, (1,))
        y_batch = y[None]  # (1, C, H, W)

        if guidance_scale == 1.0:
            pred = model(t, y, _cond, _mask, model_key)
            return _to_velocity(pred[None], y_batch, t_batch, model.prediction_type)[0]

        pred_cond = model(t, y, _cond, mask_true, model_key)
        pred_uncond = model(t, y, _cond, mask_false, model_key)
        v_cond = _to_velocity(pred_cond[None], y_batch, t_batch, model.prediction_type)[0]
        v_uncond = _to_velocity(pred_uncond[None], y_batch, t_batch, model.prediction_type)[0]
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    x0 = jax.random.normal(key, shape)
    term = diffrax.ODETerm(drift)
    saveat = diffrax.SaveAt(ts=jnp.array([t1]))
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=x0,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )
    # SaveAt(ts=[t1]) stores one time point; solution.ys shape is (1, C, H, W)
    return solution.ys[0]


def integrate_from_z(
    z: jax.Array,
    model,
    dt0: float = 0.05,
    t0: float = 0.0,
    t1: float = 1.0,
) -> jax.Array:
    """Integrate the flow ODE from an explicit initial latent ``z``.

    Unlike :func:`sample`, this function accepts ``z`` as a direct argument
    rather than drawing it from ``N(0, I)`` internally. This makes the output
    differentiable w.r.t. ``z`` via ``jax.grad``, which is required for
    gradient-based MCMC (e.g. NUTS).

    Uses a fixed Euler solver with :class:`diffrax.DirectAdjoint` so that
    gradients flow back through every ODE step without checkpointing.
    ``dt0=0.05`` (20 steps) is deliberately coarse to keep each MCMC
    step cheap; increase for higher fidelity samples after burn-in.

    Args:
        z:     Initial condition of shape ``(C, H, W)``.
        model: Trained velocity-field network. Must already be in
            inference mode (``eqx.nn.inference_mode``).
        dt0:   Fixed Euler step size. Defaults to 0.05 (20 steps over [0,1]).
        t0:    Integration start time. Defaults to 0.0.
        t1:    Integration end time. Defaults to 1.0.

    Returns:
        Generated sample of shape ``(C, H, W)``.
    """
    mask_false = jnp.array(False)
    cond_dummy = jnp.zeros(1)
    # Static key: dropout is off in inference_mode so the key is unused.
    model_key = jax.random.PRNGKey(0)

    def drift(t, y, args):
        t_batch = jnp.reshape(t, (1,))
        y_batch = y[None]
        pred = model(t, y, cond_dummy, mask_false, model_key)
        return _to_velocity(pred[None], y_batch, t_batch, model.prediction_type)[0]

    term = diffrax.ODETerm(drift)
    saveat = diffrax.SaveAt(ts=jnp.array([t1]))
    solution = diffrax.diffeqsolve(
        term,
        diffrax.Euler(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=z,
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=saveat,
        #adjoint=diffrax.DirectAdjoint(),
    )
    return solution.ys[0]
