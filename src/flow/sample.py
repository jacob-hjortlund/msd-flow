"""ODE-based sampling from a trained flow matching model.

Integrates the learned velocity field from noise (t=0) to data (t=1)
using Diffrax solvers.
"""

import jax
import diffrax

import jax.numpy as jnp

from src.utils.utils import resolve_import

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
        model:                UNet velocity-field network
        shape:                Shape of a single sample, e.g. (C, H, W)
        key:                  JAX PRNG key for initial noise
        solver:               Diffrax solver (e.g. diffrax.Euler())
        dt0:                  Initial step size
        t0:                   Start time (0.0 = noise)
        t1:                   End time (1.0 = data)
        stepsize_controller:  Diffrax step controller
        stepsize_controller_cfg: Keyword arguments forwarded to the step-size
            controller constructor.
        cond:                 Conditioning vector of shape ``(D,)``. ``None``
            is equivalent to passing ``jnp.empty(0)`` (unconditional).
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
            "pass cond=jnp.empty(0) for unconditional sampling."
        )

    if isinstance(solver, str):
        solver = resolve_import(solver)

    if isinstance(stepsize_controller, str):
        stepsize_controller = resolve_import(stepsize_controller)

    solver = solver()
    stepsize_controller = (
        stepsize_controller()
    )  # TODO: Fix when controller has args / kwargs

    mask_true = jnp.array(True)
    _cond = jnp.empty(0) if cond is None else cond

    def drift(t, y, args):
        # Python-level branch: evaluated at trace time, not a JAX conditional.
        # guidance_scale must remain a Python float (never a jax.Array).
        if guidance_scale == 1.0:
            return model(t, y, _cond, mask_true)
        mask_false = jnp.array(False)
        v_cond = model(t, y, _cond, mask_true)
        v_uncond = model(t, y, _cond, mask_false)
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
