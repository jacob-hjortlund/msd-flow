"""Bridge sampling for Image-to-Image Flow Matching.

At inference the ODE is integrated starting from y (the degraded observation)
rather than from Gaussian noise.  The training loss and OT coupling in
src/flow/otfm.py are reused unchanged — the only difference is the source of x0.

Reference: I²SB (Liu et al., 2023, arXiv:2302.05872)
"""

import jax
import diffrax
import jax.numpy as jnp


def sample_bridge(
    model,
    y: jax.Array,
    solver: diffrax.AbstractSolver,
    dt0: float,
    t0: float,
    t1: float,
    stepsize_controller: diffrax.AbstractStepSizeController,
) -> jax.Array:
    """Integrate the learned ODE forward from y (degraded input) to a reconstruction.

    Args:
        model:               UNet velocity-field network.
        y:                   Degraded observation, shape (C, H, W).  Used as x(t0).
        solver:              Diffrax solver instance (e.g. diffrax.Euler()).
        dt0:                 Initial step size.
        t0:                  Start time (0.0).
        t1:                  End time (1.0).
        stepsize_controller: Diffrax step-size controller instance.

    Returns:
        Reconstructed sample of shape (C, H, W).
    """
    term = diffrax.ODETerm(model)
    saveat = diffrax.SaveAt(ts=jnp.array([t1]))
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )
    # solution.ys shape: (1, C, H, W) — one saved time point
    return solution.ys[0]


def sample_bridge_trajectory(
    model,
    y: jax.Array,
    n_frames: int = 8,
    t0: float = 0.0,
    t1: float = 1.0,
    dt0: float = 0.01,
) -> jax.Array:
    """Integrate the ODE from y and save states at n_frames equally-spaced times.

    Useful for visualising the progressive sharpening from degraded input to
    reconstructed output.

    Args:
        model:    UNet velocity-field network.
        y:        Degraded observation, shape (C, H, W).
        n_frames: Number of snapshots to save (including t0 and t1).
        t0:       Start time.
        t1:       End time.
        dt0:      ODE step size.

    Returns:
        Array of shape (n_frames, C, H, W).
    """
    ts = jnp.linspace(t0, t1, n_frames)
    term = diffrax.ODETerm(model)
    saveat = diffrax.SaveAt(ts=ts)
    solution = diffrax.diffeqsolve(
        term,
        diffrax.Euler(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y,
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=saveat,
    )
    return solution.ys  # (n_frames, C, H, W)
