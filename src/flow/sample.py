import jax
import diffrax

import jax.numpy as jnp


def make_ode_term(model) -> diffrax.ODETerm:
    """Wrap a UNet as a Diffrax ODETerm.

    Diffrax calls the vector field as f(t, y, args).
    UNet.__call__ expects (x_t, t), so we reorder arguments.
    """
    return diffrax.ODETerm(lambda t, y, args: model(y, t))


def sample(
    model,
    shape: tuple,
    key: jax.Array,
    solver,
    dt0: float,
    t0: float,
    t1: float,
    stepsize_controller,
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

    Returns:
        Sample array of shape `shape`.
    """
    x0 = jax.random.normal(key, shape)
    # term = make_ode_term(model)
    term = diffrax.ODETerm(model)
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
