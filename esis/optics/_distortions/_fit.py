"""The optimizers of the distortion fit: a seeded capture and a local polish."""

from __future__ import annotations
from typing import Any, Callable
import datetime
import multiprocessing
import time
import numpy as np
import scipy.optimize
import named_arrays as na
import esis

__all__ = [
    "fit_distortion",
    "polish",
]


def _log(log: None | Callable[[str], None], message: str) -> None:
    if log is not None:
        log(f"{datetime.datetime.now():%H:%M:%S} | {message}")


def polish(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    scale: float = 0.02,
    num_round: int = 2,
    maxfev: int = 4000,
    log: None | Callable[[str], None] = None,
) -> tuple[np.ndarray, float, int]:
    """
    Polish a solution with restarted Nelder-Mead simplices.

    The objective is smooth and deterministic once the fit is inside its
    basin, so a simplex spends every evaluation descending; the size of the
    starting simplex decides whether it does.  A vertex a tenth of the
    bounds away is a millimetre of defocus, and stalls; a fiftieth reaches
    the optimum.  Each vertex steps into the interior, because the design
    start sits on the upper bound of two parameters and a clipped step there
    would silently leave that axis unexplored.  A second round restarts from
    the best point with a smaller simplex.

    Parameters
    ----------
    objective
        The function to minimize, of a flat parameter vector.
    x0
        The starting vector.
    lower
        The lower bounds.
    upper
        The upper bounds.
    scale
        The size of the starting simplex as a fraction of the bounds.
    num_round
        The number of rounds; each restarts from the best point seen with a
        simplex a third of the size of the previous one.
    maxfev
        The maximum number of evaluations per round.
    log
        A callable to report the result of each round.

    Returns
    -------
    The best vector seen, its objective, and the number of evaluations.
    """
    best = dict(fun=np.inf, x=np.array(x0), n=0)

    def tracked(x):
        f = objective(x)
        best["n"] += 1
        if f < best["fun"]:
            best["fun"], best["x"] = f, np.array(x)
        return f

    x = np.array(x0)
    for i in range(num_round):
        step = scale * (upper - lower)
        step = np.where(x + step <= upper, step, -step)
        simplex = np.vstack([x] + [x + step * e for e in np.eye(len(x))])
        result = scipy.optimize.minimize(
            tracked,
            x,
            method="Nelder-Mead",
            bounds=scipy.optimize.Bounds(lower, upper),
            options=dict(
                initial_simplex=simplex,
                xatol=1e-3,
                fatol=1e-5,
                maxfev=maxfev,
                adaptive=True,
            ),
        )
        _log(
            log,
            f"polish round {i + 1}: {-result.fun:.4f} after {result.nfev} evaluations",
        )
        x, scale = best["x"], scale / 3
    return best["x"], best["fun"], best["n"]


def fit_distortion(
    objective: Callable[[np.ndarray], float],
    parameters: esis.optics.DistortionParameters,
    bounds: tuple[esis.optics.DistortionParameters, esis.optics.DistortionParameters],
    workers: int = 1,
    popsize: int = 15,
    maxiter: int = 60,
    tol: float = 0.05,
    seed: int = 0,
    scale_polish: float = 0.02,
    log: None | Callable[[str], None] = None,
    kwargs_optimizer: None | dict[str, Any] = None,
) -> esis.optics.DistortionParameters:
    """
    Fit distortion parameters by a seeded capture and a local polish.

    A local method alone does not reliably find the basin of the merit from
    the as-built model, or from starts displaced by a tenth of the bounds,
    so the first stage is a seeded differential evolution with a loose
    tolerance: it only has to land in the basin.  The second stage is
    :func:`polish`.  Because the objective is deterministic and the capture
    is seeded, the whole fit is reproducible.

    Parameters
    ----------
    objective
        The function to minimize, of a flat parameter vector, for example a
        :class:`LinearMerit`.  It must be picklable if `workers` is more than
        one.
    parameters
        The initial guess, which also defines the units and structure of the
        parameter vector.
    bounds
        The lower and upper bounds of the fit, in the units of `parameters`.
    workers
        The number of processes evaluating the population of the capture.
        They are spawned rather than forked, since the parent has usually
        initialized a GPU by the time the capture starts.
    popsize
        The population size of the capture per parameter.
    maxiter
        The maximum number of generations of the capture.
    tol
        The relative tolerance at which the capture stops.
    seed
        The seed of the capture.
    scale_polish
        The size of the starting simplex of the polish as a fraction of the
        bounds.
    log
        A callable to report progress.
    kwargs_optimizer
        Additional keyword arguments passed to
        :func:`scipy.optimize.differential_evolution`.

    Examples
    --------
    Fit one channel of the ESIS-I as-built model to the Level-1 frame that
    :func:`esis.flights.f1.optics.distortion_fit` was optimized against.

    .. code-block:: python

        import named_arrays as na
        import esis

        obs = esis.flights.f1.data.level_1()[dict(time=15)]
        scene = esis.flights.f1.data.synth.scene_aia()[dict(time=15)]

        instrument = esis.flights.f1.optics.as_built(num_distribution=0)
        channel = instrument[dict(channel=1)]
        parameters = esis.optics.DistortionParameters.from_instrument(channel)

        merit = esis.optics.LinearMerit(
            instrument=channel,
            parameters=parameters,
            scene=scene,
            observation=obs.outputs[dict(channel=1)].value,
            device="cuda",
        )
        fitted = esis.optics.fit_distortion(
            objective=merit,
            parameters=parameters,
            bounds=esis.flights.f1.optics.distortion_fit_bounds(parameters),
            workers=6,
        )
    """
    if kwargs_optimizer is None:
        kwargs_optimizer = dict()
    lower, upper = bounds
    lb, ub = na.pack(lower).ndarray, na.pack(upper).ndarray
    x0 = na.pack(parameters).ndarray

    time_start = time.perf_counter()
    _log(log, f"start: {-objective(x0):.4f}")

    generation = [0]

    def callback(intermediate_result=None, convergence=None):
        generation[0] += 1
        fun = getattr(intermediate_result, "fun", None)
        if fun is not None:
            _log(log, f"capture generation {generation[0]}: {-fun:.4f}")

    kwargs = dict(
        bounds=scipy.optimize.Bounds(lb, ub),
        x0=x0,
        popsize=popsize,
        maxiter=maxiter,
        tol=tol,
        polish=False,
        seed=seed,
        callback=callback,
        updating="deferred",
    )
    kwargs.update(kwargs_optimizer)
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            result = scipy.optimize.differential_evolution(
                objective, workers=pool.map, **kwargs
            )
    else:
        result = scipy.optimize.differential_evolution(objective, **kwargs)
    _log(
        log,
        f"capture: {-result.fun:.4f} after {result.nfev} evaluations, "
        f"{time.perf_counter() - time_start:.0f} s",
    )

    x, fun, num = polish(objective, result.x, lb, ub, scale=scale_polish, log=log)
    _log(
        log,
        f"polish: {-fun:.4f} after {num} evaluations, "
        f"{time.perf_counter() - time_start:.0f} s in total",
    )
    return na.unpack(x, parameters)
