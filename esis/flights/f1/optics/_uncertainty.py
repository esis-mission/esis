"""Construction of the uncertain parameters of the optical models."""

import collections.abc
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "spawn_seeds",
    "uniform",
]


def spawn_seeds(
    seed: None | int,
) -> collections.abc.Iterator[int]:
    """
    Spawn a seed for each uncertain parameter of a model from the seed of the model.

    Each parameter needs a seed of its own, since two parameters drawn with
    the same seed would have the same samples, and so their errors would be
    perfectly correlated instead of independent.

    Parameters
    ----------
    seed
        The seed of the model.
        If :obj:`None`, the seeds are spawned from fresh entropy,
        so they are different every time.

    Yields
    ------
    int
        The seed of the next uncertain parameter.
    """
    sequence = np.random.SeedSequence(seed)
    while True:
        (child,) = sequence.spawn(1)
        yield int(child.generate_state(1, dtype=np.uint64)[0])


def uniform(
    nominal: float | u.Quantity | na.AbstractScalar,
    width: float | u.Quantity | na.AbstractScalar,
    num_distribution: int,
    seed: None | int,
) -> float | u.Quantity | na.AbstractScalar:
    """
    Create a parameter whose value is known to within a uniform distribution.

    If `num_distribution` is zero, the nominal value is returned as is,
    so that a model built without any samples has no uncertain parameters,
    rather than uncertain parameters with an empty distribution.

    Parameters
    ----------
    nominal
        The nominal value of the parameter.
    width
        The half-width of the uniform distribution of possible values,
        which span the nominal value plus or minus this width.
    num_distribution
        The number of samples drawn from the distribution.
    seed
        The seed of the random number generator which draws the samples,
        one of those given by :func:`spawn_seeds` for the model.
        If :obj:`None`, the samples are different every time.
    """
    if num_distribution == 0:
        return nominal
    return na.UniformUncertainScalarArray(
        nominal=nominal,
        width=width,
        num_distribution=num_distribution,
        seed=seed,
    )
