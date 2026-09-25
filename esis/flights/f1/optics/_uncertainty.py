"""Construction of the uncertain parameters of the optical models."""

import astropy.units as u
import named_arrays as na

__all__ = [
    "uniform",
]


def uniform(
    nominal: float | u.Quantity | na.AbstractScalar,
    width: float | u.Quantity | na.AbstractScalar,
    num_distribution: int,
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
        The width of the uniform distribution of possible values.
    num_distribution
        The number of samples drawn from the distribution.
    """
    if num_distribution == 0:
        return nominal
    return na.UniformUncertainScalarArray(
        nominal=nominal,
        width=width,
        num_distribution=num_distribution,
    )
