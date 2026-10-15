import collections.abc
from typing import Callable
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import esis


def _assert_grating_certain(
    instrument: esis.optics.abc.AbstractInstrument,
    radius: bool = True,
) -> None:
    """
    Assert that the rulings of the grating, and optionally its radius, are certain.

    Their uncertainty is not known yet,
    so they should not carry over the uncertainty of the ESIS-I gratings.

    Parameters
    ----------
    instrument
        The instrument whose grating is checked.
    radius
        Whether to check the radius of curvature of the grating as well.
    """
    coefficients = instrument.grating.rulings.spacing.coefficients
    for power in coefficients:
        assert not isinstance(coefficients[power], na.AbstractUncertainScalarArray)
    if radius:
        assert not isinstance(
            instrument.grating.sag.radius, na.AbstractUncertainScalarArray
        )


@pytest.mark.parametrize(
    argnames="function",
    argvalues=[
        esis.flights.f2.optics.design_proposed,
        esis.flights.f2.optics.design_guess,
        esis.flights.f2.optics.design_single,
        esis.flights.f2.optics.design,
        esis.flights.f2.optics.design_visible,
    ],
)
def test_seed(
    function: collections.abc.Callable[..., esis.optics.Instrument],
) -> None:
    """
    Check that the same seed draws the same samples, and another seed others.

    Each design draws its samples by way of an ESIS-I design, which draws
    the placement of the sensors along with everything else.
    """
    result = function(num_distribution=11, seed=0).camera.sensor.translation
    again = function(num_distribution=11, seed=0).camera.sensor.translation
    other = function(num_distribution=11, seed=1).camera.sensor.translation

    for component in result.components:
        a = result.components[component]
        assert np.all(again.components[component] == a), component
        assert np.all(other.components[component].distribution != a.distribution)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_proposed(num_distribution: int):
    result = esis.flights.f2.optics.design_proposed(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
    _assert_grating_certain(result, radius=False)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_guess(num_distribution: int):
    result = esis.flights.f2.optics.design_guess(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
    _assert_grating_certain(result)


def test_design_guess_nominal_geometry():
    """
    Check that the grating of an uncertain instrument fits its nominal geometry.

    The nominal geometry is the geometry of the instrument without samples.
    """
    certain = esis.flights.f2.optics.design_guess(num_distribution=0)
    uncertain = esis.flights.f2.optics.design_guess(num_distribution=11)
    coefficients = uncertain.grating.rulings.spacing.coefficients
    coefficients_certain = certain.grating.rulings.spacing.coefficients
    for power in coefficients:
        assert np.all(np.isclose(coefficients[power], coefficients_certain[power]))
    radius = uncertain.grating.sag.radius
    assert np.all(np.isclose(radius, certain.grating.sag.radius))


@pytest.mark.parametrize(
    "design",
    [
        esis.flights.f2.optics.design_guess,
        esis.flights.f2.optics.design_single,
        esis.flights.f2.optics.design,
        esis.flights.f2.optics.design_visible,
    ],
)
def test_timedelta_exposure(design: Callable[..., esis.optics.Instrument]):
    """Check that every design of ESIS-II, down to its sensor, exposes for 1.2 s."""
    result = design(num_distribution=0)
    assert result.camera.timedelta_exposure == 1.2 * u.s
    assert result.system.sensor.timedelta_exposure == 1.2 * u.s


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_single(num_distribution: int):
    result = esis.flights.f2.optics.design_single(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
    _assert_grating_certain(result)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design(num_distribution: int):
    result = esis.flights.f2.optics.design(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
    _assert_grating_certain(result)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_visible(num_distribution: int):
    result = esis.flights.f2.optics.design_visible(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)

    euv = esis.flights.f2.optics.design(
        num_distribution=num_distribution,
    )
    ratio = esis.flights.f2.wavelength_HeNe / (
        (esis.flights.f2.wavelength_Ne_VII + esis.flights.f2.wavelength_Si_XII) / 2
    )
    ratio = ratio.to(u.dimensionless_unscaled)
    coefficients = result.grating.rulings.spacing.coefficients
    coefficients_euv = euv.grating.rulings.spacing.coefficients
    for power in coefficients:
        assert np.all(np.isclose(coefficients[power], ratio * coefficients_euv[power]))
