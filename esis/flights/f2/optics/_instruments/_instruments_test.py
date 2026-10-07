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
