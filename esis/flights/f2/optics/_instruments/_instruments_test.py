import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import esis


def _assert_rulings_centered(instrument: esis.optics.abc.AbstractInstrument):
    """
    Assert that the ruling coefficients are centered on their nominal values.

    The samples of each coefficient should be distributed around its nominal
    value, not around a value that it replaced.
    """
    coefficients = instrument.grating.rulings.spacing.coefficients
    for power in coefficients:
        c = coefficients[power]
        if not isinstance(c, na.AbstractUncertainScalarArray):
            continue
        assert not isinstance(c.nominal, na.AbstractUncertainScalarArray)
        axis = c.axis_distribution
        offset = np.abs(np.mean(c.distribution, axis=axis) - c.nominal)
        spread = np.max(c.distribution, axis=axis) - np.min(c.distribution, axis=axis)
        assert np.all(offset <= spread)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_proposed(num_distribution: int):
    result = esis.flights.f2.optics.design_proposed(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
    _assert_rulings_centered(result)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_guess(num_distribution: int):
    result = esis.flights.f2.optics.design_guess(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
    _assert_rulings_centered(result)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_single(num_distribution: int):
    result = esis.flights.f2.optics.design_single(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
    _assert_rulings_centered(result)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design(num_distribution: int):
    result = esis.flights.f2.optics.design(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
    _assert_rulings_centered(result)


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
        coefficient = coefficients[power]
        coefficient_euv = coefficients_euv[power]
        if num_distribution != 0:
            coefficient = coefficient.nominal
            coefficient_euv = coefficient_euv.nominal
        assert np.all(np.isclose(coefficient, ratio * coefficient_euv))
