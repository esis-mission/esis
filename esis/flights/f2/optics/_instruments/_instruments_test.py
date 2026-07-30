import pytest
import numpy as np
import astropy.units as u
import esis


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_proposed(num_distribution: int):
    result = esis.flights.f2.optics.design_proposed(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_guess(num_distribution: int):
    result = esis.flights.f2.optics.design_guess(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_single(num_distribution: int):
    result = esis.flights.f2.optics.design_single(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design(num_distribution: int):
    result = esis.flights.f2.optics.design(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)


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
