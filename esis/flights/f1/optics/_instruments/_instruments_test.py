import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import esis


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_full(num_distribution: int):
    result = esis.flights.f1.optics.design_full(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design(num_distribution: int):
    result = esis.flights.f1.optics.design(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_single(num_distribution: int):
    result = esis.flights.f1.optics.design_single(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_as_built(num_distribution: int):
    result = esis.flights.f1.optics.as_built(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)


def test_design_focus_grating():
    """The design is already focused, so focusing it should move nothing."""
    design = esis.flights.f1.optics.design(num_distribution=0)
    result = design.focus_grating(
        wavelength=esis.flights.f1.spectrum.O_V.wavelength,
    )
    dz = result.grating.translation.z - design.grating.translation.z
    assert np.all(np.abs(dz) < 10 * u.um)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_as_built_focused(num_distribution: int):
    result = esis.flights.f1.optics.as_built_focused(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)

    as_built = esis.flights.f1.optics.as_built(
        num_distribution=num_distribution,
    )
    dz = na.nominal(result.grating.translation.z) - na.nominal(
        as_built.grating.translation.z
    )
    # the measured radii are shorter than the design radius, so every
    # grating moves toward the field stop by a fraction of a millimeter
    assert np.all(dz > 0.3 * u.mm)
    assert np.all(dz < 0.9 * u.mm)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_distortion_fit(num_distribution: int):
    result = esis.flights.f1.optics.distortion_fit(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
