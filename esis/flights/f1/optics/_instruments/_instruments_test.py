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


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_as_built_radius_grating(num_distribution: int):
    """The gratings which were built are concave, as the designed ones are."""
    result = esis.flights.f1.optics.as_built(
        num_distribution=num_distribution,
    )
    design = esis.flights.f1.optics.design(
        num_distribution=num_distribution,
    )

    radius = result.grating.sag.radius

    # a concave surface has a negative radius of curvature in the sag
    # convention, and a grating stored with a positive one is convex, which
    # sends the light somewhere else entirely
    assert np.all(radius < 0)

    # The gratings which were built are the gratings which were designed, to
    # within a millimeter. Compared against the radius the design asked for
    # rather than against a sample of its tolerance, since the samples are
    # gratings which might have been made and these are the ones which were.
    radius_design = na.nominal(design.grating.sag.radius)
    assert np.all(np.abs(radius - radius_design) < 1 * u.mm)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_distortion_fit(num_distribution: int):
    result = esis.flights.f1.optics.distortion_fit(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
