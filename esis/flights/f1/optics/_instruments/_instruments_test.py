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

    design = esis.flights.f1.optics.design(num_distribution=0)
    assert np.all(
        np.sign(na.value(result.grating.sag.radius).ndarray)
        == np.sign(na.value(design.grating.sag.radius))
    )


def test_as_built_focus():
    """The as-built model must actually focus (guards the sag sign convention)."""
    result = esis.flights.f1.optics.as_built(num_distribution=0)
    rays = result.system.rayfunction_default.outputs
    axis_pupil = ("pupil_x", "pupil_y")
    spread = rays.position - rays.position.mean(axis=axis_pupil)
    r2 = (spread.x**2 + spread.y**2).mean(axis=axis_pupil)
    rms = np.sqrt(np.mean(na.nominal(r2).ndarray))
    assert rms < 1 * u.mm


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_distortion_fit(num_distribution: int):
    result = esis.flights.f1.optics.distortion_fit(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)

    # spot-check the loaded reference against the values fit on 2026-07-07,
    # guarding the data file and its loader against silent drift
    assert np.allclose(
        result.pitch.ndarray.to(u.arcsec).value,
        [-19.7717, -21.25244, -22.08457, -21.68604],
    )
    assert np.allclose(
        result.grating.rulings.spacing.coefficients[0].ndarray.to(u.um).value,
        [0.3854, 0.3859, 0.3855, 0.3863],
    )
    assert np.allclose(
        result.primary_mirror.translation.z.ndarray.to(u.mm).value,
        [5.649, 0.02207, 2.795, 1.616],
    )


def test_distortion_fit_time():
    reference = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    result = esis.flights.f1.optics.distortion_fit(
        num_distribution=0,
        axis_time="time",
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)

    num_time = 30
    assert na.shape(result.pitch) == dict(channel=4, time=num_time)
    assert na.shape(result.yaw) == dict(channel=4, time=num_time)
    assert na.shape(result.roll) == dict(channel=4, time=num_time)

    # the pointing must be indexable per frame, and the yaw drift crosses the
    # time=15 reference from positive to negative during the flight
    frame = result[dict(time=15)]
    offset_yaw = frame.yaw - reference.yaw
    assert np.all(np.abs(offset_yaw) < 1 * u.arcsec)
    offset_first = result[dict(time=0)].yaw - reference.yaw
    offset_last = result[dict(time=~0)].yaw - reference.yaw
    assert np.all(offset_first > 1 * u.arcsec)
    assert np.all(offset_last < -1 * u.arcsec)


def test_distortion_fit_bounds():
    instrument = esis.flights.f1.optics.design(num_distribution=0)
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)

    lower, upper = esis.flights.f1.optics.distortion_fit_bounds(parameters)

    assert isinstance(lower, esis.optics.DistortionParameters)
    assert isinstance(upper, esis.optics.DistortionParameters)

    x = na.pack(parameters).ndarray
    x_lower = na.pack(lower).ndarray
    x_upper = na.pack(upper).ndarray

    assert x_lower.shape == x.shape
    assert x_upper.shape == x.shape
    assert np.all(x_lower <= x)
    assert np.all(x <= x_upper)
