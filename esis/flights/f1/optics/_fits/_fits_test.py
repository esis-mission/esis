import dataclasses
import pathlib
import astropy.table
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import esis
from . import _fits


def test_idealized():
    instrument = esis.flights.f1.optics.as_built(num_distribution=0)
    result = _fits._idealized(instrument)

    assert isinstance(result.grating.material, optika.materials.Mirror)
    assert isinstance(result.primary_mirror.material, optika.materials.Mirror)
    assert result.filter.material is None
    coefficients = result.grating.rulings.spacing.coefficients
    for k in coefficients:
        assert not isinstance(coefficients[k], na.AbstractUncertainScalarArray)
    for value in (result.grating.translation.z, result.grating.yaw):
        assert not isinstance(value, na.AbstractUncertainScalarArray)
    # and a channel of it can be linearized and masked by the merit
    channel = result[dict(channel=1)]
    channel.wavelength = _fits._wavelength_lines()
    parameters = esis.optics.DistortionParameters.from_instrument(channel)
    assert not isinstance(parameters.yaw_grating, na.AbstractUncertainScalarArray)


def test_wavelength_lines():
    result = _fits._wavelength_lines()
    assert na.shape(result) == dict(wavelength=3)
    assert np.all(np.diff(result.ndarray) > 0)


def test_wavelengths_alignment():
    result = _fits._wavelengths_alignment()
    assert set(result) == set(_fits._LINES_ALIGNMENT)
    for wavelength in result.values():
        assert wavelength.unit.is_equivalent(u.AA)


def test_pupil():
    result = _fits._pupil()
    assert na.shape(result) == dict(pupil_x=2, pupil_y=2)


def test_enforce_shared():
    instrument = esis.flights.f1.optics.design(num_distribution=0)
    parameters = [
        esis.optics.DistortionParameters.from_instrument(instrument[dict(channel=c)])
        for c in range(4)
    ]
    for c, p in enumerate(parameters):
        p.displacement_primary = -c * u.mm
        p.pitch = c * u.arcsec
        p.yaw_grating = p.yaw_grating + c * u.arcmin
    result = _fits.enforce_shared(parameters)
    for p in result:
        assert p.displacement_primary == -1.5 * u.mm
        assert p.pitch == 1.5 * u.arcsec
    # a channel's own terms are untouched, and the inputs are not modified
    for c, (p, q) in enumerate(zip(parameters, result)):
        assert q.yaw_grating == p.yaw_grating
        assert p.displacement_primary == -c * u.mm


def test_stack():
    instrument = esis.flights.f1.optics.design(num_distribution=0)
    parameters = [
        esis.optics.DistortionParameters.from_instrument(instrument[dict(channel=c)])
        for c in range(4)
    ]
    result = _fits._stack(parameters, axis="channel")
    assert na.shape(result) == dict(channel=4)
    for c in range(4):
        assert result.roll[dict(channel=c)] == parameters[c].roll


def test_pointing_relative():
    table = astropy.table.QTable(
        dict(
            frame=np.array([0, 1, 2]),
            pitch=np.array([1.0, 2.0, 3.0]) * u.arcsec,
            yaw=np.array([-1.0, 0.0, 1.0]) * u.arcsec,
            roll=np.array([0.1, 0.2, 0.3]) * u.deg,
        )
    )
    result = _fits.pointing_relative(table, frame=1)
    assert result["pitch"][1] == 0 * u.arcsec
    assert result["yaw"][1] == 0 * u.arcsec
    assert result["roll"][1] == 0 * u.deg
    assert result["pitch"][2] == 1 * u.arcsec
    assert result.meta["frame_reference"] == 1
    # the input is untouched
    assert table["pitch"][1] == 2 * u.arcsec
    with pytest.raises(ValueError):
        _fits.pointing_relative(table, frame=7)


def test_base():
    instrument = _fits._base(None)
    assert isinstance(instrument, esis.optics.Instrument)
    assert na.shape(instrument.wavelength) == dict(wavelength=3)
    # a given instrument is passed through untouched
    assert _fits._base(instrument) is instrument


def test_logger(tmp_path: pathlib.Path):
    log = _fits._logger(tmp_path, "test")
    log("hello")
    assert "hello" in (tmp_path / "test.log").read_text()
    # without a directory the logger only prints
    _fits._logger(None, "test")("hello")


def test_names_absolute():
    instrument = esis.flights.f1.optics.design(num_distribution=0)[dict(channel=1)]
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)
    names = _fits._names_absolute(parameters)
    assert names == tuple(f.name for f in dataclasses.fields(parameters))


def test_frames_by_axis():
    observation = na.ScalarArray(
        np.arange(2 * 3 * 4).reshape(2, 3, 4) * u.DN,
        axes=("channel", "detector_x", "detector_y"),
    )
    frames = _fits._frames_by_axis(observation, num_channel=2)
    assert len(frames) == 2
    assert frames[1].shape == (4, 3)
    assert frames[1][0, 0] == 12
