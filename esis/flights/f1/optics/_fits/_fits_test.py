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


def test_channel():
    from esis.flights.f1.optics._instruments import _instruments

    stacked = esis.optics.DistortionParameters.from_file(
        _instruments._directory_data / "distortion_reference.ecsv"
    )
    one = _fits._channel(stacked, 2)
    assert "channel" not in na.shape(one.yaw_grating)
    assert one.yaw_grating == stacked.yaw_grating[dict(channel=2)]
    assert one.pitch == stacked.pitch[dict(channel=2)]
    # a scalar field is passed through
    scalar = dataclasses.replace(stacked, degradation=1 * u.dimensionless_unscaled)
    assert _fits._channel(scalar, 0).degradation == 1


def test_drift_smooth():
    """
    Follow a smooth motion through the jitter of the single-frame measurements.

    Outside the measured frames the nearest measured value is held.
    """
    frames = np.arange(0, 20)
    rng = np.random.default_rng(0)
    rows = []
    for c in range(2):
        for t in frames:
            truth_x = 0.05 * (t - 10) + 0.002 * (t - 10) ** 2 * c
            truth_y = -0.08 * (t - 10)
            for side, truth in ((0, truth_x), (1, truth_x), (2, truth_y), (3, truth_y)):
                rows.append((c, int(t), side, truth + rng.normal(0, 0.1), 10))
    drift = astropy.table.QTable(
        rows=rows,
        names=("channel", "frame", "side", "shift", "num"),
    )
    drift["shift"] = drift["shift"] * u.pix

    shift = _fits._drift_smooth(drift, num_channel=2, degree=3)

    for c in range(2):
        for t in (5, 10, 15):
            truth_x = 0.05 * (t - 10) + 0.002 * (t - 10) ** 2 * c
            truth_y = -0.08 * (t - 10)
            x, y = shift(c, t)
            # the mean of two sides with 0.1 px noise leaves 0.07 px per frame;
            # a cubic through twenty frames does better than that
            assert abs(x - truth_x) < 0.1
            assert abs(y - truth_y) < 0.1
    # outside the measured frames the nearest measured value is held
    assert np.allclose(shift(0, 25), shift(0, 19))
    assert np.allclose(shift(0, -5), shift(0, 0))
    # the raw reading of a frame nobody measured is zero, which is the
    # jump the smoothing exists to remove
    assert np.allclose(_fits._drift_shift(drift, 0, 25), 0)


def _pointing_table() -> astropy.table.QTable:
    """Make a pointing table of two frames with every column the fit writes."""
    return astropy.table.QTable(
        dict(
            frame=np.array([3, 15]),
            pitch=np.array([1.0, 0.0]) * u.arcsec,
            yaw=np.array([-2.0, 0.0]) * u.arcsec,
            roll=np.array([0.1, 0.0]) * u.deg,
            z_primary=np.array([0.01, 0.0]) * u.mm,
            yaw_grating=np.array([[0.1, 0.2, 0.3, 0.4], [0, 0, 0, 0]]) * u.arcmin,
            pitch_grating=np.array([[-0.1, -0.2, -0.3, -0.4], [0, 0, 0, 0]]) * u.arcmin,
            pitch_channel=np.array([[0.3, -0.1, -0.1, -0.1], [0, 0, 0, 0]]) * u.arcsec,
            yaw_channel=np.array([[0.2, 0.0, -0.1, -0.1], [0, 0, 0, 0]]) * u.arcsec,
        )
    )


def test_frame_parameters():
    from esis.flights.f1.optics._instruments import _instruments

    reference = esis.optics.DistortionParameters.from_file(
        _instruments._directory_data / "distortion_reference.ecsv"
    )
    pointing = _pointing_table()
    base = _fits._channel(reference, 2)

    p = _fits.frame_parameters(reference, pointing, frame=3, channel=2)
    # the payload's pointing and the channel's own offset add up
    assert np.isclose(p.pitch - base.pitch, (1.0 - 0.1) * u.arcsec)
    assert np.isclose(p.yaw - base.yaw, (-2.0 - 0.1) * u.arcsec)
    assert np.isclose(p.roll - base.roll, 0.1 * u.deg)
    assert np.isclose(p.z_primary - base.z_primary, 0.01 * u.mm)
    assert np.isclose(p.yaw_grating - base.yaw_grating, 0.3 * u.arcmin)
    assert np.isclose(p.pitch_grating - base.pitch_grating, -0.3 * u.arcmin)

    # without the channel's own offset only the payload's pointing is applied
    q = _fits.frame_parameters(reference, pointing, 3, 2, channel_offsets=False)
    assert np.isclose(q.pitch - base.pitch, 1.0 * u.arcsec)
    assert np.isclose(q.yaw - base.yaw, -2.0 * u.arcsec)

    # a table from before the defocus, the drift and the registration
    old = pointing["frame", "pitch", "yaw", "roll"]
    r = _fits.frame_parameters(reference, old, frame=3, channel=2)
    assert np.isclose(r.pitch - base.pitch, 1.0 * u.arcsec)
    assert r.z_primary == base.z_primary
    assert r.yaw_grating == base.yaw_grating

    # the reference frame is the reference
    s = _fits.frame_parameters(reference, pointing, frame=15, channel=0)
    assert s.pitch == _fits._channel(reference, 0).pitch

    # one focus per channel's sector of the primary
    sectors = pointing.copy()
    sectors["z_primary"] = np.array([[0.01, 0.02, 0.03, 0.04], [0, 0, 0, 0]]) * u.mm
    v = _fits.frame_parameters(reference, sectors, frame=3, channel=2)
    assert np.isclose(v.z_primary - base.z_primary, 0.03 * u.mm)

    with pytest.raises(ValueError):
        _fits.frame_parameters(reference, pointing, frame=7, channel=0)


def test_fit_through_reference():
    """
    Follow a line that vanishes at the reference frame past a bad frame.

    The dark frames at the ends of the flight measure wildly, and a frame
    with no measurement at all is NaN; neither may bend the fit.
    """
    frames = np.arange(30)
    rng = np.random.default_rng(1)
    truth = 0.02 * (frames - 15)
    value = truth + rng.normal(0, 0.01, size=frames.size)
    value[29] = 5.0
    value[28] = np.nan

    fitted, used, scatter = _fits._fit_through_reference(frames, value, 15, floor=0.01)

    assert fitted[15] == 0
    assert np.abs(fitted - truth).max() < 0.01
    assert not used[29]
    assert not used[28]
    assert used[:28].all()
    assert scatter < 0.02

    # too few frames to fit leave the offsets at zero
    few = np.full(frames.size, np.nan)
    few[3] = 1.0
    fitted, used, scatter = _fits._fit_through_reference(frames, few, 15)
    assert np.all(fitted == 0)

    # a curve needs the degree to follow it
    curve = 0.02 * (frames - 15) + 0.003 * (frames - 15) ** 2
    fitted, _, _ = _fits._fit_through_reference(frames, curve, 15, degree=2)
    assert np.abs(fitted - curve).max() < 1e-9


def test_apply_coregistration():
    pointing = _pointing_table()["frame", "pitch", "yaw", "roll"]
    coregistration = astropy.table.QTable(
        dict(
            frame=np.array([15, 3, 20]),
            pitch_channel=np.array([[0, 0, 0, 0], [1, 2, 3, 4], [5, 6, 7, 8]])
            * u.arcsec,
            yaw_channel=np.array([[0, 0, 0, 0], [-1, -2, -3, -4], [9, 9, 9, 9]])
            * u.arcsec,
        )
    )
    coregistration.meta["provenance"] = "a test"

    result = _fits.apply_coregistration(pointing, coregistration)

    # the rows are matched by frame, not by position
    assert np.all(result["pitch_channel"][0] == [1, 2, 3, 4] * u.arcsec)
    assert np.all(result["yaw_channel"][0] == [-1, -2, -3, -4] * u.arcsec)
    assert np.all(result["pitch_channel"][1] == 0 * u.arcsec)
    assert "a test" in result.meta["coregistration"]
    # the input is untouched
    assert "pitch_channel" not in pointing.colnames

    with pytest.raises(ValueError):
        _fits.apply_coregistration(pointing, coregistration[:1])


def test_pointing_pattern():
    """
    Check that a channel's own pointing moves its sky the way the payload's does.

    The pitch and the yaw are angles on the sky, so an arcsecond of either
    moves every channel's image by the same vector there, a plate scale's
    worth of pixels, whichever way the channel disperses.
    """
    from esis.flights.f1.optics._instruments import _instruments

    reference = esis.optics.DistortionParameters.from_file(
        _instruments._directory_data / "distortion_reference.ecsv"
    )
    parameters = [_fits._channel(reference, c) for c in range(4)]
    pattern = _fits._pointing_pattern(
        _fits._base(None), parameters, _fits._wavelengths_alignment()
    )
    assert pattern.shape == (4, 2, 2)
    for c in range(4):
        # the two terms move the image at right angles, a pixel and a third
        # per arcsecond
        assert abs(np.linalg.det(pattern[c])) == pytest.approx(1.7, abs=0.2)
        assert np.abs(pattern[c] - pattern[1]).max() < 0.05
