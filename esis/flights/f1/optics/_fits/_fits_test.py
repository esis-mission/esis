import dataclasses
import types
import io
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
            drift_x=np.array([0.3, 0.0, -0.2]) * u.pix,
        )
    )
    result = _fits.pointing_relative(table, frame=1)
    # the drift is measured from the reference frame already
    assert np.all(result["drift_x"] == table["drift_x"])
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
    # a fit from scratch starts from the regular octagon of the design
    assert instrument.field_stop.vertices is None
    # a given instrument is passed through untouched
    assert _fits._base(instrument) is instrument


def test_base_measured(monkeypatch: pytest.MonkeyPatch):
    from esis.flights.f1.optics._instruments import _instruments

    path = _instruments._directory_data / "field_stop.ecsv"
    monkeypatch.setattr(_fits, "path_field_stop", path)
    instrument = _fits._base(None)
    assert na.shape(instrument.field_stop.vertices) == dict(vertex=8)


def test_logger(tmp_path: pathlib.Path):
    log = _fits._logger(tmp_path, "test")
    log("hello")
    assert "hello" in (tmp_path / "test.log").read_text()
    # without a directory the logger only prints
    _fits._logger(None, "test")("hello")


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
    # the raw reading of a frame nobody measured is missing, not zero
    assert np.all(np.isnan(_fits._drift_shift(drift, 0, 25)))


def test_drift_smooth_axis_missing():
    """
    A frame that lost one axis's sides leaves that axis out of its fit.

    Were the missing axis read as zero, the cubic through a steady motion
    would bend toward it.
    """
    rows = []
    for t in range(20):
        sides = (0, 1, 2, 3) if t != 10 else (0, 1)
        for side in sides:
            truth = 0.1 * t if side < 2 else 1.0 + 0.05 * t
            rows.append((0, t, side, truth, 10))
    drift = astropy.table.QTable(
        rows=rows,
        names=("channel", "frame", "side", "shift", "num"),
    )
    drift["shift"] = drift["shift"] * u.pix
    x, y = _fits._drift_shift(drift, 0, 10)
    assert np.isclose(x, 1.0) and np.isnan(y)
    shift = _fits._drift_smooth(drift, num_channel=1, degree=3)
    assert np.allclose(shift(0, 10), [1.0, 1.5])
    # an axis measured in too few frames cannot be fit
    few = drift[(np.asarray(drift["side"]) < 2) | (np.asarray(drift["frame"]) < 3)]
    with pytest.raises(ValueError, match="measured in y"):
        _fits._drift_smooth(few, num_channel=1, degree=3)


def _pointing_table() -> astropy.table.QTable:
    """Make a pointing table of two frames with every column the fit writes."""
    return astropy.table.QTable(
        dict(
            frame=np.array([3, 15]),
            pitch=np.array([1.0, 0.0]) * u.arcsec,
            yaw=np.array([-2.0, 0.0]) * u.arcsec,
            roll=np.array([0.1, 0.0]) * u.deg,
            z_primary=np.array([[0.01, 0.02, 0.03, 0.04], [0, 0, 0, 0]]) * u.mm,
            yaw_grating=np.array([[0.1, 0.2, 0.3, 0.4], [0, 0, 0, 0]]) * u.arcmin,
            pitch_grating=np.array([[-0.1, -0.2, -0.3, -0.4], [0, 0, 0, 0]]) * u.arcmin,
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
    assert np.isclose(p.pitch - base.pitch, 1.0 * u.arcsec)
    assert np.isclose(p.yaw - base.yaw, -2.0 * u.arcsec)
    assert np.isclose(p.roll - base.roll, 0.1 * u.deg)
    # the focus of the channel's own sector of the primary
    assert np.isclose(p.z_primary - base.z_primary, 0.03 * u.mm)
    assert np.isclose(p.yaw_grating - base.yaw_grating, 0.3 * u.arcmin)
    assert np.isclose(p.pitch_grating - base.pitch_grating, -0.3 * u.arcmin)

    # the reference frame is the reference
    s = _fits.frame_parameters(reference, pointing, frame=15, channel=0)
    assert s.pitch == _fits._channel(reference, 0).pitch

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


def test_environment():
    environment = _fits._environment()
    assert environment["python"]
    assert environment["platform"]
    assert "optika" in environment["packages"]
    assert "optika" in environment["checkouts"]
    assert "euv-snapshot-imaging-spectrograph" in environment["checkouts"]
    # it rides in the metadata of every committed table, so it has to survive
    # a trip through ECSV
    table = astropy.table.QTable(dict(a=[1, 2]), meta=dict(environment=environment))
    buffer = io.StringIO()
    table.write(buffer, format="ascii.ecsv")
    back = astropy.table.QTable.read(buffer.getvalue(), format="ascii.ecsv")
    assert back.meta["environment"]["packages"] == environment["packages"]


def test_frame_times():
    times = _fits._frame_times()
    assert len(times) == 30
    for record in times:
        assert record["start"] < record["end"]
        assert 0 < record["exposure"] < 60


def test_read_table(tmp_path: pathlib.Path):
    table = _pointing_table()
    assert _fits._read_table(None) is None
    assert _fits._read_table(table) is table
    path = tmp_path / "pointing.ecsv"
    table.write(path, format="ascii.ecsv")
    back = _fits._read_table(path)
    assert list(back["frame"]) == [3, 15]
    assert _fits._read_table(str(path))["pitch"].unit == u.arcsec


def test_map_channels():
    assert _fits._map_channels(abs, [-1, 2, -3], workers=1) == [1, 2, 3]
    assert _fits._map_channels(abs, [-1, 2, -3], workers=2) == [1, 2, 3]


def test_stamp():
    stamp = _fits._stamp(seed=3)
    assert set(stamp) == {"date", "environment", "seed"}
    assert stamp["seed"] == 3
    assert stamp["date"][:4] == str(_fits.datetime.date.today().year)
    assert "optika" in stamp["environment"]["checkouts"]


def test_mean_column():
    table = astropy.table.QTable(
        dict(a=np.array([1.0, np.nan, 3.0]) * u.pix, b=np.array([2.0, 4.0, np.nan]))
    )
    assert _fits._mean_column(table, "a") == 2.0
    assert _fits._mean_column(table, "b") == 3.0


def _drift_table(num_frame: int) -> astropy.table.QTable:
    rows = [
        (c, t, side, 0.1 * t * (1 + c), 10)
        for c in range(2)
        for t in range(num_frame)
        for side in range(4)
    ]
    drift = astropy.table.QTable(
        rows=rows, names=("channel", "frame", "side", "shift", "num")
    )
    drift["shift"] = drift["shift"] * u.pix
    return drift


def test_window_drift():
    """Each side moves by the median departure of its rows from their own medians."""
    rows = []
    for t, shift in enumerate((0.0, 0.2, -0.1)):
        for index in range(5):
            # every row sits at its own place; the side moves them together
            rows.append((0, t, 0, 1, index, 100 + index + shift, 0.1))
            rows.append((0, t, 0, 3, index, 50 - index - shift, 0.1))
        # a crossing whose error is too large is left out
        rows.append((0, t, 0, 1, 9, 500.0, 5.0))
    crossings = astropy.table.QTable(
        rows=rows,
        names=("channel", "frame", "line", "side", "index", "position", "error"),
    )
    crossings["position"] = crossings["position"] * u.pix
    crossings["error"] = crossings["error"] * u.pix
    drift = _fits.window_drift(crossings, num_channel=1)
    right = drift[drift["side"] == 1]
    bottom = drift[drift["side"] == 3]
    # each against the flight median of its rows, which is frame 0's here
    assert np.allclose(right["shift"].to_value(u.pix), [0.0, 0.2, -0.1])
    assert np.allclose(bottom["shift"].to_value(u.pix), [0.0, -0.2, 0.1])
    assert list(right["num"]) == [5, 5, 5]


def test_refer_to_frame():
    """The smoothed drift is zero at the frame, and the edges move by it."""
    drift = _drift_table(8)
    edges = astropy.table.QTable(
        rows=[(c, 0, side, 0, 10.0) for c in range(2) for side in range(4)],
        names=("channel", "line", "side", "index", "position"),
    )
    edges["position"] = edges["position"] * u.pix
    before = _fits._drift_smooth(drift, num_channel=2, degree=1)
    moved, referred = _fits._refer_to_frame(edges, drift, frame=5, degree=1)
    after = _fits._drift_smooth(referred, num_channel=2, degree=1)
    for c in range(2):
        assert np.allclose(after(c, 5), 0, atol=1e-9)
        # the drift keeps its shape
        assert np.allclose(after(c, 7) - after(c, 2), before(c, 7) - before(c, 2))
        # the edges are where the drift put them at the frame
        x, y = before(c, 5)
        mine = moved[moved["channel"] == c]
        horizontal = np.asarray(mine["side"]) <= 1
        assert np.allclose(mine["position"][horizontal].value, 10 + x)
        assert np.allclose(mine["position"][~horizontal].value, 10 + y)
    # the inputs are untouched
    assert np.all(edges["position"] == 10 * u.pix)


def test_drift_consistent():
    """A stage that smooths the drift another way than it was referred raises."""
    drift = _drift_table(8)
    drift.meta.update(drift_degree=3, frame_reference=15)
    _fits._drift_consistent(drift, degree=3, frame_reference=15)
    with pytest.raises(ValueError, match="drift_degree=3"):
        _fits._drift_consistent(drift, degree=2, frame_reference=15)
    with pytest.raises(ValueError, match="frame_reference=15"):
        _fits._drift_consistent(drift, degree=3, frame_reference=10)
    # a table that names neither is taken as it is
    _fits._drift_consistent(_drift_table(8), degree=2, frame_reference=10)


def test_drift_smooth_too_few_frames():
    with pytest.raises(ValueError, match="too few"):
        _fits._drift_smooth(_drift_table(3), num_channel=2, degree=3)


def _reference_channels() -> list[esis.optics.DistortionParameters]:
    from esis.flights.f1.optics._instruments import _instruments

    reference = esis.optics.DistortionParameters.from_file(
        _instruments._directory_data / "distortion_reference.ecsv"
    )
    return [_fits._channel(reference, c) for c in range(4)]


def test_defocus_pattern():
    """
    Check the sector pattern: a defocus moves each channel's sky its own way.

    Each channel looks through its own sector of the primary, so a defocus
    translates its image along its own dispersion; against the anchor the
    pattern is the difference of two such vectors, and the anchor's own is zero.
    """
    instrument = _fits._base(None)
    reference = _reference_channels()
    wavelengths = _fits._wavelengths_alignment()
    own = _fits._defocus_pattern(instrument, reference, wavelengths, relative=False)
    relative = _fits._defocus_pattern(instrument, reference, wavelengths)
    assert own.shape == relative.shape == (4, 2)
    assert np.allclose(relative, own - own[1])
    assert np.allclose(relative[1], 0)
    # about a pixel and a half per hundred microns, see Instrument.kappa_focus
    lengths = np.hypot(own[:, 0], own[:, 1])
    assert np.all((lengths > 5) & (lengths < 50))
    # the channels disperse 45 degrees apart, so no two vectors are parallel
    for c in range(4):
        cosine = np.dot(own[c], own[(c + 1) % 4]) / (lengths[c] * lengths[(c + 1) % 4])
        assert abs(cosine) < 0.9


def test_solve_sector_focus():
    # channels 0, 2 and 3 against the anchor, channel 1
    pattern = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [-1.0, 0.5]])
    design = np.zeros((6, 4))
    for i, c in enumerate((0, 2, 3)):
        design[2 * i : 2 * i + 2, c] = pattern[c]
        design[2 * i : 2 * i + 2, 1] = -pattern[1]
    truth = np.array([0.01, -0.02, 0.03, 0.005])
    measured = -design @ truth
    z, residual = _fits._solve_sector_focus(design, measured)
    assert np.allclose(z, truth)
    assert residual == pytest.approx(0, abs=1e-10)
    # a channel with no tiles has no focus, which is not the same as zero
    measured[2:4] = np.nan
    z, _ = _fits._solve_sector_focus(design, measured)
    assert np.isnan(z[2])
    assert np.all(np.isfinite(z[[0, 1, 3]]))
    # too few shifts for any of them
    measured[:] = np.nan
    z, residual = _fits._solve_sector_focus(design, measured)
    assert np.all(np.isnan(z)) and np.isnan(residual)


def test_pixels_per_arcsec():
    position = np.zeros(2)
    jacobians = [
        (position, np.array([[3.0, 0.0], [4.0, 2.0]])),
        (position, np.array([[5.0, 0.0], [0.0, 4.0]])),
    ]
    assert np.allclose(_fits._pixels_per_arcsec(jacobians), [5.0, 3.0])


def test_dispersion_and_footprints():
    """A redshift at He I displaces a channel's image by about 19 km/s per pixel."""
    instrument = _fits._base(None)
    p = _reference_channels()[0]
    model = p.to_instrument(instrument[dict(channel=0)])
    linear = model.system.linearize(
        wavelength=model.wavelength, degree=2, field_stop=True
    )
    footprints = _fits._footprints(linear, model.wavelength)
    assert len(footprints) == na.shape(model.wavelength)["wavelength"]
    direction, velocity = _fits._dispersion(
        linear, _fits._wavelengths_alignment()["He I"]
    )
    assert np.linalg.norm(direction) == pytest.approx(1)
    assert velocity == pytest.approx(18.9, abs=1.0)


def test_drift_sensitivity():
    """The grating steers the window by whole pixels per arcminute."""
    instrument = _fits._base(None)
    reference = _reference_channels()
    merits = [
        types.SimpleNamespace(
            linearize=lambda m: m.system.linearize(
                wavelength=m.wavelength, degree=2, field_stop=True
            )
        )
        for _ in reference
    ]
    sensitivity = _fits._drift_sensitivity(instrument, reference, merits)
    assert sensitivity.shape == (4, 2)
    assert np.all(np.abs(sensitivity) > 1)
    # the four gratings are alike, so the channels agree
    assert (
        np.abs(sensitivity - sensitivity.mean(axis=0)).max()
        < 0.2 * np.abs(sensitivity).mean()
    )
