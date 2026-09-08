import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
import esis
from . import _parameters


def _channel() -> esis.optics.Instrument:
    return esis.flights.f1.optics.design(num_distribution=0)[dict(channel=1)]


def test_from_instrument():
    instrument = _channel()
    result = esis.optics.DistortionParameters.from_instrument(instrument)
    assert result.yaw_grating.unit == u.arcmin
    assert result.pitch.unit == u.arcsec
    # the sensor terms are measured from the instrument's own placement
    assert result.z_sensor == 0 * u.mm
    assert result.yaw_sensor == instrument.camera.sensor.yaw


def test_pack_round_trip():
    instrument = _channel()
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)
    x = na.pack(parameters).ndarray
    assert x.shape == (15,)
    result = na.unpack(x, parameters)
    for name in ("yaw_grating", "pitch", "z_sensor", "yaw_sensor"):
        assert getattr(result, name) == getattr(parameters, name)


def test_to_instrument():
    instrument = _channel()
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)
    parameters.pitch = 5 * u.arcsec
    parameters.roll_sensor = 0.1 * u.deg
    parameters.x_sensor = 0.3 * u.mm
    parameters.z_sensor = 2 * u.mm
    result = parameters.to_instrument(instrument)

    assert result.pitch == 5 * u.arcsec
    assert result.camera.sensor.roll == 0.1 * u.deg
    assert result.camera.sensor.translation.x == 0.3 * u.mm
    # the compound move: the sensor by the shift, the grating by less
    assert (
        result.camera.sensor.translation.z
        == instrument.camera.sensor.translation.z + 2 * u.mm
    )
    assert np.isclose(
        (result.grating.translation.z - instrument.grating.translation.z).to_value(
            u.mm
        ),
        2 / _parameters.KAPPA_FOCUS,
    )
    # the original is untouched
    assert instrument.pitch == 0 * u.arcsec
    assert instrument.camera.sensor.roll == 0 * u.deg


def test_to_instrument_repeated():
    # applying parameters to an instrument that already carries some must
    # measure the sensor terms from the same origin, not accumulate
    instrument = _channel()
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)
    parameters.z_sensor = 2 * u.mm
    once = parameters.to_instrument(instrument)
    twice = parameters.to_instrument(once)
    assert twice.camera.sensor.translation.z == once.camera.sensor.translation.z
    assert twice.grating.translation.z == once.grating.translation.z
    again = esis.optics.DistortionParameters.from_instrument(twice)
    assert np.isclose(again.z_sensor.to_value(u.mm), 2)


def test_file_round_trip(tmp_path: pathlib.Path):
    instrument = esis.flights.f1.optics.design(num_distribution=0)
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)
    parameters.pitch_sensor = na.ScalarArray(np.arange(4) * 0.1, axes="channel") * u.deg
    path = tmp_path / "parameters.ecsv"
    parameters.to_file(path, metadata=dict(provenance="a test"))
    result = esis.optics.DistortionParameters.from_file(path)
    assert na.shape(result) == dict(channel=4)
    assert np.all(result.pitch_sensor == parameters.pitch_sensor)
    assert np.all(result.yaw_grating == parameters.yaw_grating)


def test_from_file_without_sensor_terms(tmp_path: pathlib.Path):
    # files written before the sensor terms existed load with the defaults
    instrument = esis.flights.f1.optics.design(num_distribution=0)
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)
    path = tmp_path / "parameters.ecsv"
    parameters.to_file(path)
    lines = path.read_text().splitlines()
    # drop the sensor columns from the header and the rows
    import astropy.table

    table = astropy.table.QTable.read(path, format="ascii.ecsv")
    for name in (
        "z_sensor",
        "roll_sensor",
        "pitch_sensor",
        "yaw_sensor",
        "x_sensor",
        "y_sensor",
    ):
        table.remove_column(name)
    table.write(path, format="ascii.ecsv", overwrite=True)
    result = esis.optics.DistortionParameters.from_file(path)
    assert result.z_sensor == 0 * u.mm
    assert result.roll_sensor == 0 * u.deg
    assert np.all(result.yaw_grating == parameters.yaw_grating)
    assert len(lines) > 0
