from typing import Any
import pytest
import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na
import sdo
import esis


def test_scene_aia(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Check the images are found by sdo.aia.open and passed on to scene_filtergram.

    :func:`sdo.aia.open` is replaced by images from the archive of
    :func:`esis.flights.f1.data.path_aia`, so the test doesn't depend on the
    JSOC.
    """
    # Names other than the defaults, so an argument that isn't passed through
    # changes the result or the call.
    # The detector axes only need to reach the stand-in for `sdo.aia.open`,
    # which ignores them.
    axis_time = "t"
    axis_wavelength = "wavelength"
    axis_detector_x = "x"
    axis_detector_y = "y"
    wavelength = na.ScalarArray([304, 193] * u.AA, axis_wavelength)
    time_start = astropy.time.Time("2019-09-30T18:07:00")
    time_stop = astropy.time.Time("2019-09-30T18:08:00")
    limit = 1

    files = esis.flights.f1.data.path_aia(
        wavelength=wavelength,
        axis_time=axis_time,
        time_start=time_start,
        time_stop=time_stop,
        limit=limit,
    )
    filtergram = sdo.aia.Filtergram.from_fits(
        path=sdo.aia.prep(files, cache=None),
        wavelength=wavelength,
        axis_time=axis_time,
        axis_wavelength=axis_wavelength,
    )

    calls = []

    def open_archive(**kwargs: Any) -> sdo.aia.Filtergram:
        calls.append(kwargs)
        return filtergram

    monkeypatch.setattr(sdo.aia, "open", open_archive)

    kwargs_scene = dict(
        wavelength_new=na.ScalarArray([630, 609] * u.AA, axis_wavelength),
        radiance=na.ScalarArray([1, 2] * u.erg / u.cm**2 / u.sr / u.s, axis_wavelength),
        width_doppler=30 * u.km / u.s,
        axis_velocity="v",
        num_velocity=3,
        num_std=2,
    )

    result = esis.data.synth.scene_aia(
        time_start=time_start,
        time_stop=time_stop,
        wavelength_aia=wavelength,
        axis_time=axis_time,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
        limit=limit,
        **kwargs_scene,
    )

    assert len(calls) == 1
    (call,) = calls
    assert call.keys() == {
        "time_start",
        "time_stop",
        "wavelength",
        "axis_time",
        "axis_detector_x",
        "axis_detector_y",
        "limit",
    }
    assert call["time_start"] == time_start
    assert call["time_stop"] == time_stop
    assert call["wavelength"] is wavelength
    assert call["axis_time"] == axis_time
    assert call["axis_detector_x"] == axis_detector_x
    assert call["axis_detector_y"] == axis_detector_y
    assert call["limit"] == limit

    expected = esis.data.synth.scene_filtergram(filtergram=filtergram, **kwargs_scene)
    assert result.outputs.shape == expected.outputs.shape
    assert np.all(result.inputs.time == expected.inputs.time)
    assert np.all(result.inputs.wavelength == expected.inputs.wavelength)
    assert np.all(result.outputs == expected.outputs)
