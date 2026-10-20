from typing import Any
import pytest
import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na
import sdo
import esis


def _filtergram(**kwargs: Any) -> sdo.aia.Filtergram:
    """
    Stand in for :func:`sdo.aia.open`.

    One image in each channel, twenty pixels a minute of arc across on a
    side, centered on the Sun.
    """
    time = astropy.time.Time("2016-07-15T00:00:00") + np.arange(6) * u.s
    position = na.Cartesian2dVectorLinearSpace(
        start=-600 * u.arcsec,
        stop=600 * u.arcsec,
        axis=na.Cartesian2dVectorArray(
            kwargs["axis_detector_x"],
            kwargs["axis_detector_y"],
        ),
        num=21,
    ).explicit
    shape = dict(time=1, wavelength=6, **position.shape)
    shape[kwargs["axis_detector_x"]] -= 1
    shape[kwargs["axis_detector_y"]] -= 1
    return sdo.aia.Filtergram(
        inputs=na.TemporalSpectralPositionalVectorArray(
            time=na.ScalarArray(time, axes="wavelength").add_axes("time"),
            wavelength=kwargs["wavelength"],
            position=position,
        ),
        outputs=na.ScalarArray(np.ones(tuple(shape.values())), tuple(shape)) * u.DN,
        axis_detector_x=kwargs["axis_detector_x"],
        axis_detector_y=kwargs["axis_detector_y"],
        timedelta=2 * u.s,
    )


def test_scene_dem(monkeypatch: pytest.MonkeyPatch):
    """
    Check the images around the boresight are made into a scene.

    They are passed on to :func:`esis.data.synth.scene_dem` with the passband
    of ESIS-II, and the positions of the scene are offsets from the boresight.
    """
    calls_open = []
    calls_scene = []

    def open_aia(**kwargs: Any) -> sdo.aia.Filtergram:
        calls_open.append(kwargs)
        return _filtergram(**kwargs)

    def scene_dem(**kwargs: Any) -> na.FunctionArray:
        calls_scene.append(kwargs)
        filtergram = kwargs["filtergram"]
        return na.FunctionArray(
            inputs=na.TemporalDopplerPositionalVectorArray(
                time=filtergram.inputs.time,
                wavelength=0 * u.AA,
                wavelength_rest=0 * u.AA,
                position=filtergram.inputs.position,
            ),
            outputs=filtergram.outputs.sum("wavelength"),
        )

    monkeypatch.setattr(sdo.aia, "open", open_aia)
    monkeypatch.setattr(esis.data.synth, "scene_dem", scene_dem)

    time = "2020-01-01T00:00:00"
    center = na.Cartesian2dVectorArray(-200, 100) * u.arcsec
    width = 400 * u.arcsec

    result = esis.flights.f2.data.synth.scene_dem(
        time=time,
        center=center,
        width=width,
        axis_detector_x="dx",
        axis_detector_y="dy",
        fraction_min=0.01,
    )

    (call_open,) = calls_open
    assert call_open["time_start"] == time
    assert call_open["register"]
    assert np.all(call_open["wavelength"] == [94, 131, 171, 193, 211, 335] * u.AA)

    (call_scene,) = calls_scene
    instrument = esis.flights.f2.optics.design_single(num_distribution=0)
    assert call_scene["wavelength_min"] == instrument.wavelength_min
    assert call_scene["wavelength_max"] == instrument.wavelength_max
    assert call_scene["fraction_min"] == 0.01

    # The pixels are a minute of arc across, so the square 400 arcseconds
    # across holds the centers of six of them on each side, or seven where
    # the square starts on an edge.
    filtergram = call_scene["filtergram"]
    position = filtergram.inputs.position.cell_centers()
    offset = position - center
    assert np.all(np.abs(offset.x) <= width / 2)
    assert np.all(np.abs(offset.y) <= width / 2)
    assert na.shape(filtergram.outputs) == dict(wavelength=6, dx=7, dy=7)

    # positions relative to the boresight
    expected = filtergram.inputs.position - center
    assert np.all(result.inputs.position == expected)


def test_scene_dem_outside(monkeypatch: pytest.MonkeyPatch):
    """A square which misses the images is an error, not an empty scene."""
    monkeypatch.setattr(sdo.aia, "open", _filtergram)

    with pytest.raises(ValueError, match="No AIA pixel"):
        esis.flights.f2.data.synth.scene_dem(
            center=na.Cartesian2dVectorArray(2000, 0) * u.arcsec,
        )
