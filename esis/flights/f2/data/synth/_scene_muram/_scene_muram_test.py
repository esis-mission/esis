import pathlib
from typing import Any
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import esis
from esis.flights.f2.data.synth._scene_muram import _scene_muram


def test_plate_scale():
    """
    Check the plate scale is about half an arcsecond, and positive.

    It is the angle a 15-micron pixel spans at the effective focal length of
    ESIS-II, and positive though the focal length of the primary mirror is
    negative.
    """
    result = esis.flights.f2.data.synth._scene_muram.plate_scale()
    assert isinstance(result, u.Quantity)
    assert 0.45 * u.arcsec < result < 0.5 * u.arcsec


@pytest.mark.parametrize(
    argnames="velocity",
    argvalues=[
        None,
        na.linspace(-20, 20, axis="v", num=5) * u.km / u.s,
    ],
)
def test_scene_muram(
    monkeypatch: pytest.MonkeyPatch,
    velocity: None | na.AbstractScalar,
):
    """
    Check the DEM is passed on to :func:`esis.data.synth.scene_vdem`.

    It is passed on with the passband of ESIS-II, stretched to one pixel of
    the detector.

    The download and the synthesis are replaced, so that the test needs
    neither the DEM nor the CHIANTI database.
    """
    calls = []
    path = pathlib.Path("muram.zarr")

    def scene_vdem(**kwargs: Any) -> na.FunctionArray:
        """Stand in for the synthesis, recording its arguments."""
        calls.append(kwargs)
        return na.FunctionArray(None, None)

    monkeypatch.setattr(_scene_muram, "path_muram", lambda: path)
    monkeypatch.setattr(esis.data.synth, "scene_vdem", scene_vdem)

    esis.flights.f2.data.synth.scene_muram(
        velocity=velocity,
        fraction_min=0.01,
        ions=["Ne 7"],
        axis_x="sx",
        axis_y="sy",
        axis_line="l",
        axis_velocity="v",
    )

    (call,) = calls
    instrument = esis.flights.f2.optics.design_single(num_distribution=0)
    assert call["path"] == path
    assert call["wavelength_min"] == instrument.wavelength_min
    assert call["wavelength_max"] == instrument.wavelength_max
    assert call["pressure"] == _scene_muram.pressure
    assert call["plate_scale"] == _scene_muram.plate_scale()
    assert call["fraction_min"] == 0.01
    assert call["ions"] == ["Ne 7"]
    assert call["axis_x"] == "sx"
    assert call["axis_y"] == "sy"
    assert call["axis_line"] == "l"
    assert call["axis_velocity"] == "v"

    v = call["velocity"]
    if velocity is None:
        assert na.shape(v) == dict(v=71)
        assert np.isclose(v.ndarray.min(), -400 * u.km / u.s)
        assert np.isclose(v.ndarray.max(), 300 * u.km / u.s)
    else:
        assert v is velocity
