import pytest
import astropy.units as u
import named_arrays as na
import numpy as np
import esis
from esis.flights.f1.spectrum import O_V, Mg_X, He_I


@pytest.mark.parametrize("num_velocity", [3])
def test_scene_aia(
    num_velocity: int,
):

    axis_x = "detector_x"
    axis_y = "detector_y"
    axis_xy = (axis_x, axis_y)
    axis_wavelength = "wavelength"
    axis_velocity = "velocity"

    limit = 1

    result = esis.flights.f1.data.synth.scene_aia(
        axis_detector_x=axis_x,
        axis_detector_y=axis_y,
        axis_wavelength=axis_wavelength,
        axis_velocity=axis_velocity,
        num_velocity=num_velocity,
        limit=limit,
    )

    assert result.shape[axis_velocity] == num_velocity
    assert result.outputs.unit.is_equivalent(u.erg / u.cm**2 / u.sr / u.AA / u.s)

    delta_lambda = np.diff(result.inputs.wavelength, axis=axis_velocity)
    radiance = (result.outputs * delta_lambda).sum(axis_velocity).mean(axis_xy)
    assert np.allclose(radiance[{axis_wavelength: 0}], O_V.radiance, rtol=1e-1)
    assert np.allclose(radiance[{axis_wavelength: 1}], Mg_X.radiance, rtol=1e-1)
    assert np.allclose(radiance[{axis_wavelength: 2}], He_I.radiance, rtol=1e-1)
