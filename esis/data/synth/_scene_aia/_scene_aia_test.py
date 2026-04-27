import pytest
import esis
import astropy.units as u
import named_arrays as na
import numpy as np


@pytest.mark.parametrize("num_velocity", [3])
def test_scene_aia(num_velocity: int):
    l1 = esis.flights.f1.data.level_1()
    vernazza = 200 * u.erg / u.cm**2 / u.sr / u.s
    radiance = na.ScalarArray([1, 0.25] * vernazza, axes="spectral_line")
    wavelength = na.ScalarArray([580, 630] * u.AA, axes="spectral_line")
    scene_aia = esis.data.synth.scene_aia(
        time_start=l1.inputs[dict(time=0, channel=0)].time_start.ndarray,
        time_stop=l1.inputs[dict(time=2, channel=0)].time_start.ndarray,
        wavelength_aia=na.ScalarArray([304, 193] * u.AA, axes="spectral_line"),
        wavelength_new=wavelength,
        radiance=radiance,
        width_doppler=na.ScalarArray([10, 100] * u.km / u.s, axes="spectral_line"),
        num_velocity=num_velocity,
        num_std=5,
    )

    assert scene_aia.shape["velocity"] == num_velocity
    assert scene_aia.outputs.unit == vernazza.unit / wavelength.unit

    axis_xy = ("detector_x", "detector_y")
    delta_lambda = np.diff(scene_aia.inputs.wavelength, axis="velocity")
    integrated_radiance = scene_aia.outputs.mean(axis_xy) * delta_lambda
    assert np.allclose(integrated_radiance.sum("velocity"), radiance)
