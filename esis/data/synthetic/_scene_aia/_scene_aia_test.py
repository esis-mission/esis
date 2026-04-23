import pytest
import esis
import astropy.units as u
import named_arrays as na


@pytest.mark.parametrize("num_velocity", [3])
def test_scene_aia(num_velocity: int):
    l1 = esis.flights.f1.data.level_1()
    vernazza = 200 * u.erg / u.cm ** 2 / u.sr / u.s / (0.2 * u.AA)
    wavelength = na.ScalarArray([580, 630] * u.AA, axes='spectral_line')
    scene_aia = esis.data.scene_aia(
        time_start = l1.inputs[dict(time=0,channel=0)].time_start.ndarray,
        time_stop = l1.inputs[dict(time=2,channel=0)].time_start.ndarray,
        wavelength_aia = na.ScalarArray([304, 193] * u.AA, axes='spectral_line'),
        wavelength_new =wavelength,
        radiance = na.ScalarArray([1, .25] * vernazza, axes='spectral_line'),
        width_doppler = na.ScalarArray([10 , 100] * u.km/u.s, axes='spectral_line'),
        user_email = 'jacobdparker@gmail.com',
        num_velocity=num_velocity,
    )

    assert scene_aia.shape['velocity'] == num_velocity
    assert scene_aia.outputs.unit == vernazza.unit / wavelength.unit



