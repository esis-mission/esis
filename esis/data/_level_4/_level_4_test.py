import functools
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import esis

ctis = pytest.importorskip("ctis")

if not hasattr(ctis.instruments, "OptikaInstrument"):  # pragma: nocover
    pytest.skip(
        reason="ctis is missing OptikaInstrument (needs ctis PR #18)",
        allow_module_level=True,
    )


@functools.cache
def _level_4() -> esis.data.Level_4:
    """Invert three frames of the flight data on a deliberately coarse grid."""
    spectrum = esis.flights.f1.spectrum
    wavelength_center = na.ScalarArray(
        ndarray=u.Quantity(
            [
                spectrum.He_I.wavelength,
                spectrum.Mg_X.wavelength,
                spectrum.O_V.wavelength,
            ]
        ),
        axes="line",
    )
    width_doppler = na.ScalarArray(
        ndarray=u.Quantity(
            [
                spectrum.He_I.width_doppler,
                spectrum.Mg_X.width_doppler,
                spectrum.O_V.width_doppler,
            ]
        ),
        axes="line",
    )
    return esis.data.Level_4.from_level_1(
        a=esis.flights.f1.data.level_1()[dict(time=slice(14, 17))],
        wavelength_center=wavelength_center,
        width_doppler=width_doppler,
        instrument=esis.flights.f1.optics.distortion_fit(num_distribution=0),
        pitch_scene=16 * u.arcsec,
        num_velocity=2,
        index_time_reference=1,
        num_iteration=2,
    )


class TestLevel_4:
    def test_type(self):
        assert isinstance(_level_4(), esis.data.Level_4)

    def test_outputs(self):
        a = _level_4()
        shape = a.outputs.shape
        assert shape[a.axis_time] == 3
        assert shape[a.axis_wavelength] == a.num_line * (a.num_velocity + 1) - 1
        assert shape[a.axis_x] > 0
        assert shape[a.axis_y] > 0
        assert np.all(a.outputs >= 0)

    def test_mean_chi_squared(self):
        a = _level_4()
        assert a.axis_time in a.mean_chi_squared.shape
        assert np.all(np.isfinite(a.mean_chi_squared))

    def test_num_iteration(self):
        a = _level_4()
        assert a.num_iteration.shape == {a.axis_time: 3}
        assert np.all(a.num_iteration <= 2)

    def test_factor_norm(self):
        a = _level_4()
        assert np.all(a.factor_norm > 0)

    def test_window(self):
        a = _level_4()
        num = a.num_velocity + 1
        for i in range(a.num_line):
            window = a.window(i)[a.axis_wavelength]
            assert window.stop - window.start == a.num_velocity
            assert window.start == i * num

    def test_velocity(self):
        a = _level_4()
        velocity = a.velocity
        assert velocity.shape == {a.axis_wavelength: a.num_velocity + 1}
        velocity_first = velocity[{a.axis_wavelength: 0}].ndarray
        velocity_last = velocity[{a.axis_wavelength: ~0}].ndarray
        assert np.allclose(velocity_first, -200 * u.km / u.s)
        assert np.allclose(velocity_last, +200 * u.km / u.s)

    def test_intensity(self):
        a = _level_4()
        intensity = a.intensity
        assert a.axis_line in intensity.shape
        assert intensity.shape[a.axis_line] == a.num_line
        assert np.all(intensity >= 0)

    def test_velocity_mean(self):
        a = _level_4()
        velocity = a.velocity_mean
        assert a.axis_line in velocity.shape
        assert np.all(np.abs(velocity) <= 200 * u.km / u.s)

    def test_velocity_width(self):
        a = _level_4()
        width = a.velocity_width
        assert a.axis_line in width.shape
        assert np.all(width >= 0)
