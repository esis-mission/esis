import pytest
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import optika._tests.test_mixins
import esis


class AbstractTestAbstractInstrument(
    optika._tests.test_mixins.AbstractTestPrintable,
):
    def test_name(self, a: esis.optics.abc.AbstractInstrument):
        result = a.name
        assert isinstance(result, str)

    def test_front_aperture(self, a: esis.optics.abc.AbstractInstrument):
        result = a.front_aperture
        if result is not None:
            assert isinstance(result, esis.optics.FrontAperture)

    def test_central_obscuration(self, a: esis.optics.abc.AbstractInstrument):
        result = a.central_obscuration
        if result is not None:
            assert isinstance(result, esis.optics.CentralObscuration)

    def test_primary_mirror(self, a: esis.optics.abc.AbstractInstrument):
        result = a.primary_mirror
        if result is not None:
            assert isinstance(result, esis.optics.PrimaryMirror)

    def test_field_stop(self, a: esis.optics.abc.AbstractInstrument):
        result = a.field_stop
        if result is not None:
            assert isinstance(result, esis.optics.FieldStop)

    def test_grating(self, a: esis.optics.abc.AbstractInstrument):
        result = a.grating
        if result is not None:
            assert isinstance(result, esis.optics.Grating)

    def test_filter(self, a: esis.optics.abc.AbstractInstrument):
        result = a.filter
        if result is not None:
            assert isinstance(result, esis.optics.Filter)

    def test_camera(self, a: esis.optics.abc.AbstractInstrument):
        result = a.camera
        if result is not None:
            assert isinstance(result, esis.optics.Camera)

    def test_wavelength(self, a: esis.optics.abc.AbstractInstrument):
        result = a.wavelength
        if result is not None:
            unit = na.unit_normalized(result)
            if unit.is_equivalent(u.dimensionless_unscaled):
                assert np.all(result <= 1)
                assert np.all(result >= -1)
            else:
                assert np.all(result > 0 * u.nm)

    def test_field(self, a: esis.optics.abc.AbstractInstrument):
        result = a.field
        if result is not None:
            assert isinstance(result, na.AbstractCartesian2dVectorArray)

    def test_pupil(self, a: esis.optics.abc.AbstractInstrument):
        result = a.pupil
        if result is not None:
            assert isinstance(result, na.AbstractCartesian2dVectorArray)

    def test_angle_grating_input(self, a: esis.optics.abc.AbstractInstrument):
        result = a.angle_grating_input
        assert isinstance(na.as_named_array(result), na.AbstractArray)
        assert na.unit_normalized(result).is_equivalent(u.deg)

    def test_angle_grating_output(self, a: esis.optics.abc.AbstractInstrument):
        result = a.angle_grating_output
        assert isinstance(na.as_named_array(result), na.AbstractArray)
        assert na.unit_normalized(result).is_equivalent(u.deg)

    def test_wavelength_min(self, a: esis.optics.abc.AbstractInstrument):
        result = a.wavelength_min
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.AA)

    def test_wavelength_max(self, a: esis.optics.abc.AbstractInstrument):
        result = a.wavelength_max
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.AA)

    def test_wavlength_physical(self, a: esis.optics.abc.AbstractInstrument):
        assert np.all(a.wavelength_physical > 0 * u.nm)

    def test_system(self, a: esis.optics.abc.AbstractInstrument):
        result = a.system
        assert isinstance(result, optika.systems.AbstractSequentialSystem)
        assert result.surfaces

    def test_schematic_primary(self, a: esis.optics.abc.AbstractInstrument):

        fig, ax = plt.subplots()

        a.schematic_primary()

        assert ax.has_data()

        plt.close(fig)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.flights.f1.optics.design_single(num_distribution=2),
        esis.flights.f1.optics.design(num_distribution=0),
    ],
)
class TestInstrument(
    AbstractTestAbstractInstrument,
):
    pass
