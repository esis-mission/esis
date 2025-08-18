import pytest
import numpy as np
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


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.Instrument(
            name="esis-test",
            front_aperture=esis.optics.FrontAperture(),
            central_obscuration=esis.optics.CentralObscuration(
                num_folds=8,
            ),
            primary_mirror=esis.optics.PrimaryMirror(
                sag=optika.sags.ParabolicSag(-1000 * u.mm),
                num_folds=8,
                width_clear=100 * u.mm,
                width_border=1 * u.mm,
                material=optika.materials.Mirror(),
                translation=na.Cartesian3dVectorArray(z=2000) * u.mm,
            ),
            field_stop=esis.optics.FieldStop(
                num_folds=8,
                radius_clear=2 * u.mm,
                radius_mechanical=20 * u.mm,
                translation=na.Cartesian3dVectorArray(z=1000) * u.mm,
            ),
            grating=esis.optics.Grating(
                serial_number="abc123",
                manufacturing_number="123abc",
                sag=optika.sags.SphericalSag(radius=500 * u.mm),
                material=optika.materials.Mirror(),
                rulings=optika.rulings.SawtoothRulings(
                    spacing=1 * u.um,
                    depth=10 * u.nm,
                    diffraction_order=1,
                ),
                num_folds=8,
                halfwidth_inner=15 * u.mm,
                halfwidth_outer=10 * u.mm,
                width_border=1 * u.mm,
                width_border_inner=1.5 * u.mm,
                clearance=1 * u.mm,
                distance_radial=50 * u.mm,
                translation=na.Cartesian3dVectorArray(z=750) * u.mm,
                yaw=-5 * u.deg,
            ),
            filter=esis.optics.Filter(
                radius_clear=20 * u.mm,
                width_border=1 * u.mm,
                distance_radial=75 * u.mm,
                translation=na.Cartesian3dVectorArray(z=1750) * u.mm,
            ),
            camera=esis.optics.Camera(
                sensor=esis.optics.Sensor(
                    distance_radial=85 * u.mm,
                    translation=na.Cartesian3dVectorArray(z=2000) * u.mm,
                ),
            ),
            wavelength=na.linspace(-1, 1, num=3, axis="wavelength"),
            field=na.Cartesian2dVectorArray(
                x=na.linspace(0, 1, num=5, axis="field_x"),
                y=na.linspace(0, 1, num=5, axis="field_y"),
            ),
            pupil=na.Cartesian2dVectorArray(
                x=na.linspace(0, 1, num=5, axis="pupil_x"),
                y=na.linspace(0, 1, num=5, axis="pupil_y"),
            ),
            requirements=esis.optics.Requirements(
                resolution_spatial=1.5 * u.Mm,
                resolution_spectral=18 * u.km / u.s,
                fov=10 * u.arcmin,
                snr=17.3 * u.dimensionless_unscaled,
                cadence=15 * u.s,
                length_observation=150 * u.s,
            ),
        )
    ],
)
class TestInstrument(
    AbstractTestAbstractInstrument,
):
    pass
