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


@pytest.mark.parametrize("channel", [0, 1, 2, 3])
def test_isel(channel: int):
    """Selecting a channel via the `na.Indexable` interface yields a valid system.

    ``AbstractInstrument`` inherits :class:`named_arrays.Indexable`, so a single
    channel can be selected with ``instrument.isel(channel=...)`` (equivalently
    ``instrument[{instrument.axis_channel: ...}]``) without a bespoke method.
    """
    instrument = esis.flights.f1.optics.design(num_distribution=0)
    axis = instrument.axis_channel

    # make a grating orientation and a (dict-stored) ruling coefficient vary by
    # channel, mimicking a distortion-fit model
    yaw = na.ScalarArray(np.array([1.0, 2.0, 3.0, 4.0]) * u.deg, axes=axis)
    coefficient = na.ScalarArray(np.array([10.0, 20.0, 30.0, 40.0]) * u.um, axes=axis)
    instrument.grating.yaw = yaw
    instrument.grating.rulings.spacing.coefficients[0] = coefficient

    result = instrument.isel(**{axis: channel})

    # isel and the equivalent dict-indexing agree
    assert result.grating.yaw.ndarray == instrument[{axis: channel}].grating.yaw.ndarray

    # the selected element is kept and the channel axis is removed
    assert result.grating.yaw.ndarray == yaw.ndarray[channel]
    assert axis not in result.grating.yaw.axes
    assert (
        result.grating.rulings.spacing.coefficients[0].ndarray
        == coefficient.ndarray[channel]
    )
    assert axis not in result.grating.rulings.spacing.coefficients[0].axes
    assert axis not in na.as_named_array(result.grating.azimuth).axes
    assert axis not in na.as_named_array(result.camera.channel).axes

    # the original instrument is left unmodified (na.getitem rebuilds via replace)
    assert axis in instrument.grating.yaw.axes
    assert np.all(instrument.grating.yaw.ndarray == yaw.ndarray)

    # the single-channel instrument still resolves into an optical system
    assert isinstance(result.system, optika.systems.AbstractSequentialSystem)
    assert result.system.surfaces
