import pytest
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants
import named_arrays as na
import optika._tests.test_mixins
import esis


class AbstractTestAbstractInstrument(
    optika._tests.test_mixins.AbstractTestPrintable,
    optika._tests.test_mixins.AbstractTestReplaceable,
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

    def test_distance_grating_input(
        self,
        a: esis.optics.abc.AbstractInstrument,
    ) -> None:
        result = a.distance_grating_input
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert np.all(result > 0 * u.mm)

    def test_distance_grating_output(
        self,
        a: esis.optics.abc.AbstractInstrument,
    ) -> None:
        result = a.distance_grating_output
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert np.all(result > 0 * u.mm)

    def test_magnification(
        self,
        a: esis.optics.abc.AbstractInstrument,
    ) -> None:
        result = a.magnification
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.dimensionless_unscaled)
        assert np.all(result > 0)

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

    def test_dispersion(self, a: esis.optics.abc.AbstractInstrument):
        wavelength = esis.flights.f1.spectrum.O_V.wavelength
        result = a.dispersion(wavelength)

        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mAA / u.pix)
        assert np.all(result > 0 * u.mAA / u.pix)

    def test_dispersion_doppler(self, a: esis.optics.abc.AbstractInstrument):
        wavelength = esis.flights.f1.spectrum.O_V.wavelength
        result = a.dispersion_doppler(wavelength)

        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.km / u.s / u.pix)

        # the same quantity as the dispersion, expressed as a velocity
        expected = a.dispersion(wavelength) / wavelength * astropy.constants.c
        assert np.allclose(result, expected.to(u.km / u.s / u.pix))

    def test_dispersion_varies_less_than_its_doppler_equivalent(
        self,
        a: esis.optics.abc.AbstractInstrument,
    ):
        """
        Check that the dispersion is the steadier of the two.

        The dispersion is nearly constant across the passband, while the
        velocity it corresponds to is not, since that is divided by the
        wavelength. This is why both are reported at a wavelength.
        """
        spectrum = esis.flights.f1.spectrum
        short = spectrum.He_I.wavelength
        long = spectrum.O_V.wavelength

        def spread(f):
            lower, upper = f(short), f(long)
            return np.abs((upper - lower) / lower)

        assert np.all(spread(a.dispersion) < spread(a.dispersion_doppler))

    def test_system(self, a: esis.optics.abc.AbstractInstrument):
        result = a.system
        assert isinstance(result, optika.systems.AbstractSequentialSystem)
        assert result.surfaces

    def test_focus_grating(self, a: esis.optics.abc.AbstractInstrument):
        bounds = (-3 * u.mm, 3 * u.mm)
        result = a.focus_grating(bounds=bounds)
        assert isinstance(result, esis.optics.abc.AbstractInstrument)
        dz = result.grating.translation.z - a.grating.translation.z
        assert np.all(dz >= bounds[0])
        assert np.all(dz <= bounds[1])

    def test_focus_grating_edge(self, a: esis.optics.abc.AbstractInstrument):
        """A focus beyond the bounds is not returned as if it were the best."""
        with pytest.raises(ValueError, match="edge"):
            a.focus_grating(bounds=(0.1 * u.mm, 0.5 * u.mm))

    def test_position_line(self, a: esis.optics.abc.AbstractInstrument):
        result = a.position_line()
        assert isinstance(result, na.AbstractCartesian2dVectorArray)
        assert na.unit(result).is_equivalent(u.mm)

    def test_align_grating(self, a: esis.optics.abc.AbstractInstrument):
        position = a.position_line()

        result = a.align_grating(position=position)
        assert isinstance(result, esis.optics.abc.AbstractInstrument)

        # the line is put where it was asked for, to a hundredth of a pixel
        error = result.position_line() - position
        assert np.all(np.abs(error.x) < a.camera.sensor.width_pixel / 100)
        assert np.all(np.abs(error.y) < a.camera.sensor.width_pixel / 100)

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


def test_focus_grating_edge_sample() -> None:
    """
    A sample whose focus is beyond the bounds is caught.

    The nominal focus of this model is inside the bounds, and the truth of an
    uncertain array is that of its nominal value, so only a check of every
    sample sees the one that is not.
    """
    instrument = esis.flights.f1.optics.design_single(num_distribution=0)
    sag = instrument.grating.sag
    radius = na.UncertainScalarArray(
        nominal=sag.radius,
        distribution=na.ScalarArray(
            ndarray=sag.radius * np.array([1, 1.004]),
            axes="_distribution",
        ),
    )
    instrument = instrument.replace(
        grating=instrument.grating.replace(sag=sag.replace(radius=radius)),
    )
    with pytest.raises(ValueError, match="edge"):
        instrument.focus_grating(bounds=(-0.5 * u.mm, 0.5 * u.mm))


@pytest.mark.parametrize("axis", ["x", "y"])
def test_align_grating_decenter(axis: str) -> None:
    """
    A grating a millimeter out of place is pointed back and refocused.

    Decentered along :math:`x`, it needs several arcminutes of yaw, which
    defocus it unless it is focused again afterwards; decentered along
    :math:`y`, it needs pitch.
    """
    wavelength = esis.flights.f1.spectrum.O_V.wavelength
    field = na.Cartesian2dVectorLinearSpace(
        start=-1,
        stop=1,
        axis=na.Cartesian2dVectorArray("field_x", "field_y"),
        num=5,
        centers=True,
    )
    pupil = na.Cartesian2dVectorLinearSpace(
        start=-1,
        stop=1,
        axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
        num=11,
        centers=True,
    )

    def radius_spot(instrument: esis.optics.abc.AbstractInstrument) -> u.Quantity:
        """Measure the RMS radius of the spots, averaged over the field of view."""
        rays = instrument.system.rayfunction(
            wavelength=wavelength,
            field=field,
            pupil=pupil,
            efficiency=False,
        )
        position = rays.outputs.position
        unvignetted = na.as_named_array(rays.outputs.unvignetted)
        x = np.where(unvignetted, position.x, np.nan)
        y = np.where(unvignetted, position.y, np.nan)
        axis = ("pupil_x", "pupil_y")
        variance = np.nanvar(x, axis=axis) + np.nanvar(y, axis=axis)
        return np.nanmean(np.sqrt(variance)).ndarray

    design = esis.flights.f1.optics.design_single(num_distribution=0)
    translation = design.grating.translation
    decentered = design.replace(
        grating=design.grating.replace(
            translation=translation.replace(
                **{axis: getattr(translation, axis) + 1 * u.mm},
            ),
        ),
    )

    result = decentered.align_grating(wavelength=wavelength)

    width_pixel = result.camera.sensor.width_pixel
    error = result.position_line(wavelength) - result.camera.sensor.position_image
    assert np.abs(error.x) < width_pixel / 100
    assert np.abs(error.y) < width_pixel / 100

    # the spots grow by less than 15 percent, where a single pass of the
    # focus and the yaw alone left them 30 percent to 2.6 times larger
    aligned = design.align_grating(wavelength=wavelength)
    assert radius_spot(result) < 1.15 * radius_spot(aligned)


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
