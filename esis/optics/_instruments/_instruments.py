from __future__ import annotations
from typing import Any, Self
import abc
import copy
import dataclasses
import functools
import numpy as np
import scipy.spatial
import matplotlib.axes
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants
import named_arrays as na
import optika
import esis

__all__ = [
    "AbstractInstrument",
    "Instrument",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractInstrument(
    na.Indexable,
    optika.mixins.Printable,
    optika.mixins.Rollable,
    optika.mixins.Yawable,
    optika.mixins.Pitchable,
):
    """An interface describing the entire optical system."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The human-readable name of the instrument."""

    @property
    @abc.abstractmethod
    def axis_channel(self):
        """The name of the logical axis corresponding to changing camera channel."""

    @property
    @abc.abstractmethod
    def front_aperture(self) -> None | esis.optics.abc.AbstractFrontAperture:
        """A model of the front aperture plate."""

    @property
    @abc.abstractmethod
    def central_obscuration(self) -> None | esis.optics.abc.AbstractCentralObscuration:
        """A model of the central obscuration."""

    @property
    @abc.abstractmethod
    def primary_mirror(self) -> None | esis.optics.abc.AbstractPrimaryMirror:
        """A model of the primary mirror."""

    @property
    @abc.abstractmethod
    def field_stop(self) -> None | esis.optics.abc.AbstractFieldStop:
        """A model of the field stop."""

    @property
    @abc.abstractmethod
    def grating(self) -> None | esis.optics.abc.AbstractGrating:
        """A model of the diffraction grating array."""

    @property
    @abc.abstractmethod
    def filter(self) -> None | esis.optics.abc.AbstractFilter:
        """A model of the thin-film filters."""

    @property
    @abc.abstractmethod
    def camera(self) -> None | esis.optics.Camera:
        """A model of the camera and sensors."""

    @property
    @abc.abstractmethod
    def wavelength(self) -> None | u.Quantity | na.AbstractScalar:
        """
        A default grid of wavelengths to trace through the system.

        Can be either in normalized coordinates (in the range :math:`-1` to :math:`+1`)
        or in physical coordinates (with units of length).

        See Also
        --------
        :attr:`wavelength_physical`: This value converted into in physical coordinates.
        """

    @property
    @abc.abstractmethod
    def field(self) -> None | na.AbstractCartesian2dVectorArray:
        """A default grid of field positions to trace through the system."""

    @property
    @abc.abstractmethod
    def pupil(self):
        """A default grid of pupil positions to trace through the system."""

    @property
    @abc.abstractmethod
    def kwargs_plot(self):
        """Extra keyword arguments used to plot the optical system."""

    @property
    def angle_grating_input(self) -> na.AbstractScalar:
        """
        The angle between the grating normal and the direction of the incident light.

        This is the incidence angle :math:`theta_i` in the
        `diffraction grating equation <https://en.wikipedia.org/wiki/Diffraction_grating>`_.
        """
        fs = self.field_stop.surface
        grating = self.grating.surface
        position = na.Cartesian3dVectorArray() * u.mm
        normal_surface = grating.sag.normal(position)
        normal_rulings = grating.rulings.spacing_(position, normal_surface).normalized
        transformation = grating.transformation.inverse @ fs.transformation
        wire = np.moveaxis(
            a=fs.aperture.wire(),
            source="wire",
            destination="wire_grating_input",
        )
        wire = transformation(wire)
        return np.arctan2(
            wire @ normal_rulings,
            wire @ normal_surface,
        )

    @property
    def angle_grating_output(self) -> na.AbstractScalar:
        """
        The angle between the grating normal and the direction of the diffracted light.

        This is an analogue to the diffracted angle in the
        `diffraction grating equation <https://en.wikipedia.org/wiki/Diffraction_grating>`_.
        """
        detector = self.camera.surface
        grating = self.grating.surface
        position = na.Cartesian3dVectorArray() * u.mm
        normal_surface = grating.sag.normal(position)
        normal_rulings = grating.rulings.spacing_(position, normal_surface).normalized
        transformation = grating.transformation.inverse @ detector.transformation
        wire = np.moveaxis(
            a=detector.aperture.wire(),
            source="wire",
            destination="wire_grating_output",
        )
        wire = transformation(wire)
        return np.arctan2(
            wire @ normal_rulings,
            wire @ normal_surface,
        )

    @property
    def _wavelength_test_grid(self) -> na.AbstractScalar:
        position = na.Cartesian3dVectorArray() * u.mm
        grating = self.grating.surface
        normal = grating.sag.normal(position)
        m = grating.rulings.diffraction_order
        d = grating.rulings.spacing_(position, normal).length
        a = self.angle_grating_input
        b = self.angle_grating_output
        result = np.abs((np.sin(a) + np.sin(b)) * d / m)
        return result.to(u.AA)

    @property
    def wavelength_min(self) -> u.Quantity | na.AbstractScalar:
        """The minimum wavelength permitted through the system."""
        return self._wavelength_test_grid.min(
            axis=("wire_grating_input", "wire_grating_output"),
        )

    @property
    def wavelength_max(self) -> u.Quantity | na.AbstractScalar:
        """The maximum wavelength permitted through the system."""
        return self._wavelength_test_grid.max(
            axis=("wire_grating_input", "wire_grating_output"),
        )

    def dispersion(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        delta: u.Quantity = 0.5 * u.AA,
    ) -> u.Quantity | na.AbstractScalar:
        r"""
        Compute the wavelength interval imaged onto one pixel.

        Measured from the raytrace: two wavelengths either side of the one
        asked for are traced along the optical axis, and the distance between
        where they land is divided into the width of a pixel.

        The dispersion of this instrument varies across its passband by about
        a percent, so it is reported at a wavelength rather than as a single
        number for the instrument.

        Parameters
        ----------
        wavelength
            The wavelength to measure the dispersion at.
        delta
            The half-interval either side of ``wavelength`` to measure across.

        Examples
        --------
        The dispersion at the O V line.

        .. jupyter-execute::

            import esis

            instrument = esis.flights.f1.optics.design_single(num_distribution=0)

            instrument.dispersion(esis.flights.f1.spectrum.O_V.wavelength)
        """
        wavelength_test = wavelength + delta * na.ScalarArray(
            ndarray=np.array([-1, 1]),
            axes=("_wavelength_dispersion",),
        )

        axis = na.Cartesian2dVectorArray(0, 0)
        rayfunction = self.system.rayfunction(
            wavelength=wavelength_test,
            field=axis,
            pupil=axis,
        )

        # the gratings disperse along the x axis of the sensor, so the
        # y coordinate carries none of the separation
        position = rayfunction.outputs.position.x
        index = {"_wavelength_dispersion": 0}
        separation = position[{"_wavelength_dispersion": 1}] - position[index]

        width_pixel = self.camera.sensor.width_pixel / u.pix

        return (width_pixel * 2 * delta / separation).to(u.mAA / u.pix)

    def dispersion_doppler(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        delta: u.Quantity = 0.5 * u.AA,
    ) -> u.Quantity | na.AbstractScalar:
        r"""
        Compute the Doppler velocity interval imaged onto one pixel.

        This is :meth:`dispersion` expressed as a velocity, and it therefore
        varies across the passband much more than the dispersion itself does,
        since it is divided by the wavelength. Over the ESIS passband the
        dispersion changes by about a percent while its Doppler equivalent
        changes by about ten times that, so the line it is quoted at matters.

        Parameters
        ----------
        wavelength
            The wavelength to measure the dispersion at.
        delta
            The half-interval either side of ``wavelength`` to measure across.

        Examples
        --------
        The Doppler dispersion at the O V line.

        .. jupyter-execute::

            import esis

            instrument = esis.flights.f1.optics.design_single(num_distribution=0)

            instrument.dispersion_doppler(esis.flights.f1.spectrum.O_V.wavelength)
        """
        dispersion = self.dispersion(wavelength, delta=delta)
        result = dispersion / wavelength * astropy.constants.c
        return result.to(u.km / u.s / u.pix)

    @property
    def wavelength_physical(self) -> na.ScalarArray:
        """The value of :attr:`wavelength` converted to physical units if needed."""
        wavelength = self.wavelength
        if na.unit_normalized(wavelength).is_equivalent(u.dimensionless_unscaled):
            wavelength_min = self.wavelength_min
            wavelength_max = self.wavelength_max
            wavelength_range = wavelength_max - wavelength_min
            wavelength = wavelength_range * (wavelength + 1) / 2 + wavelength_min
        return wavelength

    @functools.cached_property
    def system(self) -> optika.systems.SequentialSystem:
        """
        Convert this model into an instance of :class:`optika.systems.SequentialSystem`.

        This is a cached property that is only computed once.
        """
        surfaces = []
        surfaces += [self.front_aperture.surface]
        surfaces += [self.central_obscuration.surface]
        surfaces += [self.primary_mirror.surface]
        surfaces += [self.field_stop.surface]
        surfaces += [self.grating.surface]
        surfaces += [self.filter.surface]

        result = optika.systems.SequentialSystem(
            surfaces=surfaces,
            sensor=self.camera.surface,
            grid_input=optika.vectors.ObjectVectorArray(
                wavelength=self.wavelength_physical,
                field=self.field,
                pupil=self.pupil,
            ),
            transformation=self.transformation,
            kwargs_plot=self.kwargs_plot,
        )

        return result

    def focus_grating(
        self,
        wavelength: None | u.Quantity | na.AbstractScalar = None,
        field: None | na.AbstractCartesian2dVectorArray = None,
        pupil: None | na.AbstractCartesian2dVectorArray = None,
        bounds: tuple[u.Quantity, u.Quantity] = (-1 * u.mm, 2 * u.mm),
        min_step_size: u.Quantity = 1 * u.um,
    ) -> Self:
        r"""
        Move the gratings along the optic axis to their best focus.

        The gratings image the field stop onto the sensor, so a grating whose
        radius of curvature differs from the one it was placed for (a
        measured, as-built radius replacing the design radius, for example)
        images the field stop to a different distance and defocuses the
        system unless it is moved to compensate.

        This method finds, independently for every element of the grating
        (every channel, for example), the translation along :math:`z` which
        minimizes the root-mean-square radius of the spots imaged onto the
        sensor, and returns a copy of this instrument with the gratings moved
        there. Everything else, including the sensor, stays where it was.
        The search is a vectorized Brent minimization
        (:func:`named_arrays.optimize.minimum_brent`), so every channel is
        focused by the same handful of ray traces.

        Parameters
        ----------
        wavelength
            The wavelengths at which to focus.
            If :obj:`None` (the default), :attr:`wavelength_physical` is used.
        field
            The normalized field positions of the traced rays.
            If :obj:`None` (the default), a :math:`3 \times 3` grid spanning
            the central half of the field of view is used.
        pupil
            The normalized pupil positions of the traced rays.
            If :obj:`None` (the default), a :math:`21 \times 21` grid is used.
        bounds
            The bracket of grating translations, relative to the current
            position, within which the best focus is sought.
        min_step_size
            The tolerance on the translation of the best focus.

        Examples
        --------
        Move the gratings of the ESIS-I as-built model, which carries the
        measured radii of curvature but the design positions, to their best
        focus in the O V line.

        .. jupyter-execute::

            import astropy.units as u
            import named_arrays as na
            import esis

            instrument = esis.flights.f1.optics._as_built(num_distribution=0)

            focused = instrument.focus_grating(wavelength=629.73 * u.AA)

            na.nominal(focused.grating.translation.z - instrument.grating.translation.z)
        """
        if wavelength is None:
            wavelength = self.wavelength_physical

        if field is None:
            field = na.Cartesian2dVectorLinearSpace(
                start=-0.5,
                stop=0.5,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=3,
            )

        if pupil is None:
            pupil = na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
                num=21,
                centers=True,
            )

        axis_pupil = tuple(na.shape(pupil))
        axis_scene = tuple(na.shape(field)) + tuple(na.shape(wavelength))

        result = copy.deepcopy(self)
        result.__dict__.pop("system", None)

        z = result.grating.translation.z

        def radius_spot(dz: na.AbstractScalar) -> na.AbstractScalar:
            instrument = copy.deepcopy(result)
            instrument.grating.translation.z = z + dz
            # only the positions of the rays matter here, and the efficiency
            # of the coatings is most of what tracing them costs
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
            variance = np.nanvar(x, axis=axis_pupil) + np.nanvar(y, axis=axis_pupil)
            return np.nanmean(np.sqrt(variance), axis=axis_scene)

        a, b = bounds
        dz = na.optimize.minimum_brent(
            function=radius_spot,
            a=a,
            b=b,
            min_step_size=min_step_size,
        )

        result.grating.translation.z = z + dz

        return result

    def position_line(
        self,
        wavelength: None | u.Quantity | na.AbstractScalar = None,
    ) -> na.AbstractCartesian2dVectorArray:
        r"""
        Where the center of the field of view lands on the sensor.

        The ray from the middle of the field of view, through the middle of
        the pupil, at the wavelength asked for. This is the position a
        spectral line is at, in the sense that
        :attr:`esis.optics.Sensor.position_image` means it.

        It is taken from the one ray rather than from the centroid of the
        whole image because the centroid depends on how much of each beam
        survives to the sensor. With the primary aperture stop removed, three
        quarters of the rays are lost, and asymmetrically enough to drag the
        centroid of the O V image 0.42 mm from where its central ray lands.
        That is a real effect and it is what the flight data show, but it is
        not a property of the alignment, and it would move under any change
        to the model of the apertures.

        Parameters
        ----------
        wavelength
            The wavelength of the line.
            If :obj:`None` (the default), :attr:`wavelength_physical` is used.
        """
        if wavelength is None:
            wavelength = self.wavelength_physical

        zero = na.Cartesian2dVectorArray(0, 0)

        rays = self.system.rayfunction(
            wavelength=wavelength,
            field=zero,
            pupil=zero,
            efficiency=False,
        )

        return rays.outputs.position.xy

    def align_grating(
        self,
        wavelength: None | u.Quantity | na.AbstractScalar = None,
        position: None | na.AbstractCartesian2dVectorArray = None,
        field: None | na.AbstractCartesian2dVectorArray = None,
        pupil: None | na.AbstractCartesian2dVectorArray = None,
        **kwargs,
    ) -> Self:
        r"""
        Focus the gratings and then point them back at the sensor.

        :meth:`focus_grating` moves each grating along the optic axis, which
        also moves the image along the sensor. An instrument which was
        aligned as well as focused puts the image where it belongs, and this
        method does both, leaving the line within a hundredth of a pixel of
        the position asked for.

        Focusing alone comes closer than one might expect, because a grating
        whose radius is wrong both defocuses the image and displaces it, and
        moving the grating to correct the one largely corrects the other. For
        the as-built model it leaves a couple of pixels, which is still tens
        of kilometers per second of apparent Doppler shift.

        The two are solved one after the other rather than together. Rotating
        a grating about :math:`y` moves the image 915 microns per arcminute
        while changing the size of a spot by half a micron, so the rotation
        needed to place the line, of order ten arcseconds, does not disturb
        the focus. There is no trade to make between the two, and so no
        weighting between them to choose.

        Parameters
        ----------
        wavelength
            The wavelength of the line to place.
            If :obj:`None` (the default), :attr:`wavelength_physical` is used.
        position
            Where on the sensor to put it.
            If :obj:`None` (the default),
            :attr:`esis.optics.Sensor.position_image` is used.
        field
            The normalized field positions used to judge the focus.
            If :obj:`None` (the default), a :math:`3 \times 3` grid of cell
            centers spanning the field of view is used.
        pupil
            The normalized pupil positions used to judge the focus.
            Passed to :meth:`focus_grating`.
        kwargs
            Additional arguments passed to :meth:`focus_grating`.

        Examples
        --------
        Align the as-built model and confirm the O V line lands where the
        sensor says it should.

        .. jupyter-execute::

            import astropy.units as u
            import named_arrays as na
            import esis

            instrument = esis.flights.f1.optics._as_built(num_distribution=0)

            aligned = instrument.align_grating()

            error = aligned.position_line() - aligned.camera.sensor.position_image
            na.nominal(error.length.to(u.um))
        """
        if wavelength is None:
            wavelength = self.wavelength_physical

        if position is None:
            position = self.camera.sensor.position_image

        if field is None:
            # Sampled across the field of view rather than only its middle.
            # Which sampling is used barely matters, since defocus varies
            # little across this field: three by three and seven by seven
            # agree on the focus to within a micron. Cell centers keep the
            # samples off the corners of the field stop, where every ray is
            # vignetted and the size of a spot is not defined.
            field = na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=3,
                centers=True,
            )

        result = self.focus_grating(
            wavelength=wavelength,
            field=field,
            pupil=pupil,
            **kwargs,
        )

        result = copy.deepcopy(result)
        result.__dict__.pop("system", None)

        yaw = result.grating.yaw

        # The image moves along the sensor in proportion to the rotation, so
        # one step of a secant method lands on the target, and the second
        # evaluation measures how far off it was.
        step = 1 * u.arcmin

        position_0 = result.position_line(wavelength)

        result.grating.yaw = yaw + step
        result.__dict__.pop("system", None)
        position_1 = result.position_line(wavelength)

        slope = (position_1.x - position_0.x) / step

        result.grating.yaw = yaw + (position.x - position_0.x) / slope
        result.__dict__.pop("system", None)

        return result

    def schematic_primary(
        self,
        ax: None | matplotlib.axes.Axes = None,
        transformation: None | na.transformations.AbstractTransformation = None,
        footprint: bool = True,
        color: str = "black",
        kwargs_footprint: None | dict[str, Any] = None,
        **kwargs,
    ) -> None:
        """
        Plot a schematic of the primary mirror along with the beam footprint.

        Parameters
        ----------
        ax
            The :mod:`matplotlib` axes on which to plot this schematic.
            If :obj:`None` (the default), the schematic is plotted on the
            current axes.
        transformation
            An additional transformation to apply to the coordinates
            before plotting the schematic.
        footprint
            Whether to plot the footprint of the rays on the mirror surface.
        color
            The color of the primary mirror.
        kwargs_footprint
            Additional kwargs for plotting the footprint of the beam.
        kwargs
            Additional kwargs for plotting the primary mirror.

        """
        if ax is None:
            ax = plt.gca()

        if transformation is None:
            transformation = na.transformations.IdentityTransformation()

        if self.transformation is not None:
            transformation = transformation @ self.transformation

        if kwargs_footprint is None:
            kwargs_footprint = dict()

        kwargs_footprint = kwargs_footprint | dict(
            facecolor="none",
            edgecolor="tab:orange",
        )

        shape = self.system.shape

        primary = self.primary_mirror.surface

        components = ("x", "y")

        primary.aperture.plot(
            ax=ax,
            transformation=transformation,
            components=components,
            color="tab:blue",
            label="outer C.A.",
            **kwargs,
        )
        primary.aperture_mechanical.plot(
            ax=ax,
            transformation=transformation,
            components=components,
            color=color,
            **kwargs,
        )

        radius_inner = 20 * u.mm
        az = na.linspace(0, 360, axis="az", num=101) * u.deg
        na.plt.plot(
            radius_inner * np.cos(az),
            radius_inner * np.sin(az),
            color="tab:cyan",
            label="inner C.A.",
        )

        na.plt.dimension(
            na.Cartesian2dVectorArray(-radius_inner, 0 * u.mm),
            na.Cartesian2dVectorArray(+radius_inner, 0 * u.mm),
        )

        if footprint:

            index_primary = self.system.surfaces_all.index(primary)
            index_primary = {self.system.axis_surface: index_primary}

            rays = self.system.raytrace().outputs

            where = rays.unvignetted[{self.system.axis_surface: ~0}]

            rays = rays[index_primary]

            rays = transformation(rays)

            rays = primary.transformation.inverse(rays)

            for n, i in enumerate(na.ndindex(shape)):

                position_i = rays.position[i]
                where_i = where[i]

                position_x = na.nominal(position_i.x[where_i]).ndarray
                position_y = na.nominal(position_i.y[where_i]).ndarray

                position = np.stack(
                    arrays=[
                        position_x,
                        position_y,
                    ],
                    axis=~0,
                )

                hull = scipy.spatial.ConvexHull(position)

                px = position_x[hull.vertices]
                py = position_y[hull.vertices]

                ax.fill(px, py, **kwargs_footprint)

                sx = px.mean()
                sy = py.mean()

                rx = position_x.mean()
                ry = position_y.mean()

                ax.text(
                    x=sx,
                    y=sy,
                    s=f"Ch. {self.camera.channel[i].ndarray}",
                    ha="center",
                    va="center",
                    color=kwargs_footprint["edgecolor"],
                    # fontsize=8,
                )

                if n == 0:
                    label = "test point"
                else:
                    label = "_test point"
                ax.scatter(
                    rx,
                    ry,
                    marker="+",
                    color="black",
                    label=label,
                )

                width_clear = self.primary_mirror.width_clear
                width_border = self.primary_mirror.width_border

                halfwidth_mech = width_clear / 2 + width_border

                if n == 2:
                    na.plt.dimension(
                        na.Cartesian2dVectorArray(0 * u.mm, ry),
                        na.Cartesian2dVectorArray(0 * u.mm, -halfwidth_mech),
                        offset=-60 * u.mm,
                        rotate=False,
                        ax=ax,
                    )


@dataclasses.dataclass(eq=False, repr=False)
class Instrument(
    AbstractInstrument,
):
    """
    An object which represents the entire optical system.

    A composition of the optical elements and a grid of input rays.
    Designed to resolve the optical elements into an instance of
    :class:`optika.systems.SequentialSystem` for performance modeling.
    """

    name: str = "ESIS"
    """The human-readable name of the instrument."""

    axis_channel: str = "channel"
    """The name of the logical axis corresponding to changing camera channel."""

    front_aperture: None | esis.optics.FrontAperture = None
    """A model of the front aperture plate."""

    central_obscuration: None | esis.optics.CentralObscuration = None
    """A model of the central obscuration."""

    primary_mirror: None | esis.optics.PrimaryMirror = None
    """A model of the primary mirror."""

    field_stop: None | esis.optics.FieldStop = None
    """A model of the field stop."""

    grating: None | esis.optics.Grating = None
    """A model of the diffraction grating array."""

    filter: None | esis.optics.Filter = None
    """A model of the thin-film filters."""

    camera: None | esis.optics.Camera = None
    """A model of the camera and sensors."""

    wavelength: None | u.Quantity | na.AbstractScalar = None
    """A default grid of wavelengths to trace through the system."""

    field: None | na.AbstractCartesian2dVectorArray = None
    """A default grid of field positions to trace through the system."""

    pupil: None | na.AbstractCartesian2dVectorArray = None
    """A default grid of pupil positions to trace through the system."""

    pitch: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The pitch angle of the instrument."""

    yaw: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The yaw angle of the instrument."""

    roll: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The roll angle of the instrument."""

    kwargs_plot: None | dict = None
    """Extra keyword arguments used to plot the optical system."""
