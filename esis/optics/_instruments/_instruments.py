from __future__ import annotations
import abc
import dataclasses
import functools
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import esis

__all__ = [
    "AbstractInstrument",
    "Instrument",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractInstrument(
    optika.mixins.Printable,
    optika.mixins.Rollable,
    optika.mixins.Yawable,
    optika.mixins.Pitchable,
):
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
        """A default grid of wavelengths to trace through the system."""

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
    def requirements(self) -> None | esis.optics.Requirements:
        """The required optical performance of the instrument."""

    @property
    @abc.abstractmethod
    def kwargs_plot(self):
        """Extra keyword arguments used to plot the optical system."""

    @property
    def angle_grating_input(self) -> na.AbstractScalar:
        """The angle between the grating normal and the direction of the incident light."""
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

    @functools.cached_property
    def system(self) -> optika.systems.SequentialSystem:
        """
        Resolve this optics model into an instance of :class:`optika.systems.SequentialSystem`.

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
                wavelength=self.wavelength,
                field=self.field,
                pupil=self.pupil,
            ),
            transformation=self.transformation,
            kwargs_plot=self.kwargs_plot,
        )

        return result


@dataclasses.dataclass(eq=False, repr=False)
class Instrument(
    AbstractInstrument,
):
    """
    A generalized model of the ESIS instrument system.

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

    requirements: None | esis.optics.Requirements = None
    """The optical requirements of the instrument."""

    kwargs_plot: None | dict = None
    """Extra keyword arguments used to plot the optical system."""
