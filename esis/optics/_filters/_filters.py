import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika
from .. import mixins

__all__ = [
    "AbstractFilter",
    "Filter",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractFilter(
    optika.mixins.Printable,
    optika.mixins.Rollable,
    optika.mixins.Yawable,
    optika.mixins.Pitchable,
    optika.mixins.Translatable,
    mixins.CylindricallyTransformable,
):
    """An interface describing the visible-light filters of the instrument."""

    @property
    @abc.abstractmethod
    def material(self) -> optika.materials.AbstractThinFilmFilter:
        """A model of the filter material including the mesh and oxide."""

    @property
    @abc.abstractmethod
    def radius_clear(self) -> u.Quantity | na.AbstractScalar:
        """The radius of the filter's circular clear aperture."""

    @property
    @abc.abstractmethod
    def width_border(self) -> u.Quantity | na.AbstractScalar:
        """The width of the frame around the clear aperture."""

    @property
    def surface(self) -> optika.surfaces.Surface:
        """Represent this object as an :mod:`optika` surface."""
        radius_clear = self.radius_clear
        radius_mech = radius_clear + self.width_border
        aperture = optika.apertures.CircularAperture(radius_clear)
        aperture_mechanical = optika.apertures.CircularAperture(radius_mech)

        return optika.surfaces.Surface(
            name="filter",
            material=self.material,
            aperture=aperture,
            aperture_mechanical=aperture_mechanical,
            transformation=self.transformation,
        )


@dataclasses.dataclass(eq=False, repr=False)
class Filter(
    AbstractFilter,
):
    """
    A model of the visible-light filters of the instrument.

    These are thin-film filters supported by a fine mesh.
    """

    material: None | optika.materials.AbstractMaterial = None
    """A model of the filter material including the mesh and oxide."""

    radius_clear: u.Quantity | na.AbstractScalar = 0 * u.mm
    """The radius of the filter's circular clear aperture."""

    width_border: u.Quantity | na.AbstractScalar = 0 * u.mm
    """The width of the frame around the clear aperture."""

    distance_radial: u.Quantity | na.AbstractScalar = 0 * u.mm
    """The distance of this object from the axis of symmetry."""

    azimuth: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The angle of rotation about the axis of symmetry."""

    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
    """A transformation which can arbitrarily translate this object."""

    pitch: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The pitch angle of this object."""

    yaw: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The yaw angle of this object."""

    roll: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The roll angle of this object"""
