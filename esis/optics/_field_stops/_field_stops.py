import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "FieldStop",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractFieldStop(
    optika.mixins.Printable,
    optika.mixins.Translatable,
):
    """
    An interface describing the field stop of the instrument.
    """

    @property
    @abc.abstractmethod
    def num_folds(self) -> int:
        """
        The order of the rotational symmetry of the optical system.
        """

    @property
    def num_sides(self) -> int:
        """
        The number of sides of the field stop's aperture.
        """
        return self.num_folds

    @property
    @abc.abstractmethod
    def radius_clear(self) -> u.Quantity | na.AbstractScalar:
        """
        The distance from the center to a vertex of the clear aperture.
        """

    @property
    def width_clear(self) -> u.Quantity:
        """
        The width of the clear aperture from edge to edge.
        """
        return 2 * self.radius_clear * np.cos(360 * u.deg / self.num_sides / 2)

    @property
    @abc.abstractmethod
    def radius_mechanical(self) -> u.Quantity | na.AbstractScalar:
        """
        The radius of the exterior edge of the field stop.
        """

    @property
    def surface(self) -> optika.surfaces.Surface:
        """
        Represent this object as an :mod:`optika` surface.
        """
        return optika.surfaces.Surface(
            name="field stop",
            aperture=optika.apertures.RegularPolygonalAperture(
                radius=self.radius_clear,
                num_vertices=self.num_sides,
            ),
            aperture_mechanical=optika.apertures.CircularAperture(
                radius=self.radius_mechanical,
            ),
            is_field_stop=True,
            transformation=self.transformation,
        )


@dataclasses.dataclass(eq=False, repr=False)
class FieldStop(
    AbstractFieldStop,
):
    """
    A model of the field stop of the instrument.

    This element restricts the field of view of the spectrograph to simplify
    the inversion process.
    """

    num_folds: int = 0
    """
    The order of the rotational symmetry of the optical system.
    """

    radius_clear: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The distance from the center to a vertex of the clear aperture.
    """

    radius_mechanical: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The radius of the exterior edge of the field stop.
    """

    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
    """
    A transformation which can arbitrarily translate this object.
    """
