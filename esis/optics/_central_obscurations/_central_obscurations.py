import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "CentralObscuration",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCentralObscuration(
    optika.mixins.Printable,
    optika.mixins.Translatable,
):
    """
    An interface defining the central obscuration of the instrument.
    """

    @property
    @abc.abstractmethod
    def num_folds(self) -> int:
        """
        The order of the rotational symmetry of the optical system.
        """

    @property
    @abc.abstractmethod
    def halfwidth(self) -> u.Quantity | na.AbstractScalar:
        """
        The distance from the center to the edge of the obscuration.
        """

    @property
    def radius(self) -> u.Quantity | na.AbstractScalar:
        """
        The distance from the center to a vertex of the obscuration.
        """
        return self.halfwidth / np.cos(360 * u.deg / self.num_folds / 2)

    @property
    @abc.abstractmethod
    def remove_last_vertex(self) -> bool:
        """
        A flag controlling whether the last vertex should be removed.
        """

    @property
    def surface(self) -> optika.surfaces.Surface:
        """
        Represent this object as an :mod:`optika` surface.
        """

        num_folds = self.num_folds
        radius = self.radius
        offset_angle = 360 * u.deg / num_folds
        angle = na.linspace(
            start=0 * u.deg,
            stop=360 * u.deg,
            num=num_folds,
            axis="vertex",
            endpoint=False,
        )
        if self.remove_last_vertex:
            angle = angle[dict(vertex=slice(None, ~0))]
        angle = angle - offset_angle
        return optika.surfaces.Surface(
            name="obscuration",
            aperture=optika.apertures.PolygonalAperture(
                vertices=na.Cartesian3dVectorArray(
                    x=radius * np.cos(angle),
                    y=radius * np.sin(angle),
                    z=0 * u.mm,
                ),
                inverted=True,
            ),
            transformation=self.transformation,
        )


@dataclasses.dataclass(eq=False, repr=False)
class CentralObscuration(
    AbstractCentralObscuration,
):
    """
    A model of the Tuffet, which holds the diffraction gratings and obscures
    the central portion of the primary mirror.
    """

    num_folds: int = 0
    """
    The order of the rotational symmetry of the optical system.
    """

    halfwidth: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The distance from the center to the edge of the obscuration.
    """

    remove_last_vertex: bool = False
    """
    A flag controlling whether the last vertex should be removed.
    """

    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
    """
    A transformation which can arbitrarily translate this object.
    """
