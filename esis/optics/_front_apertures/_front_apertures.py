import dataclasses
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractFrontAperture",
    "FrontAperture",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractFrontAperture(
    optika.mixins.Translatable,
):
    """
    An interface describing the entrance aperture of the instrument.
    """

    @property
    def surface(self) -> optika.surfaces.Surface:
        """
        Represent this object as an :mod:`optika` surface.
        """
        return optika.surfaces.Surface(
            name="front aperture",
            transformation=self.transformation,
        )


@dataclasses.dataclass(eq=False, repr=False)
class FrontAperture(
    AbstractFrontAperture,
):
    """
    A model of the entrance aperture of the instrument.
    """

    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
    """
    A transformation which can arbitrarily translate this object.
    """
