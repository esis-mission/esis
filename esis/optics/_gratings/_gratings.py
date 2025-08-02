import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika.mixins
from .. import mixins

__all__ = [
    "AbstractGrating",
    "Grating",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractGrating(
    optika.mixins.Printable,
    optika.mixins.Rollable,
    optika.mixins.Yawable,
    optika.mixins.Pitchable,
    optika.mixins.Translatable,
    mixins.CylindricallyTransformable,
):
    """
    An interface describing the diffraction gratings of the instrument.
    """

    @property
    @abc.abstractmethod
    def serial_number(self) -> str:
        """
        The serial number of this diffraction grating.
        """

    @property
    @abc.abstractmethod
    def manufacturing_number(self) -> str:
        """
        An additional number describing this diffraction grating.
        """

    @property
    @abc.abstractmethod
    def angle_input(self) -> u.Quantity | na.AbstractScalar:
        """
        The nominal angle of the incident light from the field stop.
        """

    @property
    @abc.abstractmethod
    def angle_output(self) -> u.Quantity | na.AbstractScalar:
        """
        The nominal angle of reflected light to the detectors.
        """

    @property
    @abc.abstractmethod
    def sag(self) -> None | optika.sags.AbstractSag:
        """
        The sag function of this grating.
        """

    @property
    @abc.abstractmethod
    def material(self) -> None | optika.materials.AbstractMaterial:
        """
        The optical material composing this grating.
        """

    @property
    @abc.abstractmethod
    def rulings(self) -> None | optika.rulings.AbstractRulings:
        """
        The ruling pattern of this grating.
        """

    @property
    @abc.abstractmethod
    def num_folds(self) -> int:
        """
        The order of the rotational symmetry of the optical system.
        This determines the aperture wedge angle of this grating.
        """

    @property
    def angle_aperture(self) -> u.Quantity | na.AbstractScalar:
        r"""
        The angle of the grating's aperture.

        This is equal to :math:`2 \pi / n` radians, where :math:`n` is the
        order of the rotational symmetry of the optical system.
        """
        return (360 * u.deg) / self.num_folds

    @property
    @abc.abstractmethod
    def halfwidth_inner(self) -> u.Quantity | na.AbstractScalar:
        """
        The distance from the apex to the inner edge of the clear aperture.
        """

    @property
    @abc.abstractmethod
    def halfwidth_outer(self) -> u.Quantity | na.AbstractScalar:
        """
        The distance from the apex to the outer edge of the clear aperture.
        """

    @property
    @abc.abstractmethod
    def width_border(self) -> u.Quantity | na.AbstractScalar:
        """
        The nominal width of the border around the clear aperture.
        """

    @property
    @abc.abstractmethod
    def width_border_inner(self) -> u.Quantity | na.AbstractScalar:
        """
        The width of the border between the inner edge of the clear aperture
        and the substrate inner edge of the substrate.
        """

    @property
    @abc.abstractmethod
    def clearance(self) -> u.Quantity | na.AbstractScalar:
        """
        The minimum distance between adjacent physical gratings.
        """

    @property
    def transformation(self) -> na.transformations.AbstractTransformation:
        rotation = na.transformations.Cartesian3dRotationX(180 * u.deg)
        return super().transformation @ rotation

    @property
    def surface(self) -> optika.surfaces.Surface:
        """
        Represent this object as an :mod:`optika` surface.
        """
        angle_aperture = self.angle_aperture
        halfwidth_inner = self.halfwidth_inner
        halfwidth_outer = self.halfwidth_outer
        width_border = self.width_border
        width_border_inner = self.width_border_inner
        clearance = self.clearance / np.sin(angle_aperture / 2)
        distance_radial = self.distance_radial
        side_border_x = width_border / np.sin(angle_aperture / 2) + clearance
        offset_clear = distance_radial - side_border_x
        offset_mechanical = distance_radial - clearance
        return optika.surfaces.Surface(
            name="grating",
            sag=self.sag,
            material=self.material,
            aperture=optika.apertures.IsoscelesTrapezoidalAperture(
                x_left=offset_clear - halfwidth_inner,
                x_right=offset_clear + halfwidth_outer,
                angle=angle_aperture,
                transformation=na.transformations.Cartesian3dTranslation(
                    x=-offset_clear,
                ),
            ),
            aperture_mechanical=optika.apertures.IsoscelesTrapezoidalAperture(
                x_left=offset_mechanical - (halfwidth_inner + width_border_inner),
                x_right=offset_mechanical + halfwidth_outer + width_border,
                angle=angle_aperture,
                transformation=na.transformations.Cartesian3dTranslation(
                    x=-offset_mechanical,
                ),
            ),
            rulings=self.rulings,
            is_pupil_stop=True,
            transformation=self.transformation,
        )


@dataclasses.dataclass(eq=False, repr=False)
class Grating(
    AbstractGrating,
):
    """
    A model of the diffraction gratings of this instrument.

    The gratings disperse incident light onto the detectors.
    """

    serial_number: str = ""
    """
    The serial number of this diffraction grating.
    """

    manufacturing_number: str = ""
    """
    An additional number describing this diffraction grating.
    """

    angle_input: u.Quantity = 0 * u.deg
    """
    The nominal angle of the incident light from the field stop.
    """

    angle_output: u.Quantity = 0 * u.deg
    """
    The nominal angle of reflected light to the detectors.
    """

    sag: None | optika.sags.AbstractSag = None
    """
    The sag function of this grating.
    """

    material: None | optika.materials.AbstractMaterial = None
    """
    The optical material composing this grating.
    """
    rulings: None | optika.rulings.AbstractRulings = None
    """
    The ruling pattern of this grating.
    """

    num_folds: int = 0
    """
    The order of the rotational symmetry of the optical system.
    This determines the aperture wedge angle of this grating.
    """

    halfwidth_inner: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The distance from the apex to the inner edge of the clear aperture.
    """

    halfwidth_outer: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The distance from the apex to the outer edge of the clear aperture.
    """

    width_border: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The nominal width of the border around the clear aperture.
    """

    width_border_inner: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The width of the border between the inner edge of the clear aperture
    and the substrate inner edge of the substrate.
    """

    clearance: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The minimum distance between adjacent physical gratings.
    """

    distance_radial: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The distance of this object from the axis of symmetry.
    """

    azimuth: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The angle of rotation about the axis of symmetry.
    """

    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
    """
    A transformation which can arbitrarily translate this object.
    """

    pitch: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The pitch angle of this object.
    """

    yaw: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The yaw angle of this object.
    """

    roll: u.Quantity | na.AbstractScalar = 0 * u.deg
    """
    The roll angle of this object
    """
