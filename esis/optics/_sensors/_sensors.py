import dataclasses
import astropy.units as u
import named_arrays as na
import optika
import msfc_ccd
from .. import mixins

__all__ = [
    "Sensor",
]


@dataclasses.dataclass(repr=False)
class Sensor(
    optika.mixins.Printable,
    optika.mixins.Rollable,
    optika.mixins.Yawable,
    optika.mixins.Pitchable,
    optika.mixins.Translatable,
    mixins.CylindricallyTransformable,
    msfc_ccd.TeledyneCCD230,
):
    """
    A model of the CCD sensors used to detect light.
    """

    distance_radial: u.Quantity | na.AbstractScalar = 0 * u.mm
    """The distance between the axis of symmetry and the center of the detector."""

    azimuth: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The angle that the detector has been rotated about the axis of symmetry."""

    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
    """An additional translation vector."""

    pitch: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The pitch angle of this sensor."""

    yaw: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The yaw angle of this sensor."""

    roll: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The roll angle of this sensor."""

    position_image: u.Quantity | na.AbstractCartesian2dVectorArray = 0 * u.mm
    """The position of the center of the FOV on the sensor for the target wavelength."""
