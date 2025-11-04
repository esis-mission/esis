import dataclasses
import astropy.units as u
import named_arrays as na
import optika
import msfc_ccd
from .. import Sensor

__all__ = [
    "Camera",
]


@dataclasses.dataclass(eq=False, repr=False)
class Camera(
    optika.mixins.Printable,
    msfc_ccd.Camera,
):
    """A model of the cameras developed by MSFC."""

    sensor: None | Sensor = None
    """
    A model of the sensor used by this camera to capture light.

    If :obj:`None` (the default), :class:`esis.optics.Sensor()` will be used.
    """

    gain: None | u.Quantity | na.AbstractScalar = 2.5 * u.electron / u.DN
    """
    The conversion factor between electrons and DN.

    This is usually tap-dependent and contains :attr:`axis_tap_x` and
    :attr:`axis_tap_y` dimensions.
    """

    timedelta_sync: u.Quantity = 0 * u.s
    """The synchronization error between the different channels."""

    channel: str | na.AbstractScalar = ""
    """Human-readable name of each channel of this camera array."""

    channel_trigger: int = 0
    """The master channel which triggers the other channels to start exposing."""

    def __post_init__(self):
        if self.sensor is None:
            self.sensor = Sensor()

    @property
    def surface(self) -> optika.sensors.AbstractImagingSensor:
        """Represent this object as an :mod:`optika` surface."""
        return optika.sensors.ImagingSensor(
            name="sensor",
            width_pixel=self.sensor.width_pixel,
            axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
            num_pixel=self.sensor.num_pixel_active,
            timedelta_exposure=self.timedelta_exposure,
            material=optika.sensors.materials.e2v_ccd97(
                temperature=self.sensor.temperature,
            ),
            aperture_mechanical=optika.apertures.RectangularAperture(
                half_width=self.sensor.width_package / 2,
            ),
            transformation=self.sensor.transformation,
        )
