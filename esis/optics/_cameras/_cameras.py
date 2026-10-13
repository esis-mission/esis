import dataclasses
import numpy as np
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
    optika.mixins.Replaceable,
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

    dark_current: None | u.Quantity | na.AbstractScalar = None
    """
    The measured rate at which charge accumulates in a pixel which is not
    illuminated.

    This is usually tap-dependent and contains :attr:`axis_tap_x` and
    :attr:`axis_tap_y` dimensions.
    If :obj:`None` (the default), it was not measured.
    Unlike :meth:`msfc_ccd.TeledyneCCD230.dark_current`, which is the typical
    value from the datasheet of the sensor, this does not depend on the
    temperature: it holds only at the temperature it was measured at.
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
        """
        Represent this object as an :mod:`optika` surface.

        :mod:`optika` takes one read noise for the whole sensor, since every
        axis of its parameters is an axis of the system, so a read noise which
        varies from tap to tap is replaced by its quadrature mean over the
        taps.
        """
        read_noise = self.sensor.readout_noise
        axis_tap = tuple(
            axis
            for axis in (self.axis_tap_x, self.axis_tap_y)
            if axis in na.shape(read_noise)
        )
        if axis_tap:
            read_noise = np.sqrt(np.mean(np.square(read_noise), axis=axis_tap))

        return optika.sensors.ImagingSensor(
            name="sensor",
            width_pixel=self.sensor.width_pixel,
            axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
            num_pixel=self.sensor.num_pixel_active,
            timedelta_exposure=self.timedelta_exposure,
            read_noise=read_noise,
            material=self.sensor.material,
            aperture_mechanical=optika.apertures.RectangularAperture(
                half_width=self.sensor.width_package / 2,
            ),
            transformation=self.sensor.transformation,
        )
