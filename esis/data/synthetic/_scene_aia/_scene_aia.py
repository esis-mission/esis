from typing import Literal
import astropy.units as u
import astropy.time
import named_arrays as na
import sdo

__all__ = [
    "scene_aia",
]

def scene_aia(
    time_start: astropy.time.Time,
    time_stop: astropy.time.Time,
    wavelength: u.Quantity | na.AbstractScalarArray,
    wavelength_new: u.Quantity | na.AbstractScalarArray,
    radiance: u.Quantity | na.AbstractScalarArray,
    width: u.Quantity | na.AbstractScalarArray,
    axis_time: str = "time",
    axis_detector_x: str = "detector_x",
    axis_detector_y: str = "detector_y",
    axis_velocity: str = "velocity",
    num_velocity: int = 1,
    num_std: float = 3,
    user_email: None | str = None,
):
    """
    Create a synthetic solar scene using AIA images from the specified
    time range shifted to a different wavelength.

    Parameters
    ---------
    time_start
        The start time of the AIA observations.
    time_stop
        The stop time of the AIA observations.
    wavelength
        The wavelength of the AIA observations.
    wavelength_new
        The rest wavelength of each spectral line in the synthetic scene.
    radiance
        The average radiance of each spectral line in the synthetic scene.
    width
        The average standard deviation of each spectral line in the synthetic scene.
    axis_time
        The logical axis corresponding to changes in time.
    axis_detector_x
        The logical axis corresponding to changes in detector :math:`x`-coordinate.
    axis_detector_y
        The logical axis corresponding to changes in detector :math:`y`-coordinate.
    num_velocity
        The number of velocity bins in the synthetic scene.
    num_std
        The size of the domain for each spectral line in standard deviation units.
    user_email
        An email address used to notify the user that their JSOC request
        is complete.
        This email must be registered with JSOC before using this function.
        If :obj:`None`, the value is taken from the ``JSOC_EMAIL``
        environment variable.
    """

    obs = sdo.aia.Filtergram.from_time_range(
        time_start=time_start,
        time_stop=time_stop,
        wavelength=wavelength,
        user_email=user_email,
        axis_time=axis_time,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
    )

    axis_detector_xy = axis_detector_x, axis_detector_y

    outputs = radiance * obs.outputs / obs.outputs.mean(axis_detector_xy)

    velocity_max = width * num_std

    velocity = na.linspace(
        start=-velocity_max,
        stop=velocity_max,
        axis=axis_velocity,
        num=num_velocity + 1,
    )

    v = velocity.cell_centers()

    na.TemporalSpectralPositionalVectorArray
