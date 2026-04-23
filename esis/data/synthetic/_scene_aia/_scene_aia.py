from typing import Literal
import astropy.units as u
import astropy.time
import named_arrays as na
import sdo
from astropy import constants as const
import numpy as np

__all__ = [
    "scene_aia",
]

def scene_aia(
    time_start: astropy.time.Time,
    time_stop: astropy.time.Time,
    wavelength_aia: u.Quantity | na.AbstractScalarArray,
    wavelength_new: u.Quantity | na.AbstractScalarArray,
    radiance: u.Quantity | na.AbstractScalarArray,
    width_doppler: u.Quantity | na.AbstractScalarArray,
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
    wavelength_aia
        The wavelength label of the AIA channel.
    wavelength_new
        The rest wavelength of each spectral line in the synthetic scene replacing wavelength AIA.
    radiance
        The average radiance of each spectral line in the synthetic scene, units of energy/cm^2/sr/s.
    width_doppler
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

    velocity_max = width_doppler * num_std
    sigma = width_doppler/const.c * wavelength_new

    velocity = na.linspace(
        start=-velocity_max,
        stop=velocity_max,
        axis=axis_velocity,
        num=num_velocity + 1,
    )

    wavelength = (1 + velocity/const.c) * wavelength_new

    obs = sdo.aia.Filtergram.from_time_range(
        time_start=time_start,
        time_stop=time_stop,
        wavelength=wavelength_aia,
        user_email=user_email,
        axis_time=axis_time,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
    )
    axis_detector_xy = axis_detector_x, axis_detector_y

    outputs = radiance * obs.outputs / obs.outputs.mean(axis_detector_xy)

    wv_center = wavelength.cell_centers(axis=axis_velocity)

    gaussian = 1/(sigma.to(wavelength_new.unit)*np.sqrt(2*np.pi)) * np.exp(-((wv_center - wavelength_new) ** 2 / (2 * sigma ** 2)).to(''))

    outputs = outputs*gaussian

    return na.FunctionArray(
        inputs = na.TemporalSpectralPositionalVectorArray(
            time=obs.inputs.time,
            wavelength = wavelength,
            position = obs.inputs.position
        ),
        outputs = outputs,
    )
