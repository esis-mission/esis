import astropy.units as u
import astropy.time
import named_arrays as na
import sdo
from astropy import constants as const
import numpy as np
from scipy.special import erf

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
    Create a synthetic solar scene using AIA images.

    AIA images from channels wavelength_aia over a supplied time range are used to
    represent estimates of images at wavelength_new.  A supplied mean radiance is
    assigned to
    each image at wavelength_new and distributed using a gaussian distribution along
    axis_velocity with width_doppler and into num_velocity
    bins.

    Parameters
    ----------
    time_start
        The start time of the AIA observations.
    time_stop
        The stop time of the AIA observations.
    wavelength_aia
        The wavelength label of the AIA channel.
    wavelength_new
        The rest wavelength of each spectral line in the synthetic scene replacing
        wavelength AIA.
        Axes of
        wavelength_new should be aligned to the axes of wavelength_aia.
    radiance
        The average radiance of each spectral line in the synthetic scene,
        units of
        energy/cm^2/sr/s.
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
    use  # noqa: E501
        An email address used to notify the user that their JSOC request
        is complete.
        This email must be registered with JSOC before using this function.
        If :obj:`None`, the value is taken from the ``JSOC_EMAIL``
        environment variable.

    """
    velocity_max = width_doppler * num_std

    velocity = na.linspace(
        start=-velocity_max,
        stop=velocity_max,
        axis=axis_velocity,
        num=num_velocity + 1,
    )

    z_a = velocity[{axis_velocity: slice(0, -1)}] / (width_doppler * np.sqrt(2))
    z_b = velocity[{axis_velocity: slice(1, None)}] / (width_doppler * np.sqrt(2))
    gaussian = 0.5 * (erf(z_b) - erf(z_a))

    wavelength = (1 + velocity / const.c) * wavelength_new

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
    delta_lambda = np.diff(wavelength, axis=axis_velocity)
    outputs = outputs * gaussian / delta_lambda

    return na.FunctionArray(
        inputs=na.TemporalSpectralPositionalVectorArray(
            time=obs.inputs.time, wavelength=wavelength, position=obs.inputs.position
        ),
        outputs=outputs,
    )
