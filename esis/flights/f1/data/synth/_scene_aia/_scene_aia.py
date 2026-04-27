import astropy.units as u
import astropy.time
import named_arrays as na
import esis
from ... import level_1
from ....spectrum import O_V, Mg_X, He_I

__all__ = [
    "scene_aia",
]


def scene_aia(
    time_start: None | astropy.time.Time = None,
    time_stop: None | astropy.time.Time = None,
    axis_time: str = "time",
    axis_detector_x: str = "detector_x",
    axis_detector_y: str = "detector_y",
    axis_wavelength: str = "wavelength",
    axis_velocity: str = "velocity",
    num_velocity: int = 1,
    num_std: float = 3,
    user_email: None | str = None,
    limit: None | int = None,
):
    r"""
    Load a synthetic solar scene composed of AIA images captured during the flight.

    This function plugs the spectral line properties in
    :mod:`esis.flights.f1.spectrum` into :func:`esis.data.synth.scene_aia`
    to produce the synthic scene.

    Only the brightest three lines in the ESIS passband are recreated:
    :math:`\text{O\,V}\;630\,\AA`, :math:`\text{Mg\,X}\;609\,\AA`, and
    :math:`\text{He\,I}\;584\,\AA`.

    Parameters
    ----------
    time_start
        The start time of the AIA observations.
        If :obj:`None`, the start time of the ESIS Level-1 observations is used.
    time_stop
        The stop time of the AIA observations.
        If :obj:`None`, the stop time of the ESIS Level-1 observations is used.
    axis_time
        The logical axis corresponding to changes in time.
    axis_detector_x
        The logical axis corresponding to changes in detector :math:`x`-coordinate.
    axis_detector_y
        The logical axis corresponding to changes in detector :math:`y`-coordinate.
    axis_wavelength
        The logical axis corresponding to changing spectral line.
    axis_velocity
       The logical axis corresponding to changes in line-of-sight velocity.
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
    limit
        The maximum number of files to download per wavelength.
    """
    l1 = None

    if time_start is None:
        if l1 is None:
            l1 = level_1()
        time_start = l1.inputs.time_start[{l1.axis_time: 0}].ndarray.mean()

    if time_stop is None:
        if l1 is None:
            l1 = level_1()
        time_stop = l1.inputs.time_end[{l1.axis_time: ~0}].ndarray.mean()

    wavelength_aia = [304, 193, 304] * u.AA

    wavelength_esis = [
        O_V.wavelength,
        Mg_X.wavelength,
        He_I.wavelength,
    ]

    radiance_esis = [
        O_V.radiance,
        Mg_X.radiance,
        He_I.radiance,
    ]

    width_esis = [
        O_V.width_doppler,
        Mg_X.width_doppler,
        He_I.width_doppler,
    ]

    wavelength_aia = na.ScalarArray(wavelength_aia, axis_wavelength)
    wavelength_esis = na.stack(wavelength_esis, axis_wavelength)
    radiance_esis = na.stack(radiance_esis, axis_wavelength)
    width_esis = na.stack(width_esis, axis_wavelength)

    result = esis.data.synth.scene_aia(
        time_start=time_start,
        time_stop=time_stop,
        wavelength_aia=wavelength_aia,
        wavelength_new=wavelength_esis,
        radiance=radiance_esis,
        width_doppler=width_esis,
        axis_time=axis_time,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
        axis_velocity=axis_velocity,
        num_velocity=num_velocity,
        num_std=num_std,
        user_email=user_email,
        limit=limit,
    )

    shape = result.outputs.shape

    num_x = shape[axis_detector_x]
    num_y = shape[axis_detector_y]

    crop = {
        axis_detector_x: slice(num_x // 3, 2 * num_x // 3),
        axis_detector_y: slice(num_y // 3, 2 * num_y // 3),
    }

    result = result[crop]

    return result
