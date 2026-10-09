import numpy as np
import scipy.special as sp
import astropy.units as u
import astropy.time
from astropy import constants as const
import named_arrays as na
import sdo

__all__ = [
    "scene_aia",
    "scene_filtergram",
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
    limit: None | int = None,
):
    r"""
    Create a synthetic solar scene composed of AIA images.

    AIA images from channels `wavelength_aia` over a supplied time range are used to
    represent estimates of images at `wavelength_new`.
    A supplied mean radiance is assigned to each image at `wavelength_new`
    and distributed along `axis_velocity` into `num_velocity` bins
    using a Gaussian with standard deviation `width_doppler`.

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
        `wavelength_aia`.
        Elements of `wavelength_new` should correspond to the elements of .
    radiance
        The average radiance of each spectral line in the synthetic scene in
        units of :math:`\text{erg}\,\text{cm}^{-2}\,\text{sr}^{-1}\,\text{s}^{-1}.`
    width_doppler
        The average standard deviation of each spectral line in the synthetic scene.
    axis_time
        The logical axis corresponding to changes in time.
    axis_detector_x
        The logical axis corresponding to changes in detector :math:`x`-coordinate.
    axis_detector_y
        The logical axis corresponding to changes in detector :math:`y`-coordinate.
    axis_velocity
        The logical axis corresponding to changes in line-of-sight velocity.
    num_velocity
        The number of velocity bins in the synthetic scene.
    num_std
        The size of the domain for each spectral line in standard deviation units.
    limit
        The maximum number of files to download per wavelength.

    See Also
    --------
    :func:`scene_filtergram`:
        The same scene from AIA images already in hand.
    :func:`esis.flights.f1.data.synth.scene_aia`:
        A wrapper around :func:`scene_filtergram` for ESIS-I.
    """
    filtergram = sdo.aia.open(
        time_start=time_start,
        time_stop=time_stop,
        wavelength=wavelength_aia,
        axis_time=axis_time,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
        limit=limit,
    )

    return scene_filtergram(
        filtergram=filtergram,
        wavelength_new=wavelength_new,
        radiance=radiance,
        width_doppler=width_doppler,
        axis_velocity=axis_velocity,
        num_velocity=num_velocity,
        num_std=num_std,
    )


def scene_filtergram(
    filtergram: sdo.aia.Filtergram,
    wavelength_new: u.Quantity | na.AbstractScalarArray,
    radiance: u.Quantity | na.AbstractScalarArray,
    width_doppler: u.Quantity | na.AbstractScalarArray,
    axis_velocity: str = "velocity",
    num_velocity: int = 1,
    num_std: float = 3,
) -> na.FunctionArray[na.TemporalSpectralPositionalVectorArray, na.ScalarArray]:
    r"""
    Create a synthetic solar scene from a sequence of AIA images.

    The AIA images in `filtergram` are used to represent estimates of images
    at `wavelength_new`.
    A supplied mean radiance is assigned to each image at `wavelength_new`
    and distributed along `axis_velocity` into `num_velocity` bins
    using a Gaussian with standard deviation `width_doppler`.

    Parameters
    ----------
    filtergram
        The AIA images, such as those loaded by :func:`sdo.aia.open`
        or :meth:`sdo.aia.Filtergram.from_fits`.
    wavelength_new
        The rest wavelength of each spectral line in the synthetic scene
        replacing the wavelength of each AIA channel in `filtergram`.
    radiance
        The average radiance of each spectral line in the synthetic scene in
        units of :math:`\text{erg}\,\text{cm}^{-2}\,\text{sr}^{-1}\,\text{s}^{-1}.`
    width_doppler
        The average standard deviation of each spectral line in the synthetic scene.
    axis_velocity
        The logical axis corresponding to changes in line-of-sight velocity.
    num_velocity
        The number of velocity bins in the synthetic scene.
    num_std
        The size of the domain for each spectral line in standard deviation units.

    See Also
    --------
    :func:`scene_aia`:
        The same scene from the AIA images in a time range, downloaded from
        the JSOC.
    """
    axis_detector_x = filtergram.axis_detector_x
    axis_detector_y = filtergram.axis_detector_y

    velocity_max = width_doppler * num_std

    velocity = na.linspace(
        start=-velocity_max,
        stop=velocity_max,
        axis=axis_velocity,
        num=num_velocity + 1,
    )

    z_a = velocity[{axis_velocity: slice(0, -1)}] / (width_doppler * np.sqrt(2))
    z_b = velocity[{axis_velocity: slice(1, None)}] / (width_doppler * np.sqrt(2))
    gaussian = 0.5 * (sp.erf(z_b) - sp.erf(z_a))

    wavelength = (1 + velocity / const.c) * wavelength_new

    axis_detector_xy = axis_detector_x, axis_detector_y

    crop = {
        axis_detector_x: slice(1024, 1024 + 2048),
        axis_detector_y: slice(1024, 1024 + 2048),
    }

    outputs = (
        radiance * filtergram.outputs / filtergram.outputs[crop].mean(axis_detector_xy)
    )
    delta_lambda = np.diff(wavelength, axis=axis_velocity)
    outputs = outputs * gaussian / delta_lambda

    outputs = np.maximum(outputs, 0)

    return na.FunctionArray(
        inputs=na.TemporalSpectralPositionalVectorArray(
            time=filtergram.inputs.time,
            wavelength=wavelength,
            position=filtergram.inputs.position,
        ),
        outputs=outputs,
    )
