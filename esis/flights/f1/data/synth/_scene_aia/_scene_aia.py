import astropy.units as u
import astropy.time
import named_arrays as na
import esis
from ... import level_1
from ....spectrum import O_V, Mg_X, He_I

__all__ = [
    "scene_aia",
]


@esis.memory.cache
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
    limit: None | int = None,
):
    r"""
    Load a synthetic solar scene composed of AIA images captured during the flight.

    The result is cached to disk, so the JSOC queries are only performed on
    the first call.

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
    limit
        The maximum number of files to download per wavelength.
    """
    l1 = level_1()

    if time_start is None:
        time_start = l1.inputs.time_start[{l1.axis_time: 0}].ndarray.mean()

    if time_stop is None:
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

    # One spectral line at a time, cropped to the central third of the AIA
    # frame (the part covering the ESIS field of view) before the radiometric
    # arithmetic: building all three lines from full AIA frames at once peaks
    # at ~70 GiB, which exhausts the documentation build; chunked and cropped
    # it stays under ~25 GiB, and the kept pixels are identical.
    num_wavelength = wavelength_aia.shape[axis_wavelength]
    results = [
        esis.data.synth.scene_aia(
            time_start=time_start,
            time_stop=time_stop,
            wavelength_aia=wavelength_aia[{axis_wavelength: slice(i, i + 1)}],
            wavelength_new=wavelength_esis[{axis_wavelength: slice(i, i + 1)}],
            radiance=radiance_esis[{axis_wavelength: slice(i, i + 1)}],
            width_doppler=width_esis[{axis_wavelength: slice(i, i + 1)}],
            axis_time=axis_time,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
            axis_velocity=axis_velocity,
            num_velocity=num_velocity,
            num_std=num_std,
            limit=limit,
            crop={
                axis_detector_x: (1 / 3, 2 / 3),
                axis_detector_y: (1 / 3, 2 / 3),
            },
        )
        for i in range(num_wavelength)
    ]

    return na.concatenate(results, axis=axis_wavelength)
