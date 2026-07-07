from typing import Literal
import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na
import iris

__all__ = [
    "scene_iris",
]


def scene_iris(
    time_start: str | astropy.time.Time,
    time_stop: None | str | astropy.time.Time,
    wavelength_rest: u.Quantity,
    radiance: u.Quantity,
    fwhm: u.Quantity,
    axis_time: str = "time",
    axis_detector_x: str = "detector_x",
    axis_detector_y: str = "detector_y",
    axis_velocity: str = "velocity",
    velocity_max: None | u.Quantity = None,
    despike: bool = True,
    background_removal: None | Literal["trim_mean"] = None,
    dn_min: u.Quantity = 8 * u.DN,
    dn_zero: u.Quantity = 0.01 * u.DN,
    **kwargs,
) -> na.FunctionArray[
    na.TemporalDopplerPositionalVectorArray,
    na.ScalarArray,
]:
    """
    Create a synthetic solar scene using IRIS images.

    IRIS spectroheliograms from a specified time range are shifted and scaled
    to the given wavelength range to simulate a scene observed by ESIS.

    Parameters
    ----------
    time_start
        The start time of the IRIS observations.
    time_stop
        The ending time of the IRIS observations.
        If :obj:`None`, one second is added to `time_start`, which usually
        has the effect of selecting a single observation.
    wavelength_rest
        The new rest wavelength of the simulated scene.
        This replaces the actual rest wavelength of the IRIS observations.
    radiance
        The average radiance of the simulated scene.
        This replaces the actual radiance of the IRIS observations.
    fwhm
        The average full-width half maximum, in wavelength units, of the
        dominant spectral line in the simulated scene.
        The wavelength axis of the IRIS observations will be scaled to match
        this value.
    axis_time
        The logical axis corresponding to changes in time.
    axis_detector_x
        The logical axis corresponding to changes in detector :math:`x`-coordinate.
    axis_detector_y
        The logical axis corresponding to changes in detector :math:`y`-coordinate.
    axis_velocity
        The logical axis corresponding to changes in line-of-sight velocity.
    velocity_max
        The maximum doppler velocity of the simulated scene.
        Values outside this range are cropped.
    despike
        Whether to remove cosmic ray spikes using :mod:`astroscrappy`.
    background_removal
        The background removal algorithm to remove stray light.
        Currently, the options are :obj:`None` (the default) which is no
        background removal, and ``trim_mean`` which uses a trimmed mean
        to estimate a constant background value.
    dn_min
        To prevent negatives in the final scene without biasing areas with zero
        flux, pixels with a value less than `dn_min` will be set to `dn_zero`.
    dn_zero
        If a pixel truly has zero flux, this presents a problem since the
        uncertainty of this pixel is infinite.
        To prevent this, we interpret "zero flux" as a very small number called
        `dn_zero`.
    kwargs
        Additional keyword arguments passed to :func:`iris.sg.open`

    Raises
    ------
    ValueError
        If `background_removal` is not recognized.
    """
    scene = iris.sg.open(
        time=time_start,
        time_stop=time_stop,
        axis_wavelength=axis_velocity,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
        **kwargs,
    )

    scene = scene.explicit

    if despike:
        scene = na.despike(
            array=scene,
            axis=(axis_velocity, axis_detector_y),
        )

    if background_removal is not None:

        if background_removal == "trim_mean":
            bg = na.mean_trimmed(
                a=scene.outputs[np.isfinite(scene.outputs)],
                q=0.49,
            )

        else:  # pragma: nocover
            raise ValueError(f"{background_removal=} not recognized.")

        scene.outputs = scene.outputs - bg

    if velocity_max is not None:

        velocity_centers = scene.inputs.velocity.cell_centers(axis_velocity)

        where_upper = +velocity_max < velocity_centers

        index_lower = np.nanargmax(-velocity_max < velocity_centers)[axis_velocity]
        index_lower = index_lower.ndarray

        if where_upper.any():
            index_upper = np.nanargmax(where_upper)[axis_velocity].ndarray
        else:
            index_upper = None

        crop_wavelength = {
            scene.axis_wavelength: slice(index_lower, index_upper)
        }

        scene = scene[crop_wavelength]

    scene.outputs = np.nan_to_num(scene.outputs)

    scene.outputs[scene.outputs < dn_min] = dn_zero

    spectrum = scene.mean((axis_time, axis_detector_x, axis_detector_y))

    fwhm_avg = na.pdf.fwhm(
        x=spectrum.inputs.wavelength,
        f=spectrum.outputs,
        axis=axis_velocity,
    )

    scale = (fwhm / wavelength_rest) / (fwhm_avg / scene.inputs.wavelength_rest)

    scene.inputs = na.TemporalDopplerPositionalVectorArray.from_velocity(
        velocity=scene.inputs.velocity * scale,
        wavelength_rest=wavelength_rest,
        time=scene.inputs.time,
        position=scene.inputs.position,
    )

    radiance_avg = scene.integrate(
        component="wavelength",
        axis=axis_velocity,
    ).outputs.mean()

    scene.outputs = scene.outputs * radiance / radiance_avg

    return scene
