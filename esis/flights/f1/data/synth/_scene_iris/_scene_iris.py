from typing import Literal
import astropy.units as u
import astropy.time
import named_arrays as na
import esis
from ....spectrum import O_V, Si_IV

__all__ = [
    "radiance_scale_default",
    "velocity_scale_default",
    "scene_iris",
]

#: The default factor by which the radiance of the IRIS observations is
#: scaled, which is the ratio of the average quiet-sun radiances of the
#: simulated and observed lines.
radiance_scale_default = float(O_V.radiance / Si_IV.radiance)

#: The default factor by which the Doppler velocity of the IRIS observations
#: is scaled, which maps the average quiet-sun width of the observed line onto
#: the average quiet-sun width of the simulated line.
velocity_scale_default = float(
    (O_V.fwhm / O_V.wavelength) / (Si_IV.fwhm / Si_IV.wavelength)
)


def scene_iris(
    time_start: str | astropy.time.Time,
    time_stop: None | str | astropy.time.Time = None,
    wavelength_rest: u.Quantity = O_V.wavelength,
    radiance_scale: float = radiance_scale_default,
    velocity_scale: float = velocity_scale_default,
    axis_time: str = "time",
    axis_detector_x: str = "detector_x",
    axis_detector_y: str = "detector_y",
    axis_velocity: str = "velocity",
    velocity_max: None | u.Quantity = 300 * u.km / u.s,
    despike: bool = True,
    background_removal: None | Literal["trim_mean"] = None,
    dn_min: u.Quantity = 8 * u.DN,
    dn_zero: u.Quantity = 0.01 * u.DN,
    **kwargs,
) -> na.FunctionArray[
    na.TemporalDopplerPositionalVectorArray,
    na.ScalarArray,
]:
    r"""
    Create a synthetic scene of the :math:`\text{O\,V}\;630\,\AA` spectral line.

    IRIS spectroheliograms from a specified time range are shifted and scaled
    to the :math:`\text{O\,V}\;630\,\AA` wavelength range to simulate a scene
    observed by ESIS.

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
    radiance_scale
        The factor by which to scale the radiance of the IRIS observations.
        Defaults to the ratio of the average quiet-sun radiances of the
        simulated and observed lines, see :obj:`radiance_scale_default`.
    velocity_scale
        The factor by which to scale the Doppler velocity of the IRIS
        observations.
        Defaults to the ratio of the average quiet-sun widths of the simulated
        and observed lines, see :obj:`velocity_scale_default`.
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

    Examples
    --------
    Load a synthetic :math:`\text{O\,V}\;630\,\AA` scene and display it as
    a false-color image.

    .. jupyter-execute::

        import esis

        scene = esis.flights.f1.data.synth.scene_iris("2014-07-04 11:40")

        scene.show();
    """
    return esis.data.synth.scene_iris(
        time_start=time_start,
        time_stop=time_stop,
        wavelength_rest=wavelength_rest,
        radiance_scale=radiance_scale,
        velocity_scale=velocity_scale,
        axis_time=axis_time,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
        axis_velocity=axis_velocity,
        velocity_max=velocity_max,
        despike=despike,
        background_removal=background_removal,
        dn_min=dn_min,
        dn_zero=dn_zero,
        **kwargs,
    )
