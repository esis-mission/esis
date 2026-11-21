import typing
import warnings
import dataclasses
import numpy as np
import scipy.ndimage
import astropy.units as u
import named_arrays as na
import soho

__all__ = [
    "scene_sumer_observation",
]

_unit_scene = u.erg / (u.cm**2 * u.sr * u.nm * u.s)
"""The unit of the radiance of the scene, as in :func:`~esis.data.synth.scene_iris`."""


def scene_sumer_observation(
    observation: soho.sumer.SpectrographObservation,
    velocity_max: None | u.Quantity = 125 * u.km / u.s,
    velocity_continuum: None | u.Quantity = 100 * u.km / u.s,
    size_continuum: int = 5,
    fill: None | u.Quantity = None,
) -> soho.sumer.SpectrographObservation:
    """
    Create a synthetic solar scene from the spectra of SUMER.

    The Solar Ultraviolet Measurements of Emitted Radiation (SUMER)
    spectrograph observed the lines ESIS does, such as O V 629.7 Angstroms,
    so its spectra are used as they are, in their own radiances,
    without the scaling which :func:`~esis.data.synth.scene_iris` needs.
    The continuum under the line is subtracted, the spectra are cut to the
    velocities of the line, and the voxels SUMER did not observe,
    which are not a number (NaN), are filled if asked to.
    The scene keeps the type of the observation,
    so it can be displayed with
    :meth:`~soho.sumer.SpectrographObservation.show`.

    Parameters
    ----------
    observation
        The spectra of one raster, such as those loaded by
        :meth:`~soho.sumer.SpectrographObservation.from_level1`,
        with the zero of their velocities where it should be.
    velocity_max
        The largest speed of the scene, toward or away from the observer.
        The columns whose middles are beyond it are left out.
        If :obj:`None`, every column is kept.
    velocity_continuum
        The least speed of the columns from which the continuum is measured.
        The continuum of each pixel is the mean of those columns which are
        kept, averaged over `size_continuum` pixels of the raster in each
        direction, and is subtracted from every column.
        If :obj:`None`, or if no column is that fast,
        the continuum is not subtracted.
    size_continuum
        The number of pixels of the raster, in each direction,
        over which the continuum is averaged.
    fill
        The radiance of the voxels which SUMER did not observe.
        If :obj:`None`, they are left not a number (NaN),
        which an instrument cannot image,
        so give a value before imaging the scene.

    Raises
    ------
    ValueError
        If no column of the observation is within `velocity_max`.
    """
    axis_x = observation.axis_detector_x
    axis_y = observation.axis_detector_y
    axis_velocity = observation.axis_wavelength
    axes = (axis_x, axis_y, axis_velocity)

    velocity = observation.inputs.velocity.to(u.km / u.s)
    middle = typing.cast(na.ScalarArray, velocity.cell_centers(axis_velocity))
    speed = np.abs(typing.cast(u.Quantity, middle.ndarray_aligned((axis_velocity,))))

    outputs = typing.cast(na.ScalarArray, observation.outputs)
    radiance = typing.cast(u.Quantity, outputs.ndarray_aligned(axes)).to(_unit_scene)
    value = np.asarray(radiance.value, dtype=float)

    kept = np.ones(speed.shape, dtype=bool)
    if velocity_max is not None:
        kept = speed <= velocity_max

    if velocity_continuum is not None:
        columns = kept & (speed >= velocity_continuum)
        if np.any(columns):
            with warnings.catch_warnings():
                # pixels which have no data in those columns have no continuum
                warnings.simplefilter("ignore", RuntimeWarning)
                level = np.nanmean(value[..., columns], axis=~0)
            level = np.nan_to_num(level)
            level = scipy.ndimage.uniform_filter(level, size=size_continuum)
            value = value - level[..., np.newaxis]

    if fill is not None:
        value = np.where(np.isfinite(value), value, fill.to_value(_unit_scene))

    index = np.flatnonzero(kept)
    if not index.size:
        raise ValueError(f"No column of the observation is within {velocity_max}.")
    cut = {axis_velocity: slice(int(index[0]), int(index[~0]) + 1)}

    scene = typing.cast(soho.sumer.SpectrographObservation, observation[cut])
    outputs = na.ScalarArray(value[..., cut[axis_velocity]] << _unit_scene, axes)
    return dataclasses.replace(scene, outputs=outputs)
