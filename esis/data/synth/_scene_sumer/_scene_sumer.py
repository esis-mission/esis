import typing
import warnings
import dataclasses
import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.signal
import astropy.units as u
import named_arrays as na
import soho

__all__ = [
    "scene_sumer_observation",
]

_unit_scene = u.erg / (u.cm**2 * u.sr * u.nm * u.s)
"""The unit of the radiance of the scene, as in :func:`~esis.data.synth.scene_iris`."""

_unit_velocity = u.km / u.s
"""The unit in which the velocities are worked with, as plain numbers."""

_shift_neighbour_max = 30 * u.km / u.s
"""How far a neighbouring line may be from where it is expected."""

_width_neighbour = 50 * u.km / u.s
"""
How far from the center of a neighbouring line it is fitted,
on a sloping floor which takes the far wing of the line and the continuum.
"""

_sigma_neighbour_guess = 15 * u.km / u.s
"""The standard deviation of a neighbouring line which its fit starts from."""

_sigma_neighbour_min = 5 * u.km / u.s
"""The least standard deviation of a neighbouring line."""

_sigma_neighbour_max = 50 * u.km / u.s
"""The greatest standard deviation of a neighbouring line, broad near the limb."""

_num_column_min = 6
"""The fewest observed columns a neighbouring line is fitted to."""

_ratio_edge = 0.5
"""
The fraction of the column inside it below which a column at the end of the
window, in the mean profile, is taken to be only partly lit.

The edge of the window does not line up with the columns of the level-1
files: it slides across them along the slit and from one exposure to the next,
so the mean profile falls toward each end over one to three columns,
each dimmer than half the column inside it.
"""

_num_column_edge_max = 3
"""The most columns at each end of the window which are taken to be only partly lit."""

_num_sigma_covered = 2
"""
How many standard deviations past the center of a neighbouring line the window
must reach for the line to be subtracted, so that its fit sees both its sides.
"""

_prominence_max = 0.01
"""
The greatest prominence of a peak of the mean profile in a wing of the line,
as a fraction of the peak of the line,
once the neighbouring line on that side is subtracted.
A more prominent peak is what is left of the neighbouring line,
or another line, so the scene is cut at the lowest point before it.
"""

_percentile_floor = 5
"""The percentile of the mean profile taken as the floor under the lines."""

_velocity_trough_min = 80 * u.km / u.s
"""
The least speed at which the scene is cut short of a neighbouring line
which is not subtracted; the line itself is taken to be within it.
"""

_velocity_trough_max = 160 * u.km / u.s
"""
The greatest speed at which the scene is cut short of a neighbouring line
which is not found, where the neighbouring lines begin.
"""

_size_trough = 3
"""The number of columns over which the mean profile is smoothed to find its trough."""


def _gaussian(
    velocity: np.ndarray,
    center: float,
    sigma: float,
) -> np.ndarray:
    """
    Return a Gaussian of unit height.

    Parameters
    ----------
    velocity
        The velocities at which to evaluate the Gaussian.
    center
        The velocity of its peak.
    sigma
        Its standard deviation.
    """
    return np.exp(-np.square((velocity - center) / sigma) / 2)


def _neighbour(
    profile: np.ndarray,
    velocity: np.ndarray,
    velocity_expected: float,
) -> None | tuple[float, float, float]:
    """
    Fit a Gaussian on a sloping floor to a neighbouring line of the mean profile.

    Return the center, the standard deviation, and the height of the Gaussian,
    or :obj:`None` if no neighbouring line is found where it is expected.

    Parameters
    ----------
    profile
        The mean profile of the observation,
        not a number (NaN) in the columns no exposure observed.
    velocity
        The velocity of the middle of each column, in :data:`_unit_velocity`.
    velocity_expected
        Where the neighbouring line is expected, in :data:`_unit_velocity`.
    """
    shift_max = _shift_neighbour_max.to_value(_unit_velocity)
    width = _width_neighbour.to_value(_unit_velocity)

    observed = np.isfinite(profile)
    search = observed & (np.abs(velocity - velocity_expected) <= shift_max)
    if not np.any(search):
        return None
    index = np.flatnonzero(search)[np.argmax(profile[search])]
    guess = velocity[index]
    near = observed & (np.abs(velocity - guess) <= width)
    if np.count_nonzero(near) < _num_column_min:
        return None
    floor = np.percentile(profile[observed], _percentile_floor)

    def model(
        v: np.ndarray,
        height: float,
        center: float,
        sigma: float,
        level: float,
        slope: float,
    ) -> np.ndarray:
        """
        Return a Gaussian on a sloping floor.

        Parameters
        ----------
        v
            The velocities at which to evaluate the model.
        height
            The height of the Gaussian.
        center
            The velocity of the peak of the Gaussian.
        sigma
            The standard deviation of the Gaussian.
        level
            The floor at the velocity of the guess.
        slope
            The slope of the floor.
        """
        return height * _gaussian(v, center, sigma) + level + slope * (v - guess)

    try:
        with warnings.catch_warnings():
            # the covariance is not used
            warnings.simplefilter("ignore", scipy.optimize.OptimizeWarning)
            p, _ = scipy.optimize.curve_fit(
                f=model,
                xdata=velocity[near],
                ydata=profile[near],
                p0=(
                    profile[index] - floor,
                    guess,
                    _sigma_neighbour_guess.to_value(_unit_velocity),
                    floor,
                    0,
                ),
                maxfev=5000,
            )
    except RuntimeError:
        return None

    height = float(p[0])
    center = float(p[1])
    sigma = abs(float(p[2]))
    if height <= 0:
        return None
    if not (
        _sigma_neighbour_min.to_value(_unit_velocity)
        <= sigma
        <= _sigma_neighbour_max.to_value(_unit_velocity)
    ):
        return None
    if abs(center - velocity_expected) > shift_max:
        return None
    return center, sigma, height


def _amplitude(
    value: np.ndarray,
    velocity: np.ndarray,
    center: float,
    sigma: float,
) -> np.ndarray:
    """
    Fit the height of a neighbouring line in each pixel.

    In each pixel, a constant, a slope, and the Gaussian of the neighbouring
    line, whose center and width are those of the mean profile,
    are fitted by least squares to the columns near the line which were
    observed.
    The height is zero where it would be negative,
    or where too few columns were observed.

    Parameters
    ----------
    value
        The radiance of each pixel, with the columns along the last axis.
    velocity
        The velocity of the middle of each column, in :data:`_unit_velocity`.
    center
        The center of the neighbouring line, in :data:`_unit_velocity`.
    sigma
        The standard deviation of the neighbouring line,
        in :data:`_unit_velocity`.
    """
    width = _width_neighbour.to_value(_unit_velocity)
    columns = np.abs(velocity - center) <= width
    v = velocity[columns]
    basis = np.stack(
        arrays=[np.ones_like(v), (v - center) / width, _gaussian(v, center, sigma)],
        axis=~0,
    )
    data = value[..., columns]
    weight = np.isfinite(data)
    data = np.where(weight, data, 0)
    normal = np.einsum("...k,ki,kj->...ij", weight, basis, basis)
    right = np.einsum("...k,ki->...i", data, basis)
    solution = (np.linalg.pinv(normal) @ right[..., np.newaxis])[..., 0]
    amplitude = solution[..., ~0]
    enough = np.count_nonzero(weight, axis=~0) >= _num_column_min
    return np.where(enough, np.maximum(amplitude, 0), 0)


def _smooth(profile: np.ndarray) -> np.ndarray:
    """
    Return the mean of the observed columns of a profile near each column.

    The columns no exposure observed, and those beyond the ends of the window,
    are left out of the means, so that the ends do not look like troughs.

    Parameters
    ----------
    profile
        A mean profile, not a number (NaN) in the columns no exposure observed.
    """
    observed = np.isfinite(profile)
    kernel = np.ones(_size_trough)
    total = np.convolve(np.where(observed, profile, 0), kernel, mode="same")
    count = np.convolve(observed, kernel, mode="same")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.where(observed, total / count, np.nan)


def _level(
    value: np.ndarray,
    columns: np.ndarray,
    size: int,
) -> np.ndarray:
    """
    Return the continuum of each pixel, from the mean of some columns.

    The mean is averaged over the neighbouring pixels.
    Pixels which observed none of the columns have no continuum.

    Parameters
    ----------
    value
        The radiance of each pixel, with the columns along the last axis.
    columns
        Which columns to average.
    size
        The number of pixels, in each direction, over which to average.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        level = np.nanmean(value[..., columns], axis=~0)
    level = np.nan_to_num(level)
    return scipy.ndimage.uniform_filter(level, size=size)


def scene_sumer_observation(
    observation: soho.sumer.SpectrographObservation,
    velocity_max: None | u.Quantity = None,
    num_continuum: None | int = 3,
    size_continuum: int = 5,
    velocities_neighbour: None | u.Quantity = [-175, +215] * u.km / u.s,
    height_neighbour_max: float = 0.15,
    fill: None | u.Quantity = None,
) -> soho.sumer.SpectrographObservation:
    r"""
    Create a synthetic solar scene from the spectra of SUMER.

    The Solar Ultraviolet Measurements of Emitted Radiation (SUMER)
    spectrograph observed the lines ESIS does, such as O V 629.7 Angstroms,
    so its spectra are used as they are, in their own radiances,
    without the scaling which :func:`~esis.data.synth.scene_iris` needs.

    The wings of the line reach into those of its neighbouring lines.
    Where the spectral window reaches two standard deviations past a
    neighbouring line and that line is weaker than `height_neighbour_max`,
    the line is subtracted from each pixel and the scene is kept out to the
    edge of the window,
    unless the mean profile still has a peak in that wing,
    in which case the scene is cut at the lowest point before the peak.
    Otherwise, the scene is cut short of the neighbouring line,
    where the mean profile is lowest between the two lines.
    The columns at each end of the window which are only partly lit,
    each dimmer than half the column inside it in the mean profile,
    up to three, are left out.

    The scene reaches as far toward the blue as toward the red,
    as far as the shorter side allows,
    so that the wings it keeps on one side are not missing on the other.
    The continuum under the line is subtracted,
    and the voxels SUMER did not observe, which are not a number (NaN),
    are filled if asked to.

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
        The columns whose middles are beyond it are left out,
        besides those left out because of the neighbouring lines.
        If :obj:`None`, the scene is as wide as the neighbouring lines allow.
    num_continuum
        The number of columns from which the continuum is measured:
        those, anywhere in the window, where the mean profile of the
        observation is lowest, before the neighbouring lines are subtracted.
        The continuum of each pixel is the mean of those columns,
        averaged over `size_continuum` pixels of the raster in each
        direction, and is subtracted from every column.
        If :obj:`None`, the continuum is not subtracted.
    size_continuum
        The number of pixels of the raster, in each direction,
        over which the continuum is averaged.
    velocities_neighbour
        Where the neighbouring lines are expected,
        at most one on each side of the line.
        A neighbouring line is fitted to the mean profile of the observation
        within 30 km/s of where it is expected.
        The default is for O V 629.7 Angstroms in the second order of the
        grating of SUMER, whose neighbours are Si II 1260.4 Angstroms,
        in the first order, about 215 km/s to the red,
        and another line about 175 km/s to the blue.
        If :obj:`None`, no neighbouring line is subtracted,
        and the scene is cut only at `velocity_max` and the ends of the window.
    height_neighbour_max
        The height, as a fraction of the peak of the line, above which a
        neighbouring line is not subtracted.
        A strong neighbouring line would leave a residual larger than the wing
        of the line under it.
        Both heights are measured on the mean profile,
        above its fifth percentile.
    fill
        The radiance of the voxels which SUMER did not observe.
        If :obj:`None`, they are left not a number (NaN),
        which an instrument cannot image,
        so give a value before imaging the scene.

    Raises
    ------
    ValueError
        If two neighbouring lines are expected on the same side of the line.
    ValueError
        If no column of the observation is kept.
    """
    axis_x = observation.axis_detector_x
    axis_y = observation.axis_detector_y
    axis_velocity = observation.axis_wavelength
    axes = (axis_x, axis_y, axis_velocity)

    velocity = observation.inputs.velocity
    middle = typing.cast(na.ScalarArray, velocity.cell_centers(axis_velocity))
    middle = typing.cast(u.Quantity, middle.ndarray_aligned((axis_velocity,)))
    middle = np.asarray(middle.to_value(_unit_velocity), dtype=float)
    speed = np.abs(middle)
    width_column = np.abs(np.diff(middle)).mean()

    outputs = typing.cast(na.ScalarArray, observation.outputs)
    radiance = typing.cast(u.Quantity, outputs.ndarray_aligned(axes)).to(_unit_scene)
    value = np.asarray(radiance.value, dtype=float)

    with warnings.catch_warnings():
        # the columns which no exposure observed
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmean(value, axis=(0, 1))
    observed = np.isfinite(profile)

    # the columns at each end of the window which are only partly lit
    # are left out of the fits, the continuum, and the scene
    lit = observed.copy()
    for sign in (-1, +1):
        side = np.flatnonzero(observed & (sign * middle > 0))
        side = side[np.argsort(sign * middle[side])]
        for k in range(min(_num_column_edge_max, side.size - 1)):
            end = side[side.size - 1 - k]
            inside = side[side.size - 2 - k]
            if not profile[end] < _ratio_edge * profile[inside]:
                break
            lit[end] = False
    profile_lit = np.where(lit, profile, np.nan)

    # the speed of the outermost lit column on each side, if any,
    # and how far each side of the scene reaches: no further than the window
    outermost: dict[int, float] = dict()
    limit: dict[int, float] = dict()
    for sign in (-1, +1):
        reach = lit & (sign * middle > 0)
        limit[sign] = 0.0
        if np.any(reach):
            outermost[sign] = float((sign * middle[reach]).max())
            limit[sign] = outermost[sign] + width_column / 2
    subtracted: set[int] = set()

    if velocities_neighbour is not None:
        expected = np.atleast_1d(velocities_neighbour.to_value(_unit_velocity))
        signs = np.sign(expected)
        if np.any(signs == 0) or len(set(signs)) < len(signs):
            raise ValueError(
                f"At most one neighbouring line may be expected on each side "
                f"of the line, but {velocities_neighbour} were given."
            )

        line = lit & (speed < _velocity_trough_min.to_value(_unit_velocity))
        floor = np.percentile(profile[lit], _percentile_floor)
        # without the line, no neighbouring line is weak enough to subtract
        peak = profile[line].max() - floor if np.any(line) else 0

        smooth = _smooth(profile_lit)
        trough_min = _velocity_trough_min.to_value(_unit_velocity)

        for velocity_expected in expected:
            sign = int(np.sign(velocity_expected))
            s = sign * middle
            if sign not in outermost:
                continue
            outer = outermost[sign]

            # a neighbouring line is looked for only if the window reaches
            # where it is expected, so that the drop of the window at its end
            # is not taken for the line
            found = None
            if outer >= abs(velocity_expected):
                found = _neighbour(profile_lit, middle, velocity_expected)
            if found is not None:
                center, sigma, height = found
                reach_min = sign * center + _num_sigma_covered * sigma
                covered = outer >= reach_min
                if covered and height < height_neighbour_max * peak:
                    amplitude = _amplitude(
                        value=np.where(lit, value, np.nan),
                        velocity=middle,
                        center=center,
                        sigma=sigma,
                    )
                    gaussian = _gaussian(middle, center, sigma)
                    value = value - amplitude[..., np.newaxis] * gaussian
                    limit[sign] = outer + width_column / 2
                    subtracted.add(sign)
                    continue
                stop = sign * center - 2 * sigma
            else:
                trough_max = _velocity_trough_max.to_value(_unit_velocity)
                stop = min(outer, trough_max)

            between = np.isfinite(smooth) & (s >= trough_min) & (s <= stop)
            if np.any(between):
                trough = s[between][np.argmin(smooth[between])]
            else:
                # the window ends, or the neighbouring line begins,
                # before the least speed at which the scene is cut
                trough = min(outer, trough_min)
            limit[sign] = float(trough)

        # a peak in a wing of the mean profile, once the neighbouring line is
        # subtracted, is what is left of it, or another line;
        # a continuum which only rises toward the edge of the window is not.
        # The mean profile is not smoothed, which would flatten narrow peaks;
        # it averages so many pixels that its noise is far below the peaks.
        if subtracted:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                after = np.nanmean(value, axis=(0, 1))
            for sign in sorted(subtracted):
                s = sign * middle
                wing = lit & np.isfinite(after)
                wing = wing & (s >= trough_min) & (s <= limit[sign])
                wing = np.flatnonzero(wing)
                wing = wing[np.argsort(s[wing])]
                peaks, _ = scipy.signal.find_peaks(
                    x=after[wing],
                    prominence=_prominence_max * peak,
                )
                if peaks.size:
                    inner = wing[: peaks[0] + 1]
                    index = inner[np.argmin(after[inner])]
                    limit[sign] = float(s[index])

    # as far on both sides, so that a wing kept on one side is not missing
    # on the other
    kept = speed <= min(limit.values())
    if velocity_max is not None:
        kept = kept & (speed <= velocity_max.to_value(_unit_velocity))

    if num_continuum is not None:
        darkest = np.argsort(np.where(lit, profile, np.inf))[:num_continuum]
        columns = np.zeros_like(lit)
        columns[darkest] = True
        columns = columns & lit
        continuum = _level(value, columns, size_continuum)
        value = value - continuum[..., np.newaxis]

    if fill is not None:
        value = np.where(np.isfinite(value), value, fill.to_value(_unit_scene))

    index = np.flatnonzero(kept)
    if not index.size:
        raise ValueError(
            f"No column of the observation is kept, "
            f"with {velocity_max=} and {velocities_neighbour=}."
        )
    cut = {axis_velocity: slice(int(index[0]), int(index[~0]) + 1)}

    scene = typing.cast(soho.sumer.SpectrographObservation, observation[cut])
    outputs = na.ScalarArray(value[..., cut[axis_velocity]] << _unit_scene, axes)
    return dataclasses.replace(scene, outputs=outputs)
