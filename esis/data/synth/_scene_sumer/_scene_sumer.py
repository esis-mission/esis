import typing
import warnings
import dataclasses
import numpy as np
import scipy.ndimage
import scipy.optimize
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

_width_continuum = 30 * u.km / u.s
"""
The width of the strip at the end of a side whose neighbouring line is
subtracted, from which the continuum is measured.
"""

_num_continuum_min = 2
"""
The fewest columns the continuum is measured from,
below which it is measured from the darkest columns of the mean profile.
"""

_num_continuum_darkest = 3
"""The number of darkest columns of the mean profile the continuum falls back to."""


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
    velocity_continuum: None | u.Quantity = 100 * u.km / u.s,
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
    Where the spectral window reaches past a neighbouring line
    and that line is weaker than `height_neighbour_max`,
    the line is subtracted from each pixel and the scene is kept out to the
    edge of the window.
    Otherwise, the scene is cut short of the neighbouring line,
    where the mean profile is lowest between the two lines.
    The continuum under the line is then subtracted,
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
    velocity_continuum
        The least speed of the columns from which the continuum is measured,
        if no neighbouring line was subtracted.
        The continuum of each pixel is the mean of those columns which are
        kept, averaged over `size_continuum` pixels of the raster in each
        direction, and is subtracted from every column.
        If a neighbouring line was subtracted, the continuum is measured
        instead at the end of the window beyond it,
        and if both were, it slopes linearly between the two ends.
        If fewer than two columns are left to measure it from,
        it is measured from the three columns where the mean profile of the
        observation is lowest.
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
        and the scene is cut only at `velocity_max`.
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

    # how far each side of the scene reaches, and the strips at the end of
    # the sides whose neighbouring line is subtracted
    limit: dict[int, float] = {-1: np.inf, +1: np.inf}
    strips: dict[int, np.ndarray] = dict()

    if velocities_neighbour is not None:
        expected = np.atleast_1d(velocities_neighbour.to_value(_unit_velocity))
        signs = np.sign(expected)
        if np.any(signs == 0) or len(set(signs)) < len(signs):
            raise ValueError(
                f"At most one neighbouring line may be expected on each side "
                f"of the line, but {velocities_neighbour} were given."
            )

        line = observed & (speed < _velocity_trough_min.to_value(_unit_velocity))
        floor = np.percentile(profile[observed], _percentile_floor)
        # without the line, no neighbouring line is weak enough to subtract
        peak = profile[line].max() - floor if np.any(line) else 0

        # the mean of the observed columns near each column,
        # so that the ends of the window do not look like troughs
        kernel = np.ones(_size_trough)
        total = np.convolve(np.where(observed, profile, 0), kernel, mode="same")
        count = np.convolve(observed, kernel, mode="same")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            smooth = np.where(observed, total / count, np.nan)

        for velocity_expected in expected:
            sign = int(np.sign(velocity_expected))
            s = sign * middle
            reach = observed & (s > 0)
            if not np.any(reach):
                continue
            outermost = s[reach].max()
            # the outermost column is left out, which may be only partly lit
            edge = outermost - width_column / 2

            # a neighbouring line is looked for only if the window reaches
            # where it is expected, so that the drop of a partly lit column
            # at the end of the window is not taken for the line
            found = None
            if outermost >= abs(velocity_expected):
                found = _neighbour(profile, middle, velocity_expected)
            if found is not None:
                center, sigma, height = found
                covered = outermost >= sign * center + sigma
                if covered and height < height_neighbour_max * peak:
                    amplitude = _amplitude(value, middle, center, sigma)
                    gaussian = _gaussian(middle, center, sigma)
                    value = value - amplitude[..., np.newaxis] * gaussian
                    limit[sign] = edge
                    width = _width_continuum.to_value(_unit_velocity)
                    strips[sign] = observed & (s >= edge - width) & (s <= edge)
                    continue
                stop = sign * center - 2 * sigma
            else:
                trough_max = _velocity_trough_max.to_value(_unit_velocity)
                stop = min(outermost, trough_max)

            trough_min = _velocity_trough_min.to_value(_unit_velocity)
            between = np.isfinite(smooth) & (s >= trough_min) & (s <= stop)
            if np.any(between):
                trough = s[between][np.argmin(smooth[between])]
            else:
                trough = edge
            limit[sign] = min(trough, edge)

    kept = (-limit[-1] <= middle) & (middle <= limit[+1])
    if velocity_max is not None:
        kept = kept & (speed <= velocity_max.to_value(_unit_velocity))

    if velocity_continuum is not None:
        strips = {
            sign: strip
            for sign, strip in strips.items()
            if np.count_nonzero(strip) >= 2
        }
        if len(strips) == 2:
            v_blue = middle[strips[-1]].mean()
            v_red = middle[strips[+1]].mean()
            weight = (middle - v_blue) / (v_red - v_blue)
            level_blue = _level(value, strips[-1], size_continuum)
            level_red = _level(value, strips[+1], size_continuum)
            continuum = (
                level_blue[..., np.newaxis] * (1 - weight)
                + level_red[..., np.newaxis] * weight
            )
            value = value - continuum
        else:
            if strips:
                (columns,) = strips.values()
            else:
                v_min = velocity_continuum.to_value(_unit_velocity)
                columns = kept & (speed >= v_min)
            if np.count_nonzero(columns) < _num_continuum_min:
                darkest = np.argsort(np.where(observed, profile, np.inf))
                columns = np.zeros_like(observed)
                columns[darkest[:_num_continuum_darkest]] = True
                columns = columns & observed
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
