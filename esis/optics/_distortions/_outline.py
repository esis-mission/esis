"""
The outline of the windows: where the field stop's image crosses the frame.

The edge pixels are a fraction of a percent of a window, so the merit
cannot tell a moved grating from a moved sky.  The edges can: they are the
image of the field stop through the optics alone, and do not move when the
pointing does.  This module measures where the edges of each window cross
the rows and columns of a frame, predicts the same crossings from a
linearized channel, and fits the terms that place the windows to the
difference.
"""

from __future__ import annotations
import dataclasses
import numpy as np
import scipy.optimize
import scipy.special
import scipy.ndimage
import astropy.units as u
import astropy.table
import named_arrays as na
import esis
from . import _fit

__all__ = [
    "SIDES",
    "measure_edges",
    "predict_edges",
    "outline_residual",
    "width_residual",
    "fit_distortion_outline",
]

SIDES = ("left", "right", "top", "bottom")
"""
The four sides of a window's outline.

The left and right sides are where the outline crosses a row of the frame,
the top and bottom where it crosses a column.
"""


_INSIDE = (1, -1, 1, -1)
"""
Which way the inside of the window lies across each side of :data:`SIDES`.

``1`` toward larger indices along the row (the left side) or the column
(the top), ``-1`` toward smaller (the right side and the bottom).
"""


def _basis(distance: np.ndarray, sigma: float) -> np.ndarray:
    """
    Evaluate the terms of the model of a profile across an edge.

    The terms are a constant, a unit step at the edge, and a ramp that
    starts at the edge and rises by one per pixel inward, the step and the
    ramp blurred by a Gaussian of standard deviation `sigma`, so that the
    scene just inside the edge may brighten or darken across it.

    Parameters
    ----------
    distance
        The signed distance of every sample from the edge, in pixels,
        positive inside the window.
    sigma
        The standard deviation of the blur, in pixels.

    Returns
    -------
    The three terms, stacked along a new last axis.
    """
    z = distance / sigma
    step = (1 + scipy.special.erf(z / np.sqrt(2))) / 2
    ramp = distance * step + sigma * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
    return np.stack([np.ones_like(distance), step, ramp], axis=-1)


def _solve(
    basis: np.ndarray,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit the terms of every profile by linear least squares.

    Parameters
    ----------
    basis
        The terms of every profile, see :func:`_basis`, indexed
        ``[profile, sample, term]``.
    data
        The profiles, indexed ``[profile, sample]``.

    Returns
    -------
    The coefficients of the terms, indexed ``[profile, term]``, and the
    residuals, indexed like `data`.
    """
    normal = np.einsum("jki,jkl->jil", basis, basis)
    normal = normal + 1e-9 * np.eye(basis.shape[-1])
    projection = np.einsum("jki,jk->ji", basis, data)
    coefficients = np.linalg.solve(normal, projection[..., np.newaxis])[..., 0]
    residual = data - np.einsum("jki,ji->jk", basis, coefficients)
    return coefficients, residual


@dataclasses.dataclass
class _SideFit:
    """One side of a window, fit as a single blurred straight edge."""

    position: np.ndarray
    """The crossing of the edge at every row or column, in pixels."""

    error: np.ndarray
    """The formal error of every crossing, in pixels."""

    kept: np.ndarray
    """Whether each row or column was kept by the clip."""

    width: float
    """The standard deviation of the blur of the edge, normal to it, in pixels."""

    coefficients: np.ndarray
    """
    The background, step and ramp of every row or column, see
    :func:`_basis`, :obj:`numpy.nan` where the profile left the frame.
    """


def _fit_side(
    frame: np.ndarray,
    index: np.ndarray,
    guess: np.ndarray,
    side: int,
    half: int = 30,
    band: int = 6,
    clip: float = 3.0,
    minimum: int = 10,
    num_iteration: int = 4,
) -> None | _SideFit:
    """
    Fit one side of a window as a single blurred straight edge.

    Along every row (the left and right sides) or column (the top and
    bottom) that crosses the side, within `band` pixels of the edge, the
    frame is modelled with the terms of :func:`_basis`: a background, the
    step of the field stop and a ramp of the scene inside it, blurred.  The
    position and slope of the edge and the width of its blur are shared by
    the whole side; the brightness of each row or column is its own, solved
    for exactly at every trial edge.  Bright structure next to the edge in
    a few rows or columns cannot pull a straight edge that hundreds share,
    and the ramp keeps the scene's brightening inward from widening it.

    The edge is first found within `half` pixels of the guesses, as the
    steepest rise of the median profile across them.  Then, until the rows
    or columns left out stop changing, those whose residual from the edge
    is more than `clip` times the median are left out and the edge is refit
    to the rest.

    Parameters
    ----------
    frame
        The frame, indexed ``[y, x]``.
    index
        The rows or columns that cross the side.
    guess
        The predicted crossing of each, in pixels, in the coordinates of
        :func:`predict_edges`, where pixel ``i`` spans ``[i, i + 1)``.
    side
        The index of the side in :data:`SIDES`.
    half
        How far from the guesses the edge is searched for, in pixels.
    band
        Half the length of the profile fit across the edge, in pixels.
    clip
        The residual, as a multiple of the median over the rows or columns,
        beyond which a row or column is left out.
    minimum
        The fewest rows or columns a side is fit with.
    num_iteration
        The most refits.

    Returns
    -------
    The fit, or :obj:`None` if fewer than `minimum` rows or columns cross
    the side inside the frame or are left by the clip, or if no edge is
    found: a step less than a fifth of the brightness just inside it, or
    one softer than half the band.
    """
    index = np.asarray(index, dtype=int)
    guess = np.asarray(guess, dtype=float)
    if index.size < minimum:
        return None
    image = frame if side < 2 else frame.T
    inside = _INSIDE[side]
    num = image.shape[1]
    along = index + 1 / 2
    center = along.mean()
    slope, offset = np.polyfit(along - center, guess, deg=1)

    def profiles(
        parameters: np.ndarray,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read the profiles within `width` pixels of an edge, and which are whole."""
        line = parameters[0] + parameters[1] * (along - center)
        start = np.round(line - 1 / 2).astype(int)
        samples = start[:, np.newaxis] + np.arange(-width, width + 1)
        whole = (samples.min(axis=1) >= 0) & (samples.max(axis=1) < num)
        samples = np.clip(samples, 0, num - 1)
        data = image[index[:, np.newaxis], samples]
        whole &= np.all(np.isfinite(data), axis=1)
        return samples + 1 / 2, np.where(whole[:, np.newaxis], data, 0), whole

    def fit(
        parameters: np.ndarray,
        across: np.ndarray,
        data: np.ndarray,
        which: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit the profiles `which` for an edge: their coefficients and residuals."""
        position, slope, sigma = parameters
        line = position + slope * (along[which] - center)
        distance = across[which] - line[:, np.newaxis]
        distance = inside * distance / np.sqrt(1 + slope**2)
        return _solve(_basis(distance, sigma), data[which])

    # find the edge near the guesses
    _, data, whole = profiles(np.array([offset, slope]), half)
    if whole.sum() < minimum:
        return None
    data = data[whole]
    scale = np.mean(np.abs(data), axis=1, keepdims=True)
    median = np.median(data / np.where(scale > 0, scale, 1), axis=0)
    rise = inside * np.gradient(scipy.ndimage.uniform_filter1d(median, 3))
    shift = np.arange(-half, half + 1)[band:-band]
    parameters = np.array([offset + shift[np.argmax(rise[band:-band])], slope, 1])

    # leave out the rows or columns that do not fit, then refit the rest
    kept = whole
    result = None
    for _ in range(num_iteration):
        across, data, whole = profiles(parameters, band)
        rms = np.sqrt(np.mean(fit(parameters, across, data, whole)[1] ** 2, axis=1))
        clipped = np.zeros_like(whole)
        clipped[whole] = rms <= clip * np.median(rms[kept[whole]])
        if result is not None and np.array_equal(clipped, kept):
            break
        kept = clipped
        if kept.sum() < minimum:
            return None
        result = scipy.optimize.least_squares(
            lambda p, *args: fit(p, *args)[1].ravel(),
            parameters,
            bounds=([-np.inf, -np.inf, 0.1], [np.inf, np.inf, band]),
            args=(across, data, kept),
        )
        parameters = result.x

    position, slope, sigma = parameters
    across, data, whole = profiles(parameters, band)
    coefficients = np.full((index.size, 3), np.nan)
    coefficients[whole] = fit(parameters, across, data, whole)[0]
    background, step, _ = np.median(coefficients[kept & whole], axis=0)
    if sigma > band / 2 or step < 0.2 * (background + step):
        return None
    jacobian = result.jac
    num_free = jacobian.shape[1] + coefficients.shape[1] * kept.sum()
    variance = np.sum(result.fun**2) / max(result.fun.size - num_free, 1)
    covariance = np.linalg.pinv(jacobian.T @ jacobian) * variance
    d = along - center
    error = covariance[0, 0] + 2 * d * covariance[0, 1] + d**2 * covariance[1, 1]
    return _SideFit(
        position=position + slope * d,
        error=np.sqrt(error),
        kept=kept,
        width=float(sigma),
        coefficients=coefficients,
    )


def _polygon(
    footprint: na.AbstractCartesian2dVectorArray,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the x and y of a footprint's vertices as plain arrays, in pixels."""
    x = np.asarray(
        na.value(na.as_named_array(footprint.x).to(u.pix)).ndarray, dtype=float
    )
    y = np.asarray(
        na.value(na.as_named_array(footprint.y).to(u.pix)).ndarray, dtype=float
    )
    return x.ravel(), y.ravel()


def _crossings(
    px: np.ndarray,
    py: np.ndarray,
    y: float,
) -> tuple[float, float]:
    """Find the smallest and largest x at which a closed polygon crosses the row y."""
    px2, py2 = np.roll(px, -1), np.roll(py, -1)
    cross = (py <= y) != (py2 <= y)
    if cross.sum() < 2:
        return np.nan, np.nan
    t = (y - py[cross]) / (py2[cross] - py[cross])
    xs = px[cross] + t * (px2[cross] - px[cross])
    return float(xs.min()), float(xs.max())


def _side_crossings(
    px: np.ndarray,
    py: np.ndarray,
    shape: tuple[int, int],
    fraction: float = 0.35,
    skip: int = 25,
    stride: int = 2,
    margin: int = 40,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Choose the rows and columns that cross each side of an outline.

    Parameters
    ----------
    px
        The x of the outline's vertices, in pixels.
    py
        The y of the outline's vertices, in pixels.
    shape
        The shape of the frame, ``(num_y, num_x)``.
    fraction
        See :func:`measure_edges`.
    skip
        See :func:`measure_edges`.
    stride
        See :func:`measure_edges`.
    margin
        See :func:`measure_edges`.

    Returns
    -------
    For each side of :data:`SIDES`, by its index, the rows or columns that
    cross it and the outline's crossing of each, in pixels.
    """
    num_y, num_x = shape
    xc, yc = px.mean(), py.mean()
    ry = 0.5 * (py.max() - py.min())
    rx = 0.5 * (px.max() - px.min())
    crossings = {side: ([], []) for side in range(len(SIDES))}
    for dy in range(-int(fraction * ry), int(fraction * ry)):
        if abs(dy) < skip:
            continue
        y = int(round(yc + dy))
        if y < 0 or y >= num_y:  # pragma: nocover
            continue
        for side, guess in zip((0, 1), _crossings(px, py, y)):
            if np.isfinite(guess) and margin <= guess <= num_x - margin:
                crossings[side][0].append(y)
                crossings[side][1].append(guess)
    for dx in range(-int(fraction * rx), int(fraction * rx), stride):
        x = int(round(xc + dx))
        if x < 0 or x >= num_x:  # pragma: nocover
            continue
        for side, guess in zip((2, 3), _crossings(py, px, x)):
            if np.isfinite(guess) and margin <= guess <= num_y - margin:
                crossings[side][0].append(x)
                crossings[side][1].append(guess)
    return {
        side: (np.array(index, dtype=int), np.array(guess, dtype=float))
        for side, (index, guess) in crossings.items()
    }


def measure_edges(
    frame: np.ndarray,
    footprints: list[na.AbstractCartesian2dVectorArray],
    axis_field: tuple[str, str] = ("detector_x", "detector_y"),
    fraction: float = 0.35,
    skip: int = 25,
    stride: int = 2,
    half: int = 30,
    band: int = 6,
    clip: float = 3.0,
    margin: int = 40,
) -> astropy.table.QTable:
    """
    Measure where the edges of every window cross the rows and columns of a frame.

    Each side of each predicted outline is fit as one blurred straight
    edge, see :func:`_fit_side`, along the rows that cross the left and
    right sides and the columns that cross the top and bottom, within a
    band about the window's centre so that the corners of the octagon,
    where two sides meet, are left out.

    Parameters
    ----------
    frame
        The frame, indexed ``[y, x]`` in pixels.
    footprints
        The predicted outline of each window, one per spectral line, in
        pixels, see :meth:`optika.systems.AbstractLinearSystem.footprint`.
    axis_field
        The names of the sensor axes, for the metadata only.
    fraction
        The half-height of the band of rows (columns) about the window's
        centre, as a fraction of the window's half-extent.
    skip
        Rows within this distance of the window's centre are left out of
        the left and right sides.
    stride
        Every ``stride``-th column is fit on the top and bottom sides.
    half
        How far from the predicted outline each edge is searched for, in
        pixels.
    band
        Half the length of the profile fit across each edge, in pixels.
    clip
        Rows and columns whose residual is more than this multiple of the
        side's median are left out.
    margin
        Crossings closer than this to the edge of the frame are left out.

    Returns
    -------
    A table with a row per crossing: ``line``, ``side``, ``index`` (the row
    for the left and right sides, the column for the top and bottom),
    ``position`` and ``error``, where the side's straight edge crosses that
    row or column, and ``width``, the standard deviation of the side's
    blur normal to the edge, all in pixels.  The width includes the
    pixels' own, 1/sqrt(12) px in quadrature, and on the left and
    right sides the width of the spectral line.  Only the rows and columns
    the clip kept are listed.
    """
    frame = np.asarray(frame, dtype=float)
    rows = []
    for line, footprint in enumerate(footprints):
        px, py = _polygon(footprint)
        if not np.all(np.isfinite(px)):  # pragma: nocover
            continue
        sides = _side_crossings(px, py, frame.shape, fraction, skip, stride, margin)
        for side, (index, guess) in sides.items():
            fit = _fit_side(frame, index, guess, side, half=half, band=band, clip=clip)
            if fit is None:
                continue
            for k in np.flatnonzero(fit.kept):
                rows.append(
                    (line, side, index[k], fit.position[k], fit.error[k], fit.width)
                )
    result = astropy.table.QTable(
        rows=rows or None,
        names=("line", "side", "index", "position", "error", "width"),
        dtype=(int, int, int, float, float, float),
    )
    for name in ("position", "error", "width"):
        result[name].unit = u.pix
    result.meta["axis_field"] = list(axis_field)
    return result


def predict_edges(
    footprints: list[na.AbstractCartesian2dVectorArray],
    edges: astropy.table.QTable,
) -> np.ndarray:
    """
    Predict the crossing of every measured edge from the model's outlines.

    Parameters
    ----------
    footprints
        The predicted outline of each window, one per spectral line, in
        pixels.
    edges
        Measured crossings, see :func:`measure_edges`.

    Returns
    -------
    The model position of every row of `edges`, in pixels,
    :obj:`numpy.nan` where the outline does not cross that row or column.
    """
    polygons = [_polygon(f) for f in footprints]
    result = np.full(len(edges), np.nan)
    line = np.asarray(edges["line"])
    side = np.asarray(edges["side"])
    index = np.asarray(edges["index"])
    for k in range(len(edges)):
        px, py = polygons[line[k]]
        # the row or column of index i is centred at i + 1/2
        if side[k] < 2:
            left, right = _crossings(px, py, index[k] + 0.5)
            result[k] = right if side[k] == 1 else left
        else:
            top, bottom = _crossings(py, px, index[k] + 0.5)
            result[k] = bottom if side[k] == 3 else top
    return result


def outline_residual(
    footprints: list[na.AbstractCartesian2dVectorArray],
    edges: astropy.table.QTable,
    clip: float = 3.0,
) -> float:
    """
    Compute the root-mean-square distance between the measured and modelled edges.

    Every distance is clipped at `clip` pixels first, so that a handful of
    bad edge fits cannot steer the window.

    Parameters
    ----------
    footprints
        The predicted outline of each window, in pixels.
    edges
        Measured crossings, see :func:`measure_edges`.
    clip
        The distance, in pixels, beyond which a point counts as no worse.
    """
    residual = predict_edges(footprints, edges) - edges["position"].to_value(u.pix)
    finite = np.isfinite(residual)
    if finite.mean() < 0.9:
        return float(clip)
    clipped = np.minimum(np.abs(residual[finite]), clip)
    return float(np.sqrt(np.mean(clipped**2)))


def width_residual(
    footprints: list[na.AbstractCartesian2dVectorArray],
    edges: astropy.table.QTable,
    clip: float = 3.0,
) -> float:
    """
    Compute the root-mean-square error of the width of every window.

    The width across a row is the right crossing minus the left, and down
    a column the bottom minus the top, measured and modelled at the same
    rows and columns, so the residual is blind to where a window sits and
    sees only how large it is.  Every error is clipped at `clip` pixels.

    Parameters
    ----------
    footprints
        The predicted outline of each window, in pixels.
    edges
        Measured crossings, see :func:`measure_edges`.
    clip
        The error, in pixels, beyond which a width counts as no worse.
    """
    model = predict_edges(footprints, edges)
    measured = edges["position"].to_value(u.pix)
    line = np.asarray(edges["line"])
    side = np.asarray(edges["side"])
    index = np.asarray(edges["index"])
    errors = []
    for low, high in ((0, 1), (2, 3)):
        a = {(line[k], index[k]): k for k in np.where(side == low)[0]}
        b = {(line[k], index[k]): k for k in np.where(side == high)[0]}
        for key in a.keys() & b.keys():
            i, j = a[key], b[key]
            width_model = model[j] - model[i]
            width_measured = measured[j] - measured[i]
            if np.isfinite(width_model):
                errors.append(width_model - width_measured)
    if len(errors) < 10:
        return float(clip)
    clipped = np.minimum(np.abs(np.asarray(errors)), clip)
    return float(np.sqrt(np.mean(clipped**2)))


def fit_distortion_outline(
    merit,
    edges: astropy.table.QTable,
    window: tuple[str, ...],
    sky: tuple[str, ...],
    bounds: tuple[esis.optics.DistortionParameters, esis.optics.DistortionParameters],
    num_pass: int = 2,
    clip: float = 3.0,
    maxfev_window: int = 400,
    maxfev_sky: int = 1500,
    log=None,
) -> esis.optics.DistortionParameters:
    """
    Place the windows on the measured edges, then re-polish the sky terms.

    Alternates a Nelder-Mead fit of the `window` terms to the edge residual
    with a polish of the `sky` terms against the merit, `num_pass` times.
    The window terms are those that move the image of the field stop on
    the sensor but not the sky's registration within it, which the merit
    cannot see; the sky terms are the rest.

    Parameters
    ----------
    merit
        The :class:`esis.optics.LinearMerit` of the channel, whose
        ``parameters`` are the starting point.
    edges
        The measured crossings of this channel's windows, see
        :func:`measure_edges`.
    window
        The names of the fields fit to the edges.
    sky
        The names of the fields polished against the merit.
    bounds
        The lower and upper bounds of every field.
    num_pass
        The number of alternations.
    clip
        The clip of :func:`outline_residual`.
    maxfev_window
        The evaluation budget of every window fit.
    maxfev_sky
        The evaluation budget of every sky polish.
    log
        A callable that records progress.
    """
    parameters = merit.parameters
    names = [f.name for f in dataclasses.fields(parameters)]
    index_window = np.array([names.index(n) for n in window])
    index_sky = np.array([names.index(n) for n in sky])
    lower, upper = bounds
    lb, ub = na.pack(lower).ndarray, na.pack(upper).ndarray
    x = na.pack(parameters).ndarray.copy()
    wavelength = merit.instrument.wavelength
    num_lines = na.shape(wavelength).get("wavelength", 1)

    def footprints(x_full: np.ndarray) -> list[na.AbstractCartesian2dVectorArray]:
        p = na.unpack(x_full, parameters)
        linear = merit.linearize(p.to_instrument(merit.instrument))
        return [
            linear.footprint(
                wavelength[dict(wavelength=i)] if num_lines > 1 else wavelength
            )
            for i in range(num_lines)
        ]

    def residual(y: np.ndarray) -> float:
        x_trial = x.copy()
        x_trial[index_window] = y
        try:
            return outline_residual(footprints(x_trial), edges, clip=clip)
        except (
            ValueError,
            np.linalg.LinAlgError,
            FloatingPointError,
        ):  # pragma: nocover
            return float(clip)

    if log is not None:
        log(
            f"outline: start merit {-merit(x):.4f}, "
            f"edge rms {residual(x[index_window]):.2f} px"
        )
    for it in range(num_pass):
        step = 0.01 * (ub - lb)[index_window]
        simplex = np.vstack(
            [x[index_window]]
            + [x[index_window] + step * e for e in np.eye(len(index_window))]
        )
        fit = scipy.optimize.minimize(
            residual,
            x[index_window],
            method="Nelder-Mead",
            bounds=scipy.optimize.Bounds(lb[index_window], ub[index_window]),
            options=dict(
                initial_simplex=simplex, xatol=1e-3, fatol=1e-3, maxfev=maxfev_window
            ),
        )
        x[index_window] = fit.x
        if log is not None:
            log(
                f"outline pass {it}: edge rms {fit.fun:.2f} px after "
                f"{fit.nfev} evaluations, merit {-merit(x):.4f}"
            )
        objective = _fit._Subset(merit, x.copy(), index_sky)
        y, fun, num = _fit.polish(
            objective,
            x[index_sky],
            lb[index_sky],
            ub[index_sky],
            scale=0.01,
            num_round=1,
            maxfev=maxfev_sky,
        )
        x[index_sky] = y
        if log is not None:
            log(
                f"outline pass {it}: merit {-fun:.4f} after {num} evaluations, "
                f"edge rms {residual(x[index_window]):.2f} px"
            )
    return na.unpack(x, parameters)
