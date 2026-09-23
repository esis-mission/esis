"""
The outline of the windows: where the field stop's image crosses the frame.

The merit compares the interior of every window with the scene, and the
edge pixels are a fraction of a percent of it, so it cannot tell a moved
grating from a moved sky.  The edges can: they are the image of the field
stop through the optics alone, and do not move when the pointing does.
This module measures where the edges of each window cross the rows and
columns of a frame, predicts the same crossings from a linearized channel,
and fits the terms that place the windows to the difference.
"""

from __future__ import annotations
import dataclasses
import warnings
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
    "fit_distortion_outline",
]

SIDES = ("left", "right", "top", "bottom")
"""
The four sides of a window's outline.

The left and right sides are where the outline crosses a row of the frame,
the top and bottom where it crosses a column.
"""


def _erf_step(x, a, b, x0, sigma):
    return a + b * scipy.special.erf((x - x0) / (np.sqrt(2) * sigma))


def _fit_edge(
    profile: np.ndarray,
    guess: float,
    half: int,
) -> tuple[float, float]:
    """
    Fit an error-function step to a profile near a guessed position.

    Parameters
    ----------
    profile
        The intensity along a row or column of the frame.
    guess
        The index along the profile where the edge is expected.
    half
        Half the length of the profile fit, in pixels.

    Returns
    -------
    The position of the step and its formal error, both :obj:`numpy.nan`
    if the fit fails or is not a step.
    """
    lo, hi = int(guess - half), int(guess + half)
    if lo < 2 or hi > profile.size - 2:
        return np.nan, np.nan
    x = np.arange(lo, hi)
    y = profile[lo:hi]
    if not np.all(np.isfinite(y)):
        return np.nan, np.nan
    b0 = (y[-8:].mean() - y[:8].mean()) / 2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, cov = scipy.optimize.curve_fit(
                _erf_step,
                x,
                y,
                p0=[y.mean(), b0, guess, 2.0],
                maxfev=4000,
            )
    except Exception:
        return np.nan, np.nan
    if abs(p[3]) > 12 or abs(p[2] - guess) > half / 2 or abs(p[1]) < 0.2 * abs(p[0]):
        return np.nan, np.nan
    error = np.sqrt(cov[2, 2]) if np.all(np.isfinite(cov)) else np.nan
    return float(p[2]), float(error)


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


def measure_edges(
    frame: np.ndarray,
    footprints: list[na.AbstractCartesian2dVectorArray],
    axis_field: tuple[str, str] = ("detector_x", "detector_y"),
    fraction: float = 0.35,
    skip: int = 25,
    stride: int = 2,
    half: int = 30,
    margin: int = 40,
) -> astropy.table.QTable:
    """
    Measure where the edges of every window cross the rows and columns of a frame.

    Near each predicted outline, an error-function step is fit along the
    rows that cross the left and right sides and along the columns that
    cross the top and bottom, within a band about the window's centre so
    that the corners of the octagon, where two sides meet, are left out.

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
        Rows and columns within this distance of the window's centre are
        left out.
    stride
        Every ``stride``-th column is fit on the top and bottom sides.
    half
        Half the length of every profile fit, in pixels.
    margin
        Crossings closer than this to the edge of the frame are left out.

    Returns
    -------
    A table with a row per crossing: ``line``, ``side``, ``index`` (the row
    for the left and right sides, the column for the top and bottom),
    ``position`` and ``error`` in pixels.
    """
    frame = scipy.ndimage.median_filter(np.asarray(frame, dtype=float), size=(3, 1))
    rows = []
    for line, footprint in enumerate(footprints):
        px, py = _polygon(footprint)
        if not np.all(np.isfinite(px)):
            continue
        xc, yc = px.mean(), py.mean()
        ry = 0.5 * (py.max() - py.min())
        rx = 0.5 * (px.max() - px.min())
        band_y = [
            dy
            for dy in range(-int(fraction * ry), int(fraction * ry))
            if abs(dy) >= skip
        ]
        for dy in band_y:
            y = int(round(yc + dy))
            if y < 0 or y >= frame.shape[0]:
                continue
            left, right = _crossings(px, py, y)
            for side, guess in ((0, left), (1, right)):
                if (
                    not np.isfinite(guess)
                    or guess < margin
                    or guess > frame.shape[1] - margin
                ):
                    continue
                position, error = _fit_edge(frame[y], guess, half)
                if np.isfinite(position) and np.isfinite(error):
                    rows.append((line, side, y, position, error))
        band_x = [dx for dx in range(-int(fraction * rx), int(fraction * rx), stride)]
        for dx in band_x:
            x = int(round(xc + dx))
            if x < 0 or x >= frame.shape[1]:
                continue
            top, bottom = _crossings(py, px, x)
            for side, guess in ((2, top), (3, bottom)):
                if (
                    not np.isfinite(guess)
                    or guess < margin
                    or guess > frame.shape[0] - margin
                ):
                    continue
                position, error = _fit_edge(frame[:, x], guess, half)
                if np.isfinite(position) and np.isfinite(error):
                    rows.append((line, side, x, position, error))
    result = astropy.table.QTable(
        rows=rows or None,
        names=("line", "side", "index", "position", "error"),
        dtype=(int, int, int, float, float),
    )
    result["position"].unit = u.pix
    result["error"].unit = u.pix
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
        if side[k] < 2:
            left, right = _crossings(px, py, index[k])
            result[k] = right if side[k] == 1 else left
        else:
            top, bottom = _crossings(py, px, index[k])
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
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
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
