"""
The internal alignment of the channels, solved in the physical parameters.

A fit of each channel against a proxy image of the Sun places the channel to
about a pixel; the channels compared with one another on the sky plane
resolve a tenth of a pixel.  This module measures that disagreement (each
channel's frame sampled on a common sky grid through its own distortion,
tile by tile against an anchor) and solves for the increments of the
physical parameters whose change of the mapping reproduces the measured
shift field.
"""

from __future__ import annotations
from typing import Callable
import dataclasses
import numpy as np
import scipy.ndimage
import astropy.units as u
import named_arrays as na
import optika
import esis

__all__ = [
    "highpass",
    "shift_fft",
    "sky_grid",
    "sensor_coordinates",
    "window_mask",
    "sample_on_sky",
    "measure_shifts",
    "measure_channel_shifts",
    "median_shifts",
    "shift_modes",
    "ShiftField",
    "align_channels",
]


def highpass(a: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """
    Remove the large-scale intensity structure of an image.

    Vignetting and effective-area differences between the channels appear
    as smooth gradients, which a correlation would lock onto instead of the
    solar structure that carries the alignment.

    Parameters
    ----------
    a
        The image to filter.
    sigma
        The standard deviation, in pixels, of the smoothing kernel.
    """
    return a - scipy.ndimage.gaussian_filter(a, sigma=sigma, mode="nearest")


def shift_fft(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Locate the shift of `b` relative to `a` by phase correlation.

    The shift :math:`s` is defined so that `b` shows at :math:`r` what `a`
    shows at :math:`r + s`.  The correlation peak is refined to sub-pixel
    accuracy with a parabola through its immediate neighbors along each axis.

    Parameters
    ----------
    a
        The reference tile.
    b
        The tile whose shift relative to `a` is measured.
    """
    a = highpass(a)
    b = highpass(b)
    a = a - a.mean()
    b = b - b.mean()
    if not (np.any(a) and np.any(b)):
        return np.nan, np.nan

    window = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    correlation = np.fft.ifft2(
        np.fft.fft2(a * window) * np.conj(np.fft.fft2(b * window))
    ).real
    correlation = np.fft.fftshift(correlation)

    index = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    center = np.array(correlation.shape) // 2

    shift = []
    for axis in (0, 1):
        i = index[axis]
        if 1 <= i < correlation.shape[axis] - 1:
            j = list(index)
            j[axis] = i - 1
            left = correlation[tuple(j)]
            j[axis] = i + 1
            right = correlation[tuple(j)]
            denominator = left - 2 * correlation[index] + right
            delta = 0.5 * (left - right) / denominator if denominator else 0.0
        else:  # pragma: nocover
            delta = 0.0
        shift.append(i + delta - center[axis])

    return shift[0], shift[1]


def sky_grid(
    position: na.AbstractCartesian2dVectorArray,
    num: int,
    axis: tuple[str, str] = ("sky_x", "sky_y"),
) -> na.Cartesian2dVectorLinearSpace:
    """
    Build a regular grid on the sky spanning the given positions.

    Parameters
    ----------
    position
        Positions on the sky whose extent the grid should span, usually the
        vertices of the scene.
    num
        The number of samples along each axis.
    axis
        The logical axes of the grid.
    """
    return na.Cartesian2dVectorLinearSpace(
        start=na.Cartesian2dVectorArray(x=position.x.min(), y=position.y.min()),
        stop=na.Cartesian2dVectorArray(x=position.x.max(), y=position.y.max()),
        axis=na.Cartesian2dVectorArray(*axis),
        num=num,
    )


def sensor_coordinates(
    distortion: optika.distortion.AbstractDistortionModel,
    sky: na.AbstractCartesian2dVectorArray,
    wavelength: u.Quantity,
    axis: tuple[str, str] = ("sky_x", "sky_y"),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map a sky grid onto the sensor at one wavelength.

    Parameters
    ----------
    distortion
        The distortion model of the channel.
    sky
        The grid on the sky.
    wavelength
        The wavelength at which to map it.
    axis
        The logical axes of the grid.
    """
    coordinates = na.SpectralPositionalVectorArray(wavelength=wavelength, position=sky)
    sensor = distortion.distort(coordinates).position
    return (
        np.asarray(na.value(sensor.x).ndarray_aligned(axis)),
        np.asarray(na.value(sensor.y).ndarray_aligned(axis)),
    )


def window_mask(
    footprint: na.AbstractCartesian2dVectorArray,
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Mark the pixels of a frame that lie inside one window's outline.

    The outline of the field stop on the sensor at one line, see
    :meth:`optika.systems.LinearSystem.footprint`, bounds the window that
    line illuminates.  Pixels outside it hold other lines' windows or the
    gaps between them, which, read through this line's distortion, land on
    the sky as structure that belongs elsewhere.

    Parameters
    ----------
    footprint
        The vertices of the window's outline, in pixels.
    shape
        The shape of the frame, ``(num_y, num_x)``.
    """
    import matplotlib.path

    vertices = np.stack(
        [
            np.asarray(na.value(footprint.x).ndarray).ravel(),
            np.asarray(na.value(footprint.y).ndarray).ravel(),
        ],
        axis=-1,
    )
    # pixel i spans [i, i + 1) in optika's sensor coordinates, so its
    # centre is at i + 1/2
    num_y, num_x = shape
    x, y = np.meshgrid(np.arange(num_x) + 0.5, np.arange(num_y) + 0.5)
    inside = matplotlib.path.Path(vertices).contains_points(
        np.stack([x.ravel(), y.ravel()], axis=-1)
    )
    return inside.reshape(shape)


def sample_on_sky(
    frame: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Read a frame bilinearly at the given sensor coordinates.

    Points off the sensor read as NaN, not zero: an edge of zeros would be
    mistaken for structure by the correlation.

    The coordinates are optika's, see
    :meth:`optika.sensors.AbstractImagingSensor.pixels`: pixel ``i`` spans
    ``[i, i + 1)`` and its value sits at ``i + 1/2``, so a coordinate on a
    pixel's centre reads that pixel alone.

    Parameters
    ----------
    frame
        The observed image, indexed ``[y, x]``.
    x
        The sensor :math:`x` coordinate of every sky-grid point, in pixels.
    y
        The sensor :math:`y` coordinate of every sky-grid point, in pixels.
    """
    num_y, num_x = frame.shape
    sampled = np.full(x.shape, np.nan, dtype=float)
    # from optika's edge-based coordinates to the index whose value sits there
    x = np.asarray(x) - 0.5
    y = np.asarray(y) - 0.5
    ix = np.floor(x).astype(int)
    iy = np.floor(y).astype(int)
    inside = (ix >= 0) & (ix + 1 < num_x) & (iy >= 0) & (iy + 1 < num_y)
    fx = (x - ix)[inside]
    fy = (y - iy)[inside]
    ix, iy = ix[inside], iy[inside]
    sampled[inside] = (
        frame[iy, ix] * (1 - fx) * (1 - fy)
        + frame[iy, ix + 1] * fx * (1 - fy)
        + frame[iy + 1, ix] * (1 - fx) * fy
        + frame[iy + 1, ix + 1] * fx * fy
    )
    return sampled


@dataclasses.dataclass
class ShiftField:
    """The tile shifts of one channel against the anchor, for one line."""

    wavelength: u.Quantity
    """The wavelength at which the sky images were made."""

    i: np.ndarray
    """The sky-grid index of each tile's center along the first axis."""

    j: np.ndarray
    """The sky-grid index of each tile's center along the second axis."""

    dx: np.ndarray
    """The measured shift of each tile along the first axis, in sky-grid pixels."""

    dy: np.ndarray
    """The measured shift of each tile along the second axis, in sky-grid pixels."""


def measure_shifts(
    images: list[np.ndarray],
    num_tile: int,
    anchor: int,
    fraction_valid: float = 0.95,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """
    Measure the tile-by-tile shift of every channel against the anchor.

    Parameters
    ----------
    images
        The sky-plane image of each channel, as returned by
        :func:`sample_on_sky`.
    num_tile
        The number of tiles along each axis of the sky grid.
    anchor
        The index of the channel the others are measured against.
    fraction_valid
        The fraction of a tile which must be on the detector in both
        channels for its shift to be measured.  Partially valid tiles are
        padded with zeros, whose edges the correlation mistakes for
        structure, so the threshold is deliberately strict.

    Returns
    -------
    For every channel but the anchor, a list of ``(i, j, dx, dy)``: the
    sky-grid index of the tile center and the measured shift in sky-grid
    pixels.
    """
    reference = images[anchor]
    num = reference.shape[0] // num_tile
    result = {}
    for c, image in enumerate(images):
        if c == anchor:
            continue
        rows = []
        for i in range(num_tile):
            for j in range(num_tile):
                s = (slice(i * num, (i + 1) * num), slice(j * num, (j + 1) * num))
                a, b = reference[s], image[s]
                valid = np.isfinite(a) & np.isfinite(b)
                if valid.mean() < fraction_valid:
                    continue
                # the pixels outside the window take the tile's own level, so
                # that the edge of the window is not a step the correlation
                # could lock onto
                a = np.where(valid, a, np.nanmean(a[valid]))
                b = np.where(valid, b, np.nanmean(b[valid]))
                dx, dy = shift_fft(a, b)
                if not (np.isfinite(dx) and np.isfinite(dy)):  # pragma: nocover
                    continue
                rows.append(((i + 0.5) * num, (j + 0.5) * num, dx, dy))
        result[c] = rows
    return result


def measure_channel_shifts(
    linears: list,
    frames: list[np.ndarray],
    sky: na.AbstractCartesian2dVectorArray,
    wavelengths: dict[str, u.Quantity],
    anchor: int = 1,
    num_tile: int = 6,
    mask_windows: bool = True,
    axis: tuple[str, str] = ("sky_x", "sky_y"),
    fraction_valid: float = 0.95,
) -> tuple[dict[int, list[ShiftField]], list[float]]:
    """
    Measure every channel's tile shifts against the anchor on the sky.

    Each channel's frame is read onto the sky grid through its linearized
    system at every wavelength, inside the window that wavelength
    illuminates, and cross-correlated tile by tile with the anchor's.  This
    is the measurement step of :func:`align_channels`, on its own so that
    the same numbers can be taken frame by frame through a flight.

    Parameters
    ----------
    linears
        The linearized system of each channel, see
        :meth:`optika.systems.SequentialSystem.linearize`, with a field
        stop.
    frames
        Each channel's frame as a plain ``[y, x]`` array.
    sky
        The sky grid, see :func:`sky_grid`.
    wavelengths
        The lines to measure at, by name.
    anchor
        The channel the others are measured against.
    num_tile
        The number of tiles along each axis of the sky grid.
    mask_windows
        Whether to read each frame only inside the line's window, see
        :func:`window_mask`.
    axis
        The logical axes of the sky grid.
    fraction_valid
        The fraction of a tile which must be inside both windows for its
        shift to be measured, see :func:`measure_shifts`.

    Returns
    -------
    For every channel but the anchor, one :class:`ShiftField` per line, in
    sky-grid samples; and the anchor's detector pixels per sky sample at
    each line, which converts them to pixels.
    """
    fields = {c: [] for c in range(len(linears)) if c != anchor}
    scales = []
    for wavelength in wavelengths.values():
        images = []
        for c, linear in enumerate(linears):
            xc, yc = sensor_coordinates(linear.distortion, sky, wavelength, axis)
            if c == anchor:
                scales.append(
                    float(
                        np.nanmedian(
                            np.hypot(np.gradient(xc, axis=0), np.gradient(yc, axis=0))
                        )
                    )
                )
            frame = frames[c]
            if mask_windows:
                mask = window_mask(linear.footprint(wavelength), frame.shape)
                frame = np.where(mask, frame, np.nan)
            images.append(sample_on_sky(frame, xc, yc))
        for c, rows in measure_shifts(images, num_tile, anchor, fraction_valid).items():
            if rows:
                r = np.array(rows)
                fields[c].append(
                    ShiftField(wavelength, r[:, 0], r[:, 1], r[:, 2], r[:, 3])
                )
    return fields, scales


def median_shifts(
    fields: dict[int, list[ShiftField]],
    scales: list[float],
) -> dict[int, np.ndarray]:
    """
    Reduce tile shifts to one median shift per channel, in detector pixels.

    The median of the signed components over every tile and line, scaled
    by the anchor's pixels per sky sample: the number a whole-channel
    misregistration shows as, which averages the tiles' noise down rather
    than floors at it.

    Parameters
    ----------
    fields
        The tile shifts of every channel, see :func:`measure_channel_shifts`.
    scales
        The anchor's pixels per sky sample at each line.
    """
    scale = float(np.mean(scales))
    result = {}
    for c, shift_fields in fields.items():
        if not shift_fields:
            result[c] = np.full(2, np.nan)
            continue
        dx = np.concatenate([f.dx for f in shift_fields])
        dy = np.concatenate([f.dy for f in shift_fields])
        result[c] = np.array([np.median(dx), np.median(dy)]) * scale
    return result


def shift_modes(
    fields: list[ShiftField],
    scale: float,
    num_sky: int,
) -> dict[str, float]:
    """
    Decompose a channel's tile shifts into optical modes and what is left.

    After the median shift, a translation, is removed, a field of tile
    shifts can still carry structure: shifts growing radially from the
    centre are a magnification error, a curl is a rotation, a stretch along
    one axis against the other is an anisotropy, and the rest is either
    higher-order distortion or the noise of the tiles.  The linear part is
    fit by least squares; each mode is reported as the shift it produces at
    the edge of the field, so that the modes and the noise are in the same
    pixels.  Whether what remains is structure or noise is told by the
    correlation of neighbouring tiles' residuals, which noise does not have.

    Parameters
    ----------
    fields
        The tile shifts of one channel, one per line, see
        :func:`measure_channel_shifts`.
    scale
        The anchor's detector pixels per sky sample.
    num_sky
        The number of samples along each axis of the sky grid, whose half
        width is the field's edge.

    Returns
    -------
    A dictionary in detector pixels: ``translation``, the length of the
    median shift; ``magnification``, ``rotation``, ``anisotropy`` and
    ``shear``, the shift each mode makes at the field's edge; ``scatter``,
    the rms of the tiles about the median; ``residual``, the rms after the
    linear modes; and ``correlation``, the correlation of the residuals of
    neighbouring tiles, dimensionless, which is near zero for noise.
    """
    i = np.concatenate([f.i for f in fields])
    j = np.concatenate([f.j for f in fields])
    dx = np.concatenate([f.dx for f in fields])
    dy = np.concatenate([f.dy for f in fields])
    if dx.size < 6:
        keys = (
            "translation",
            "magnification",
            "rotation",
            "anisotropy",
            "shear",
            "scatter",
            "residual",
            "correlation",
        )
        return {k: float("nan") for k in keys}
    half = 0.5 * (num_sky - 1)
    # coordinates from the centre of the field, in units of its half width
    u_ = (i - half) / half
    v_ = (j - half) / half
    mx, my = np.median(dx), np.median(dy)
    rx, ry = dx - mx, dy - my
    design = np.stack([np.ones_like(u_), u_, v_], axis=-1)
    cx, *_ = np.linalg.lstsq(design, rx, rcond=None)
    cy, *_ = np.linalg.lstsq(design, ry, rcond=None)
    # the linear map [[a, b], [c, d]] of shift against position, at the edge
    a, b = cx[1], cx[2]
    c, d = cy[1], cy[2]
    modes = dict(
        magnification=0.5 * (a + d),
        rotation=0.5 * (c - b),
        anisotropy=0.5 * (a - d),
        shear=0.5 * (b + c),
    )
    residual_x = rx - design @ cx
    residual_y = ry - design @ cy
    # neighbouring tiles: pairs closer than 1.5 tile spacings
    spacing = np.min(np.diff(np.unique(i))) if np.unique(i).size > 1 else 1.0
    distance = np.hypot(i[:, None] - i[None, :], j[:, None] - j[None, :])
    near = (distance > 0) & (distance < 1.5 * spacing)
    if near.any():
        r = np.concatenate([residual_x, residual_y])
        pairs = np.argwhere(near)
        left = np.concatenate([residual_x[pairs[:, 0]], residual_y[pairs[:, 0]]])
        right = np.concatenate([residual_x[pairs[:, 1]], residual_y[pairs[:, 1]]])
        correlation = float(np.corrcoef(left, right)[0, 1]) if r.std() > 0 else 0.0
    else:  # pragma: nocover
        correlation = float("nan")
    return dict(
        translation=float(np.hypot(mx, my) * scale),
        **{k: float(abs(v) * scale) for k, v in modes.items()},
        scatter=float(np.sqrt(np.mean(rx**2 + ry**2)) * scale),
        residual=float(np.sqrt(np.mean(residual_x**2 + residual_y**2)) * scale),
        correlation=correlation,
    )


def align_channels(
    instruments: list[esis.optics.abc.AbstractInstrument],
    parameters: list[esis.optics.DistortionParameters],
    frames: list[np.ndarray],
    scene_position: na.AbstractCartesian2dVectorArray,
    wavelengths: dict[str, u.Quantity],
    free: None | tuple[str, ...] = None,
    anchor: int = 1,
    num_sky: int = 1600,
    num_tile: int = 6,
    num_pass: int = 4,
    ridge: float = 1e-3,
    damping: float = 0.8,
    degree: int = 2,
    mask_windows: bool = True,
    log: None | Callable[[str], None] = None,
) -> tuple[list[esis.optics.DistortionParameters], list[float]]:
    """
    Align the channels to one another in their physical parameters.

    Each pass samples every channel's frame on a common sky grid through its
    current distortion at every wavelength in `wavelengths`, measures the
    tile shifts of each channel against the anchor, and solves for the
    increments of that channel's free parameters whose change of the mapping
    best reproduces the measured shifts.  The mapping's Jacobian with respect
    to the parameters comes from one linearization per free parameter; the
    solve is a ridge-regularized least squares with sigma clipping of tiles
    whose correlation locked onto the wrong peak.  Using two lines lets a
    geometric error (the same shift in both) be told from a dispersion error
    (different shifts).

    Parameters
    ----------
    instruments
        The base model of each channel.
    parameters
        The current parameters of each channel, usually the result of the
        absolute fit against the proxy scene.
    frames
        The observed frame of each channel, indexed ``[y, x]``.
    scene_position
        The sky positions spanned by the scene, which set the extent of the
        sky grid.
    wavelengths
        The lines at which the sky images are made, by name.
    free
        The names of the parameters the alignment may change.  If
        :obj:`None`, the grating terms and the sensor placement are free and
        the pointing, the primary terms and the field-stop roll are held
        where the absolute fit put them.
    anchor
        The channel the others are aligned to.
    num_sky
        The number of samples along each axis of the sky grid.
    num_tile
        The number of tiles along each axis when measuring shifts.
    num_pass
        The number of measure-and-solve passes.
    ridge
        The ridge penalty on the parameter increments, in units of the
        parameter bounds, which keeps the solve off the flat directions of
        the mapping.
    damping
        The fraction of each solved increment that is applied.
    degree
        The degree of the polynomial distortion model.
    mask_windows
        Whether to read each channel's frame only inside the window the
        line illuminates, see :func:`window_mask`.  Outside it the frame
        holds other lines, which the correlation would otherwise take for
        misplaced structure.
    log
        A callable to report the median shift of each channel after every
        pass.

    Returns
    -------
    The aligned parameters of every channel, and the median shift of every
    channel against the anchor after the last pass, in detector pixels.
    """
    if free is None:
        free = (
            "yaw_grating",
            "pitch_grating",
            "roll_grating",
            "spacing_rulings",
            "z_sensor",
            "roll_sensor",
            "pitch_sensor",
            "yaw_sensor",
            "x_sensor",
            "y_sensor",
        )
    names = [f.name for f in dataclasses.fields(parameters[0])]
    index_free = [names.index(name) for name in free]
    lower, upper = esis.flights.f1.optics.distortion_fit_bounds(parameters[0])
    lb, ub = na.pack(lower).ndarray, na.pack(upper).ndarray
    scale = ub - lb
    delta = 0.002 * scale

    sky = sky_grid(scene_position, num_sky)
    axis = ("sky_x", "sky_y")
    x0, x1 = scene_position.x.min().ndarray, scene_position.x.max().ndarray
    y0, y1 = scene_position.y.min().ndarray, scene_position.y.max().ndarray
    step_x = (x1 - x0) / (num_sky - 1)
    step_y = (y1 - y0) / (num_sky - 1)

    def linear(c, x):
        p = na.unpack(np.asarray(x), parameters[c])
        instrument = p.to_instrument(instruments[c])
        return instrument.system.linearize(
            wavelength=instrument.wavelength,
            degree=degree,
            field_stop=True,
        )

    x = [na.pack(p).ndarray.copy() for p in parameters]
    linears = [linear(c, x[c]) for c in range(len(x))]
    distortions = [system.distortion for system in linears]
    medians = None

    for it in range(num_pass + 1):
        fields, scales = measure_channel_shifts(
            linears, frames, sky, wavelengths, anchor, num_tile, mask_windows, axis
        )
        # NaN for a channel with no tile to measure
        reduced = median_shifts(fields, scales)
        medians = [
            (
                0.0
                if c == anchor
                else float(np.hypot(*reduced.get(c, np.full(2, np.nan))))
            )
            for c in range(len(x))
        ]
        if log is not None:
            log(
                f"pass {it}: median |shift| vs channel {anchor} [px] "
                + " ".join(f"{m:.3f}" for m in medians)
            )
        if it == num_pass:
            break

        for c, shift_fields in fields.items():
            if not shift_fields:  # pragma: nocover
                continue
            targets, columns = [], []
            for field in shift_fields:
                ax = x0 + (x1 - x0) * field.i / (num_sky - 1)
                ay = y0 + (y1 - y0) * field.j / (num_sky - 1)
                pos = na.Cartesian2dVectorArray(
                    x=na.ScalarArray(ax, axes="tile"),
                    y=na.ScalarArray(ay, axes="tile"),
                )
                # the shift says this channel shows, at sky point r, what
                # belongs at r + s; the mapping must move so that r lands
                # where r - s lands now
                pos_shifted = na.Cartesian2dVectorArray(
                    x=na.ScalarArray(ax - field.dx * step_x, axes="tile"),
                    y=na.ScalarArray(ay - field.dy * step_y, axes="tile"),
                )

                def sensor(dist, position, wavelength=field.wavelength):
                    s = dist.distort(
                        na.SpectralPositionalVectorArray(
                            wavelength=wavelength, position=position
                        )
                    ).position
                    return np.concatenate(
                        [
                            np.asarray(na.value(s.x).ndarray),
                            np.asarray(na.value(s.y).ndarray),
                        ]
                    )

                s0 = sensor(distortions[c], pos)
                targets.append(sensor(distortions[c], pos_shifted) - s0)
                cols = np.zeros((len(s0), len(x[c])))
                for k in index_free:
                    v = x[c].copy()
                    v[k] += delta[k]
                    cols[:, k] = (sensor(linear(c, v).distortion, pos) - s0) / delta[k]
                columns.append(cols)
            A = np.vstack(columns)
            b = np.concatenate(targets)
            A_s = A * scale[None, :]
            lam = ridge * np.sqrt(np.mean(A_s[:, index_free] ** 2)) * np.sqrt(len(b))
            keep = np.ones(len(b), dtype=bool)
            for _ in range(4):
                dp_s, *_ = np.linalg.lstsq(
                    np.vstack([A_s[keep], lam * np.eye(A.shape[1])]),
                    np.concatenate([b[keep], np.zeros(A.shape[1])]),
                    rcond=None,
                )
                residual = b - A_s @ dp_s
                sigma = max(residual[keep].std(), 0.05)
                keep_new = np.abs(residual) < 3 * sigma
                if keep_new.sum() < len(index_free) + 4 or (keep_new == keep).all():
                    break
                keep = keep_new  # pragma: nocover
            dp = dp_s * scale * damping
            # the frozen parameters stay exactly where they are
            dp[[k for k in range(len(dp)) if k not in index_free]] = 0
            x[c] = np.clip(x[c] + dp, lb, ub)
            linears[c] = linear(c, x[c])
            distortions[c] = linears[c].distortion

    return [na.unpack(x[c], parameters[c]) for c in range(len(x))], medians
