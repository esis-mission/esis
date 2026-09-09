"""
The internal alignment of the channels, solved in the physical parameters.

A fit of each channel against a proxy image of the Sun places the channel to
about a pixel; the channels compared with one another on the sky plane
resolve a tenth of a pixel.  This module measures that disagreement the way
the coalignment analysis does (each channel's frame sampled on a common sky
grid through its own distortion, tile by tile against an anchor) and solves
for the increments of the physical parameters whose change of the mapping
reproduces the measured shift field.
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
    "sky_grid",
    "sample_on_sky",
    "measure_shifts",
    "ShiftField",
    "align_channels",
]


def highpass(a: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """
    Remove the large-scale intensity structure of an image.

    Vignetting and effective-area differences between the channels appear
    as smooth multiplicative gradients, which a correlation would happily
    lock onto instead of the solar structure that actually carries the
    alignment.  Subtracting a smoothed copy leaves the structure.

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
        else:
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


def sample_on_sky(
    frame: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Read a frame bilinearly at the given sensor coordinates.

    Points off the sensor read as NaN, not zero: an edge of zeros would be
    mistaken for structure by the correlation.

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
                dx, dy = shift_fft(np.nan_to_num(a), np.nan_to_num(b))
                if not (np.isfinite(dx) and np.isfinite(dy)):
                    continue
                rows.append(((i + 0.5) * num, (j + 0.5) * num, dx, dy))
        result[c] = rows
    return result


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
        the pointing, the primary displacement and the field-stop roll are
        held where the absolute fit put them: the alignment then never
        revises what the absolute fit constrains well, and it still reaches
        the floor of the measurement.
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
            "roll",
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
    scale = na.pack(upper).ndarray - na.pack(lower).ndarray
    delta = 0.002 * scale

    sky = sky_grid(scene_position, num_sky)
    axis = ("sky_x", "sky_y")
    x0, x1 = scene_position.x.min().ndarray, scene_position.x.max().ndarray
    y0, y1 = scene_position.y.min().ndarray, scene_position.y.max().ndarray
    step = (x1 - x0) / (num_sky - 1)

    def distortion(c, x):
        p = na.unpack(np.asarray(x), parameters[c])
        instrument = p.to_instrument(instruments[c])
        return instrument.system.linearize(
            wavelength=instrument.wavelength,
            degree=degree,
        ).distortion

    x = [na.pack(p).ndarray.copy() for p in parameters]
    distortions = [distortion(c, x[c]) for c in range(len(x))]
    medians = None

    for it in range(num_pass + 1):
        # measure: one shift field per channel and line
        fields = {c: [] for c in range(len(x)) if c != anchor}
        scales = []
        for wavelength in wavelengths.values():
            images = []
            for c in range(len(x)):
                xc, yc = sensor_coordinates(distortions[c], sky, wavelength, axis)
                if c == anchor:
                    scales.append(
                        float(
                            np.nanmedian(
                                np.hypot(
                                    np.gradient(xc, axis=0), np.gradient(yc, axis=0)
                                )
                            )
                        )
                    )
                images.append(sample_on_sky(frames[c], xc, yc))
            for c, rows in measure_shifts(images, num_tile, anchor).items():
                if rows:
                    r = np.array(rows)
                    fields[c].append(
                        ShiftField(wavelength, r[:, 0], r[:, 1], r[:, 2], r[:, 3])
                    )
        px_per_step = float(np.mean(scales))
        medians = [
            (
                float(
                    np.median(
                        np.hypot(
                            np.concatenate([f.dx for f in fields[c]]),
                            np.concatenate([f.dy for f in fields[c]]),
                        )
                    )
                    * px_per_step
                )
                if c != anchor and fields[c]
                else 0.0
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

        # solve: the increment of each channel's free parameters
        for c, shift_fields in fields.items():
            if not shift_fields:
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
                    x=na.ScalarArray(ax - field.dx * step, axes="tile"),
                    y=na.ScalarArray(ay - field.dy * step, axes="tile"),
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
                    cols[:, k] = (sensor(distortion(c, v), pos) - s0) / delta[k]
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
                keep = keep_new
            dp = dp_s * scale * damping
            # the frozen parameters stay exactly where they are
            dp[[k for k in range(len(dp)) if k not in index_free]] = 0
            x[c] = np.clip(x[c] + dp, na.pack(lower).ndarray, na.pack(upper).ndarray)
            distortions[c] = distortion(c, x[c])

    return [na.unpack(x[c], parameters[c]) for c in range(len(x))], medians
