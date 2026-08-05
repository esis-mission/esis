#!/usr/bin/env python3
"""
Shared machinery for measuring and modeling the inter-channel misalignment.

The ESIS channels are compared to one another rather than to AIA, so the
measurement is insensitive to how well AIA proxies the ESIS lines — the
systematic that bounds the absolute distortion fit. Channel 1 is the
anchor, matching the coalignment of the 2022 paper.
"""

import numpy as np
import astropy.units as u
import astropy.constants
import named_arrays as na

import esis
from esis.flights.f1 import spectrum

ANCHOR = 1

LINES = {
    "He I": spectrum.He_I.wavelength,
    "O V": spectrum.O_V.wavelength,
}


def highpass(a: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """
    Remove the large-scale intensity structure of an image.

    Vignetting and effective-area differences between the channels appear
    as smooth multiplicative gradients, which a correlation would happily
    lock onto instead of the solar structure that actually carries the
    alignment. Subtracting a smoothed copy leaves the structure.

    Parameters
    ----------
    a
        The image to filter.
    sigma
        The standard deviation, in pixels, of the smoothing kernel.
    """
    import scipy.ndimage

    return a - scipy.ndimage.gaussian_filter(a, sigma=sigma, mode="nearest")


def shift_fft(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Locate the shift of `b` relative to `a` by phase correlation.

    The correlation peak is refined to sub-pixel accuracy with a parabola
    through its immediate neighbors along each axis.

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


def sky_grid(system, num: int) -> na.Cartesian2dVectorLinearSpace:
    """
    Build a regular grid on the sky spanning the field of view.

    Parameters
    ----------
    system
        The optical system whose field of view is spanned.
    num
        The number of samples along each axis.
    """
    field = system.rayfunction_default.inputs.field
    return na.Cartesian2dVectorLinearSpace(
        start=na.Cartesian2dVectorArray(x=field.x.min(), y=field.y.min()),
        stop=na.Cartesian2dVectorArray(x=field.x.max(), y=field.y.max()),
        axis=na.Cartesian2dVectorArray("sky_x", "sky_y"),
        num=num,
    )


def project(
    distortion,
    image: na.AbstractScalar,
    sky: na.Cartesian2dVectorLinearSpace,
    wavelength: u.Quantity,
    num_channel: int,
    warp: None | dict = None,
) -> list[np.ndarray]:
    """
    Sample each channel's image on a common sky grid at one wavelength.

    Parameters
    ----------
    distortion
        The distortion model mapping the sky onto the sensor.
    image
        The observed image, with a ``channel`` axis.
    sky
        The grid on the sky to project onto.
    wavelength
        The wavelength at which the sky grid is mapped through the model.
    num_channel
        The number of channels of `image`.
    warp
        An optional per-channel correction applied to the sky coordinates
        before mapping, as returned by :func:`evaluate_warp`.
    """
    num_y = na.shape(image)["detector_y"]
    num_x = na.shape(image)["detector_x"]

    results = []
    for c in range(num_channel):
        position = sky
        if warp is not None and c in warp:
            position = warp[c]

        coordinates = na.SpectralPositionalVectorArray(
            wavelength=wavelength,
            position=position,
        )
        sensor = distortion.distort(coordinates).position

        index = dict(channel=c)
        xc = na.value(sensor.x[index]).ndarray
        yc = na.value(sensor.y[index]).ndarray
        frame = na.value(image[index]).ndarray

        ix = np.rint(xc).astype(int)
        iy = np.rint(yc).astype(int)
        inside = (ix >= 0) & (ix < num_x) & (iy >= 0) & (iy < num_y)

        sampled = np.full(ix.shape, np.nan, dtype=float)
        sampled[inside] = frame[iy[inside], ix[inside]]
        results.append(sampled)

    return results


def measure_shifts(
    sky_images: list[np.ndarray],
    num_tile: int,
    fraction_valid: float = 0.95,
    include_anchor: bool = False,
) -> dict[int, list[tuple]]:
    """
    Measure the tile-by-tile shift of every channel against the anchor.

    Parameters
    ----------
    sky_images
        The sky-plane image of each channel, as returned by :func:`project`.
    num_tile
        The number of tiles along each axis of the sky grid.
    fraction_valid
        The fraction of a tile which must be on the detector in both
        channels for its shift to be measured. Partially valid tiles are
        padded with zeros, whose edges the correlation mistakes for
        structure, so the threshold is deliberately strict.
    include_anchor
        If :obj:`True`, the anchor is also measured against itself, which
        must return zero and so calibrates the noise floor of the method.
    """
    anchor = sky_images[ANCHOR]
    num = anchor.shape[0] // num_tile

    result = {}
    for c, sky_image in enumerate(sky_images):
        if c == ANCHOR and not include_anchor:
            continue
        rows = []
        for i in range(num_tile):
            for j in range(num_tile):
                s = (slice(i * num, (i + 1) * num), slice(j * num, (j + 1) * num))
                a, b = anchor[s], sky_image[s]
                valid = np.isfinite(a) & np.isfinite(b)
                if valid.mean() < fraction_valid:
                    continue
                dx, dy = shift_fft(np.nan_to_num(a), np.nan_to_num(b))
                if not (np.isfinite(dx) and np.isfinite(dy)):
                    continue
                # the tile center, in units of the sky grid normalized to
                # [-1, 1], which keeps the design matrix well conditioned
                cx = 2 * (i + 0.5) / num_tile - 1
                cy = 2 * (j + 0.5) / num_tile - 1
                rows.append((cx, cy, dx, dy))
        result[c] = rows

    return result


def design_matrix(cx: np.ndarray, cy: np.ndarray, dw: np.ndarray) -> np.ndarray:
    """
    Build the design matrix of the correction model.

    The model is affine in the sky coordinates — a constant, a scale, and a
    shear along each axis, which together absorb translation, rotation,
    scale and shear — plus a term linear in wavelength, which absorbs an
    error in the dispersion of a channel. The affine part is shared by the
    lines, because the geometry of a channel cannot depend on wavelength,
    while the dispersion term is what makes a channel place two lines
    inconsistently.

    Parameters
    ----------
    cx
        The normalized sky x coordinate of each measurement.
    cy
        The normalized sky y coordinate of each measurement.
    dw
        The wavelength of each measurement, offset from the mean and
        normalized.
    """
    return np.stack([np.ones_like(cx), cx, cy, dw], axis=~0)


def fit_model(
    matrix: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Solve the correction model for one channel, rejecting outliers.

    A tile whose correlation locked onto the wrong peak is an outlier of
    several pixels, which least squares would chase, so the fit is iterated
    with sigma clipping.

    Parameters
    ----------
    matrix
        The design matrix, from :func:`design_matrix`.
    dx
        The measured shift along the sky x axis of each tile.
    dy
        The measured shift along the sky y axis of each tile.

    Returns
    -------
    The two coefficient vectors, the two residual standard deviations of
    the tiles that survived the clipping, and how many survived.
    """
    keep = np.ones(len(dx), dtype=bool)
    for _ in range(3):
        bx, *_ = np.linalg.lstsq(matrix[keep], dx[keep], rcond=None)
        by, *_ = np.linalg.lstsq(matrix[keep], dy[keep], rcond=None)
        rx = dx - matrix @ bx
        ry = dy - matrix @ by
        scale_x = max(rx[keep].std(), 1e-6)
        scale_y = max(ry[keep].std(), 1e-6)
        keep_new = (np.abs(rx) < 3 * scale_x) & (np.abs(ry) < 3 * scale_y)
        if keep_new.sum() < matrix.shape[1] + 2 or (keep_new == keep).all():
            break
        keep = keep_new
    return bx, by, rx[keep].std(), ry[keep].std(), int(keep.sum())


def velocity_per_pixel(wavelength: u.Quantity, dispersion: u.Quantity) -> u.Quantity:
    """
    Convert a shift of one detector pixel into an apparent Doppler velocity.

    Parameters
    ----------
    wavelength
        The rest wavelength of the line.
    dispersion
        The dispersion of the instrument, in wavelength per pixel.
    """
    return (astropy.constants.c * dispersion / wavelength).to(u.km / u.s)
