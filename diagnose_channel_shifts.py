#!/usr/bin/env python3
"""
Measure how far the ESIS channels disagree about the sky, before fitting anything.

For each spectral line, a regular grid on the sky is mapped forward through
every channel's linearized distortion model onto its sensor, and the Level-1
image is sampled there. The resulting sky-plane images are cross-correlated
tile by tile against channel 1 (the anchor of the 2022 coalignment), giving a
shift field across the field of view.

The shape of that field says what the disagreement is:

* shifts under a tenth of a pixel — the disagreement is radiometric
  (vignetting or effective-area gains), not geometric, and coaligning the
  channels will not help;
* a constant shift — a pointing/translation error;
* a rotational pattern — a roll error;
* a pattern growing with field radius — genuine warping, which needs a
  higher-order correction than an affine one.

Nothing is fitted or written here except the measurements.
"""

import os
import pathlib

import numpy as np
import astropy.units as u
import astropy.constants
import named_arrays as na

import esis
from esis.flights.f1 import spectrum

TIME = int(os.environ.get("ESIS_TIME", "15"))
DEGREE = int(os.environ.get("ESIS_DEGREE", "2"))
NUM_SKY = int(os.environ.get("ESIS_NUM_SKY", "512"))
NUM_TILE = int(os.environ.get("ESIS_NUM_TILE", "4"))
ANCHOR = 1

LINES = {
    "He I": spectrum.He_I.wavelength,
    "O V": spectrum.O_V.wavelength,
}


def _shift_fft(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Locate the shift of `b` relative to `a` by phase correlation.

    The correlation peak is refined to sub-pixel accuracy with a parabola
    through its immediate neighbors along each axis.
    """
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
            peak = correlation[index]
            denominator = left - 2 * peak + right
            delta = 0.5 * (left - right) / denominator if denominator else 0.0
        else:
            delta = 0.0
        shift.append(i + delta - center[axis])

    return shift[0], shift[1]


def main() -> None:
    """Measure and report the inter-channel shift field."""
    print(f"frame {TIME}, degree {DEGREE}, {NUM_SKY}^2 sky grid, "
          f"{NUM_TILE}x{NUM_TILE} tiles, anchor channel {ANCHOR}", flush=True)

    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system

    l1 = esis.flights.f1.data.level_1()
    image = l1.outputs[dict(time=TIME)]
    image = na.value(image)

    # linearize on the rest wavelength of each line, plus a small Doppler
    # bracket so the polynomial has something to interpolate along
    wavelength_line = na.stack(list(LINES.values()), axis="line")
    velocity = na.linspace(-100, 100, axis="wavelength", num=3) * u.km / u.s
    wavelength = wavelength_line * (1 + velocity / astropy.constants.c)
    wavelength = wavelength.to(u.AA).combine_axes(
        axes=("line", "wavelength"),
        axis_new="wavelength",
    )

    print("linearizing...", flush=True)
    linear = system.linearize(wavelength=wavelength, degree=DEGREE)
    distortion = linear.distortion

    # a regular sky grid spanning the field of view of the instrument
    field = system.rayfunction_default.inputs.field
    sky = na.Cartesian2dVectorLinearSpace(
        start=field.min(),
        stop=field.max(),
        axis=na.Cartesian2dVectorArray("sky_x", "sky_y"),
        num=NUM_SKY,
    )

    num_channel = na.shape(image)["channel"]
    shape_pixel = na.shape(image)
    print(f"image shape {shape_pixel}", flush=True)

    for name, wavelength_rest in LINES.items():
        print(f"\n=== {name} ({wavelength_rest}) ===", flush=True)

        coordinates = na.SpectralPositionalVectorArray(
            wavelength=wavelength_rest,
            position=sky,
        )
        sensor = distortion.distort(coordinates).position

        # sample the Level-1 image at the mapped sensor coordinates
        x = na.value(sensor.x).ndarray
        y = na.value(sensor.y).ndarray
        data = na.value(image).ndarray

        # the sensor coordinates carry a channel axis; broadcast the sample
        # indices against it
        shape = na.shape(sensor.x)
        axes = list(shape)
        print(f"  sensor coords axes {axes}", flush=True)

        sky_images = []
        for c in range(num_channel):
            index = {ax: c for ax in axes if ax == "channel"}
            xc = na.value(sensor.x)[index].ndarray if index else x
            yc = na.value(sensor.y)[index].ndarray if index else y
            frame = data[c] if data.ndim == 3 else data

            ix = np.rint(xc).astype(int)
            iy = np.rint(yc).astype(int)
            inside = (
                (ix >= 0) & (ix < frame.shape[0]) & (iy >= 0) & (iy < frame.shape[1])
            )
            sampled = np.zeros(ix.shape, dtype=float)
            sampled[inside] = frame[ix[inside], iy[inside]]
            sky_images.append(sampled)
            print(f"  channel {c}: {inside.mean():.1%} of the sky grid on the "
                  f"detector, mean signal {sampled[inside].mean():.1f}", flush=True)

        anchor = sky_images[ANCHOR]
        n = NUM_SKY // NUM_TILE

        for c in range(num_channel):
            if c == ANCHOR:
                continue
            rows = []
            for i in range(NUM_TILE):
                for j in range(NUM_TILE):
                    s = (slice(i * n, (i + 1) * n), slice(j * n, (j + 1) * n))
                    dx, dy = _shift_fft(anchor[s], sky_images[c][s])
                    rows.append((i, j, dx, dy))
            dxs = np.array([r[2] for r in rows])
            dys = np.array([r[3] for r in rows])
            good = np.isfinite(dxs) & np.isfinite(dys)
            print(
                f"  channel {c} vs {ANCHOR}: "
                f"median shift ({np.median(dxs[good]):+.2f}, "
                f"{np.median(dys[good]):+.2f}) sky px, "
                f"scatter ({dxs[good].std():.2f}, {dys[good].std():.2f})",
                flush=True,
            )
            for i, j, dx, dy in rows:
                print(f"      tile ({i},{j}): ({dx:+.2f}, {dy:+.2f})", flush=True)


if __name__ == "__main__":
    main()
