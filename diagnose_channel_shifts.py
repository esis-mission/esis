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
        start=na.Cartesian2dVectorArray(x=field.x.min(), y=field.y.min()),
        stop=na.Cartesian2dVectorArray(x=field.x.max(), y=field.y.max()),
        axis=na.Cartesian2dVectorArray("sky_x", "sky_y"),
        num=NUM_SKY,
    )

    num_channel = na.shape(image)["channel"]
    print(f"image shape {na.shape(image)}", flush=True)
    print(f"sky span x {na.value(field.x.min()).ndarray} .. "
          f"{na.value(field.x.max()).ndarray}, "
          f"y {na.value(field.y.min()).ndarray} .. "
          f"{na.value(field.y.max()).ndarray}", flush=True)

    for name, wavelength_rest in LINES.items():
        print(f"\n=== {name} ({wavelength_rest}) ===", flush=True)

        coordinates = na.SpectralPositionalVectorArray(
            wavelength=wavelength_rest,
            position=sky,
        )
        sensor = distortion.distort(coordinates).position
        print(f"  sensor coordinate axes {list(na.shape(sensor.x))}", flush=True)
        print(f"  sensor x range {na.value(sensor.x.min()).ndarray:.1f} .. "
              f"{na.value(sensor.x.max()).ndarray:.1f}", flush=True)
        print(f"  sensor y range {na.value(sensor.y.min()).ndarray:.1f} .. "
              f"{na.value(sensor.y.max()).ndarray:.1f}", flush=True)

        # The image axes are ordered (detector_y, detector_x), and the sensor
        # coordinates are already detector pixel coordinates measured from the
        # corner: a line lands on part of the detector and the rest of the sky
        # grid falls off it, which is exactly the dispersed slitless layout.
        num_y, num_x = na.shape(image)["detector_y"], na.shape(image)["detector_x"]

        sky_images = []
        for c in range(num_channel):
            index = dict(channel=c)
            xc = na.value(sensor.x[index]).ndarray
            yc = na.value(sensor.y[index]).ndarray
            frame = na.value(image[index]).ndarray

            ix = np.rint(xc).astype(int)
            iy = np.rint(yc).astype(int)
            inside = (ix >= 0) & (ix < num_x) & (iy >= 0) & (iy < num_y)

            # off-detector samples are held apart rather than zero-filled, so
            # that they cannot masquerade as dark structure in the correlation
            sampled = np.full(ix.shape, np.nan, dtype=float)
            sampled[inside] = frame[iy[inside], ix[inside]]
            sky_images.append(sampled)
            signal = np.nanmean(sampled) if inside.any() else np.nan
            print(f"  channel {c}: {inside.mean():.1%} of the sky grid on the "
                  f"detector, mean signal {signal:.1f}", flush=True)

        anchor = sky_images[ANCHOR]
        n = NUM_SKY // NUM_TILE
        scale = (
            na.value((sky.stop.x - sky.start.x)).ndarray / NUM_SKY * u.deg
        ).to(u.arcsec)

        for c in range(num_channel):
            if c == ANCHOR:
                continue
            rows = []
            for i in range(NUM_TILE):
                for j in range(NUM_TILE):
                    s = (slice(i * n, (i + 1) * n), slice(j * n, (j + 1) * n))
                    a, b = anchor[s], sky_images[c][s]

                    # a tile is only measurable where both channels actually
                    # saw the sky; a mostly-empty tile yields a meaningless
                    # correlation peak
                    valid = np.isfinite(a) & np.isfinite(b)
                    if valid.mean() < 0.5:
                        rows.append((i, j, np.nan, np.nan, valid.mean()))
                        continue
                    dx, dy = _shift_fft(np.nan_to_num(a), np.nan_to_num(b))
                    rows.append((i, j, dx, dy, valid.mean()))

            dxs = np.array([r[2] for r in rows])
            dys = np.array([r[3] for r in rows])
            good = np.isfinite(dxs) & np.isfinite(dys)
            if not good.any():
                print(f"  channel {c} vs {ANCHOR}: no measurable tiles", flush=True)
                continue
            print(
                f"  channel {c} vs {ANCHOR}: {good.sum()}/{len(rows)} tiles | "
                f"median ({np.median(dxs[good]):+.2f}, {np.median(dys[good]):+.2f}) "
                f"sky px = ({np.median(dxs[good]) * scale.value:+.2f}, "
                f"{np.median(dys[good]) * scale.value:+.2f}) arcsec | "
                f"scatter ({dxs[good].std():.2f}, {dys[good].std():.2f}) px",
                flush=True,
            )
            for i, j, dx, dy, frac in rows:
                if np.isfinite(dx):
                    print(f"      tile ({i},{j}): ({dx:+6.2f}, {dy:+6.2f}) px "
                          f"[{frac:.0%} valid]", flush=True)


if __name__ == "__main__":
    main()
