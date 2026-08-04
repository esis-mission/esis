#!/usr/bin/env python3
"""
Fit the inter-channel coalignment of the ESIS channels.

Measures the shift field of every channel against channel 1 at both bright
lines, then fits each channel a correction which is affine in the sky
coordinates and linear in wavelength. The two parts answer different
questions: the affine part is the geometric disagreement between the
channels, and the wavelength part is a disagreement about dispersion,
which is what makes a channel place He I and O V inconsistently.

The fit is compared against two alternatives to justify its terms: a pure
translation, and a separate affine fit per line. The residual of each says
whether the extra terms earn their place.
"""

import os
import pathlib

import numpy as np
import astropy.units as u
import astropy.constants
import astropy.table
import named_arrays as na

import esis
import coalign

TIME = int(os.environ.get("ESIS_TIME", "15"))
DEGREE = int(os.environ.get("ESIS_DEGREE", "2"))
NUM_SKY = int(os.environ.get("ESIS_NUM_SKY", "512"))
NUM_TILE = int(os.environ.get("ESIS_NUM_TILE", "8"))


def main() -> None:
    """Measure the shift fields, fit the correction, and report it."""
    directory = pathlib.Path(__file__).parent / "coalignment_20260804"
    directory.mkdir(parents=True, exist_ok=True)

    print(f"frame {TIME}, {NUM_SKY}^2 sky grid, {NUM_TILE}x{NUM_TILE} tiles, "
          f"anchor channel {coalign.ANCHOR}", flush=True)

    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system

    l1 = esis.flights.f1.data.level_1()
    image = na.value(l1.outputs[dict(time=TIME)])
    num_channel = na.shape(image)["channel"]

    wavelength_line = na.stack(list(coalign.LINES.values()), axis="line")
    velocity = na.linspace(-100, 100, axis="wavelength", num=3) * u.km / u.s
    wavelength = (wavelength_line * (1 + velocity / astropy.constants.c)).to(u.AA)
    wavelength = wavelength.combine_axes(
        axes=("line", "wavelength"),
        axis_new="wavelength",
    )

    print("linearizing...", flush=True)
    linear = system.linearize(wavelength=wavelength, degree=DEGREE)
    distortion = linear.distortion

    sky = coalign.sky_grid(system, NUM_SKY)
    scale = float(
        na.value(sky.stop.x - sky.start.x).ndarray / NUM_SKY
    ) * u.deg.to(u.arcsec)

    # measure the shift field at each line
    measurements = {}
    for name, wavelength_rest in coalign.LINES.items():
        sky_images = coalign.project(
            distortion=distortion,
            image=image,
            sky=sky,
            wavelength=wavelength_rest,
            num_channel=num_channel,
        )
        measurements[name] = coalign.measure_shifts(sky_images, NUM_TILE)
        counts = {c: len(r) for c, r in measurements[name].items()}
        print(f"  {name}: measurable tiles {counts}", flush=True)

    # the dispersion, from where the two lines land on the detector
    centers = {}
    for name, wavelength_rest in coalign.LINES.items():
        coordinates = na.SpectralPositionalVectorArray(
            wavelength=wavelength_rest,
            position=sky,
        )
        x = distortion.distort(coordinates).position.x
        centers[name] = float(na.value(x.mean()).ndarray)
    names = list(coalign.LINES)
    d_pixel = centers[names[1]] - centers[names[0]]
    d_wavelength = coalign.LINES[names[1]] - coalign.LINES[names[0]]
    dispersion = d_wavelength / d_pixel
    v_pixel = coalign.velocity_per_pixel(coalign.LINES[names[1]], abs(dispersion))
    print(f"\ndispersion {dispersion:.5f}/px -> {v_pixel:.1f} per detector pixel",
          flush=True)

    wavelength_mean = np.mean([w.to_value(u.AA) for w in coalign.LINES.values()])
    wavelength_half = abs(
        coalign.LINES[names[1]].to_value(u.AA) - wavelength_mean
    )

    rows = []
    print(f"\n{'':8s} {'model':>22s} {'rms dx':>9s} {'rms dy':>9s}", flush=True)
    for c in range(num_channel):
        if c == coalign.ANCHOR:
            continue

        cx, cy, dw, dx, dy = [], [], [], [], []
        for name in coalign.LINES:
            for _cx, _cy, _dx, _dy in measurements[name][c]:
                cx.append(_cx)
                cy.append(_cy)
                dw.append(
                    (coalign.LINES[name].to_value(u.AA) - wavelength_mean)
                    / wavelength_half
                )
                dx.append(_dx)
                dy.append(_dy)

        cx = np.array(cx)
        cy = np.array(cy)
        dw = np.array(dw)
        dx = np.array(dx)
        dy = np.array(dy)

        full = coalign.design_matrix(cx, cy, dw)

        def solve(matrix):
            bx, *_ = np.linalg.lstsq(matrix, dx, rcond=None)
            by, *_ = np.linalg.lstsq(matrix, dy, rcond=None)
            rx = dx - matrix @ bx
            ry = dy - matrix @ by
            return bx, by, rx.std(), ry.std()

        # nested comparison: does each group of terms earn its place?
        _, _, rms_x0, rms_y0 = solve(full[:, :1])
        _, _, rms_x1, rms_y1 = solve(full[:, :3])
        bx, by, rms_x, rms_y = solve(full)

        print(f"channel {c}", flush=True)
        print(f"{'':8s} {'translation only':>22s} {rms_x0:9.3f} {rms_y0:9.3f}",
              flush=True)
        print(f"{'':8s} {'+ affine':>22s} {rms_x1:9.3f} {rms_y1:9.3f}", flush=True)
        print(f"{'':8s} {'+ dispersion':>22s} {rms_x:9.3f} {rms_y:9.3f}", flush=True)
        print(f"{'':8s} dx = {bx[0]:+.3f} {bx[1]:+.3f} X {bx[2]:+.3f} Y "
              f"{bx[3]:+.3f} W", flush=True)
        print(f"{'':8s} dy = {by[0]:+.3f} {by[1]:+.3f} X {by[2]:+.3f} Y "
              f"{by[3]:+.3f} W", flush=True)
        print(f"{'':8s} dispersion disagreement "
              f"{abs(bx[3]) * v_pixel.value:.1f} km/s equivalent", flush=True)

        rows.append(
            dict(
                channel=c,
                dx_0=bx[0], dx_x=bx[1], dx_y=bx[2], dx_w=bx[3],
                dy_0=by[0], dy_x=by[1], dy_y=by[2], dy_w=by[3],
                rms_x=rms_x, rms_y=rms_y,
            )
        )

    table = astropy.table.QTable(rows)
    table.meta["anchor"] = coalign.ANCHOR
    table.meta["scale_arcsec"] = scale
    table.meta["velocity_per_pixel"] = v_pixel.value
    table.meta["description"] = (
        "Inter-channel coalignment of the ESIS-I channels against channel 1, "
        "in units of the sky grid, as a function of the sky coordinates "
        "normalized to [-1, 1] and the wavelength normalized so that the two "
        "lines sit at -1 and +1."
    )
    table.write(directory / "coalignment.ecsv", format="ascii.ecsv", overwrite=True)
    print(f"\nwrote {directory / 'coalignment.ecsv'}", flush=True)


if __name__ == "__main__":
    main()
