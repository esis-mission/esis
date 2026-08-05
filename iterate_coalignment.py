#!/usr/bin/env python3
"""
Fit the inter-channel coalignment by iterating on its own residual.

`validate_recovery` showed that the shift a tile correlation returns is
biased low, and that the shortfall grows with the size of the shift: an
injected gradient of 0.45 sky pixels comes back about a fifth short, while
one of 0.18 comes back within two per cent.  A tile displaced by a large
shift samples a displaced patch of the scene, shares less structure with
its counterpart, and its correlation peak is pulled toward zero.

A single pass therefore under-corrects.  The cure is not a better
correlation but a second pass: apply what has been fitted so far, measure
what is left, and add that to the accumulated coefficients.  Each pass
works on a smaller residual than the last, where the shrinkage is smaller,
so the accumulated correction converges on the true warp.

The measured median shift of every pass is printed, which is the honest
check that it is converging rather than wandering.
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
NUM_TILE = int(os.environ.get("ESIS_NUM_TILE", "6"))
NUM_PASS = int(os.environ.get("ESIS_NUM_PASS", "4"))
INTERPOLATION = os.environ.get("ESIS_INTERPOLATION", "linear")

TERMS = ("0", "x", "y", "w")


def main() -> None:
    """Iterate the fit and write the accumulated correction."""
    directory = pathlib.Path(__file__).parent / "coalignment_iterated"
    directory.mkdir(parents=True, exist_ok=True)

    print(
        f"frame {TIME}, {NUM_SKY}^2 sky grid, {NUM_TILE}x{NUM_TILE} tiles,"
        f" anchor channel {coalign.ANCHOR}, {NUM_PASS} passes,"
        f" {INTERPOLATION} sampling",
        flush=True,
    )

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
    distortion = system.linearize(wavelength=wavelength, degree=DEGREE).distortion

    sky = coalign.sky_grid(system, NUM_SKY)
    span_x = sky.stop.x - sky.start.x
    span_y = sky.stop.y - sky.start.y
    pixel_x, pixel_y = span_x / NUM_SKY, span_y / NUM_SKY
    center_x = (sky.stop.x + sky.start.x) / 2
    center_y = (sky.stop.y + sky.start.y) / 2
    scale = float(na.value(span_x).ndarray / NUM_SKY) * u.deg.to(u.arcsec)

    names = list(coalign.LINES)
    wavelength_mean = np.mean([w.to_value(u.AA) for w in coalign.LINES.values()])
    wavelength_half = abs(coalign.LINES[names[1]].to_value(u.AA) - wavelength_mean)

    # the dispersion, for reporting the wavelength term as a velocity
    centers = {}
    for name, wavelength_rest in coalign.LINES.items():
        coordinates = na.SpectralPositionalVectorArray(
            wavelength=wavelength_rest, position=sky
        )
        x = distortion.distort(coordinates).position.x
        centers[name] = float(na.value(x.mean()).ndarray)
    dispersion = (coalign.LINES[names[1]] - coalign.LINES[names[0]]) / (
        centers[names[1]] - centers[names[0]]
    )
    v_pixel = coalign.velocity_per_pixel(coalign.LINES[names[1]], abs(dispersion))

    # the accumulated correction, zero to begin with
    total = {
        c: {f"d{a}_{t}": 0.0 for a in "xy" for t in TERMS}
        for c in range(num_channel)
        if c != coalign.ANCHOR
    }

    def warp_for(c: int, wavelength_rest: u.Quantity):
        """Sky coordinates of one channel with the accumulated correction."""
        row = total[c]
        cx = 2 * (sky.x - center_x) / span_x
        cy = 2 * (sky.y - center_y) / span_y
        dw = (wavelength_rest.to_value(u.AA) - wavelength_mean) / wavelength_half

        dx = row["dx_0"] + row["dx_x"] * cx + row["dx_y"] * cy + row["dx_w"] * dw
        dy = row["dy_0"] + row["dy_x"] * cx + row["dy_y"] * cy + row["dy_w"] * dw

        # the correction subtracts the fitted shift from the sky coordinates,
        # the direction verified to reduce the disagreement rather than double it
        return na.Cartesian2dVectorArray(
            x=sky.x - dx * pixel_x,
            y=sky.y - dy * pixel_y,
        )

    for index_pass in range(NUM_PASS):
        measurements = {}
        for name, wavelength_rest in coalign.LINES.items():
            warp_line = (
                {c: warp_for(c, wavelength_rest) for c in total} if index_pass else None
            )
            sky_images = coalign.project(
                distortion=distortion,
                image=image,
                sky=sky,
                wavelength=wavelength_rest,
                num_channel=num_channel,
                warp=warp_line,
                interpolation=INTERPOLATION,
            )
            measurements[name] = coalign.measure_shifts(sky_images, NUM_TILE)

        print(f"\n=== pass {index_pass} ===", flush=True)
        for c in sorted(total):
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

            dx = np.array(dx)
            dy = np.array(dy)
            median = float(np.median(np.hypot(dx, dy)))

            matrix = coalign.design_matrix(np.array(cx), np.array(cy), np.array(dw))
            bx, by, rms_x, rms_y, kept = coalign.fit_model(matrix, dx, dy)

            for i, t in enumerate(TERMS):
                total[c][f"dx_{t}"] += bx[i]
                total[c][f"dy_{t}"] += by[i]

            print(
                f"channel {c}: median |shift| before this pass {median:.3f} px"
                f" ({median * scale:.2f} arcsec),"
                f" fitted increment |d0| {np.hypot(bx[0], by[0]):.3f},"
                f" |dX| {np.hypot(bx[1], by[1]):.3f},"
                f" rms {rms_x:.3f}/{rms_y:.3f}, {kept} tiles",
                flush=True,
            )

    print("\n=== accumulated correction ===", flush=True)
    rows = []
    for c in sorted(total):
        row = dict(channel=c, **total[c])
        rows.append(row)
        print(
            f"channel {c}: dx = {row['dx_0']:+.3f} {row['dx_x']:+.3f} X"
            f" {row['dx_y']:+.3f} Y {row['dx_w']:+.3f} W",
            flush=True,
        )
        print(
            f"{'':10s} dy = {row['dy_0']:+.3f} {row['dy_x']:+.3f} X"
            f" {row['dy_y']:+.3f} Y {row['dy_w']:+.3f} W",
            flush=True,
        )
        print(
            f"{'':10s} dispersion disagreement"
            f" {abs(row['dx_w']) * v_pixel.value:.1f} km/s equivalent",
            flush=True,
        )

    table = astropy.table.QTable(rows)
    table.meta["anchor"] = coalign.ANCHOR
    table.meta["scale_arcsec"] = scale
    table.meta["velocity_per_pixel"] = v_pixel.value
    table.meta["num_pass"] = NUM_PASS
    table.meta["num_tile"] = NUM_TILE
    table.meta["interpolation"] = INTERPOLATION
    table.meta["description"] = (
        "Inter-channel coalignment of the ESIS-I channels against channel 1, "
        "accumulated over repeated passes so that the amplitude-dependent "
        "shrinkage of a single pass is iterated away. Units of the sky grid, "
        "as a function of the sky coordinates normalized to [-1, 1] and the "
        "wavelength normalized so that the two lines sit at -1 and +1."
    )
    table.write(directory / "coalignment.ecsv", format="ascii.ecsv", overwrite=True)
    print(f"\nwrote {directory / 'coalignment.ecsv'}", flush=True)


if __name__ == "__main__":
    main()
