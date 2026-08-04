#!/usr/bin/env python3
"""
Check that applying the fitted coalignment actually flattens the shift field.

The fit reports its own residual, but a residual is only evidence that a
model describes the measurements — not that applying it to the instrument
model removes the disagreement. This re-projects every channel through its
corrected mapping and measures the shift field again from scratch.

Both signs of the correction are applied, because which one removes the
disagreement depends on the direction the correlation measures, and a
correction applied backwards doubles the error instead of removing it.
That is a cheap and unambiguous check.
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


def main() -> None:
    """Re-measure the shift field with and without the correction."""
    directory = pathlib.Path(__file__).parent / "coalignment_20260804"
    table = astropy.table.QTable.read(
        directory / "coalignment.ecsv",
        format="ascii.ecsv",
    )
    correction = {int(row["channel"]): row for row in table}
    print(f"loaded correction for channels {sorted(correction)}", flush=True)

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

    # the sky grid in the normalized coordinates the correction is written in
    span_x = sky.stop.x - sky.start.x
    span_y = sky.stop.y - sky.start.y
    pixel_x = span_x / NUM_SKY
    pixel_y = span_y / NUM_SKY
    center_x = (sky.stop.x + sky.start.x) / 2
    center_y = (sky.stop.y + sky.start.y) / 2

    names = list(coalign.LINES)
    wavelength_mean = np.mean([w.to_value(u.AA) for w in coalign.LINES.values()])
    wavelength_half = abs(
        coalign.LINES[names[1]].to_value(u.AA) - wavelength_mean
    )

    def warp_for(c: int, wavelength_rest: u.Quantity, sign: float):
        """Build the corrected sky coordinates of one channel."""
        row = correction[c]
        cx = 2 * (sky.x - center_x) / span_x
        cy = 2 * (sky.y - center_y) / span_y
        dw = (wavelength_rest.to_value(u.AA) - wavelength_mean) / wavelength_half

        dx = row["dx_0"] + row["dx_x"] * cx + row["dx_y"] * cy + row["dx_w"] * dw
        dy = row["dy_0"] + row["dy_x"] * cx + row["dy_y"] * cy + row["dy_w"] * dw

        return na.Cartesian2dVectorArray(
            x=sky.x + sign * dx * pixel_x,
            y=sky.y + sign * dy * pixel_y,
        )

    for label, sign in (("uncorrected", None), ("corrected -", -1.0), ("corrected +", +1.0)):
        print(f"\n=== {label} ===", flush=True)
        for name, wavelength_rest in coalign.LINES.items():
            warp = None
            if sign is not None:
                warp = {
                    c: warp_for(c, wavelength_rest, sign)
                    for c in correction
                }

            sky_images = coalign.project(
                distortion=distortion,
                image=image,
                sky=sky,
                wavelength=wavelength_rest,
                num_channel=num_channel,
                warp=warp,
            )
            shifts = coalign.measure_shifts(sky_images, NUM_TILE)

            for c, rows in shifts.items():
                if not rows:
                    print(f"  {name} channel {c}: no measurable tiles", flush=True)
                    continue
                dx = np.array([r[2] for r in rows])
                dy = np.array([r[3] for r in rows])
                magnitude = np.hypot(dx, dy)

                # a handful of tiles whose correlation locks onto the wrong
                # peak lands several pixels out and dominates an rms, so the
                # typical residual is reported as a median and the bad tiles
                # are counted separately rather than averaged in
                outlier = magnitude > 0.5
                print(f"  {name} channel {c}: median |shift| "
                      f"{np.median(magnitude):6.3f} px | "
                      f"rms {np.sqrt((magnitude**2).mean()):6.3f} | "
                      f"{outlier.sum()}/{len(rows)} tiles > 0.5 px", flush=True)


if __name__ == "__main__":
    main()
