#!/usr/bin/env python3
"""
Assess how well the modeled vignetting explains the channel-to-channel gains.

The difference movie showed that coaligning the channels removes the edge
dipoles of a misalignment but leaves a smooth imbalance across the field.
A smooth imbalance is multiplicative, so it is measured as a *ratio* of
two channels rather than a difference, and the question is whether the
model already predicts it.

Two things make this measurable now. The channels are coaligned to about a
tenth of a pixel, so a ratio is no longer contaminated by structure
sliding against itself. And the two lines separate the two candidate
causes: vignetting is geometric and so nearly wavelength-independent over
584 to 630 Angstrom, whereas the multilayer effective area is not. A
residual which is the same at both lines is vignetting; one which differs
between them is effective area.

Note what this cannot measure. A ratio only constrains the *relative*
response of two channels: vignetting common to all four is degenerate with
the overall radiometric calibration and is invisible here.
"""

import os
import pathlib

import numpy as np
import astropy.units as u
import astropy.constants
import astropy.table
import named_arrays as na
import scipy.ndimage

import esis
import coalign

TIME = int(os.environ.get("ESIS_TIME", "15"))
DEGREE = int(os.environ.get("ESIS_DEGREE", "2"))
NUM_SKY = int(os.environ.get("ESIS_NUM_SKY", "512"))
SIGMA = float(os.environ.get("ESIS_SIGMA", "24"))


def main() -> None:
    """Compare the observed channel ratios against the modeled ones."""
    directory = pathlib.Path(__file__).parent / "coalignment_20260804"
    table = astropy.table.QTable.read(
        directory / "coalignment.ecsv", format="ascii.ecsv"
    )
    correction = {int(row["channel"]): row for row in table}

    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system

    l1 = esis.flights.f1.data.level_1()
    image = na.value(l1.outputs[dict(time=TIME)])
    num_channel = na.shape(image)["channel"]

    wavelength_line = na.stack(list(coalign.LINES.values()), axis="line")
    velocity = na.linspace(-100, 100, axis="wavelength", num=3) * u.km / u.s
    wavelength = (wavelength_line * (1 + velocity / astropy.constants.c)).to(u.AA)
    wavelength = wavelength.combine_axes(
        axes=("line", "wavelength"), axis_new="wavelength"
    )

    print("linearizing...", flush=True)
    linear = system.linearize(wavelength=wavelength, degree=DEGREE)
    distortion = linear.distortion
    vignetting = linear.vignetting
    print(f"vignetting model: {type(vignetting).__name__}", flush=True)

    sky = coalign.sky_grid(system, NUM_SKY)
    span_x = sky.stop.x - sky.start.x
    span_y = sky.stop.y - sky.start.y
    pixel_x, pixel_y = span_x / NUM_SKY, span_y / NUM_SKY
    center_x = (sky.stop.x + sky.start.x) / 2
    center_y = (sky.stop.y + sky.start.y) / 2

    names = list(coalign.LINES)
    wavelength_mean = np.mean([w.to_value(u.AA) for w in coalign.LINES.values()])
    wavelength_half = abs(coalign.LINES[names[1]].to_value(u.AA) - wavelength_mean)

    def warp_for(c, wavelength_rest):
        row = correction[c]
        cx = 2 * (sky.x - center_x) / span_x
        cy = 2 * (sky.y - center_y) / span_y
        dw = (wavelength_rest.to_value(u.AA) - wavelength_mean) / wavelength_half
        dx = row["dx_0"] + row["dx_x"] * cx + row["dx_y"] * cy + row["dx_w"] * dw
        dy = row["dy_0"] + row["dy_x"] * cx + row["dy_y"] * cy + row["dy_w"] * dw
        return na.Cartesian2dVectorArray(
            x=sky.x - dx * pixel_x,
            y=sky.y - dy * pixel_y,
        )

    observed = {}
    predicted = {}
    for name, wavelength_rest in coalign.LINES.items():
        warp = {c: warp_for(c, wavelength_rest) for c in correction}
        sky_images = coalign.project(
            distortion=distortion,
            image=image,
            sky=sky,
            wavelength=wavelength_rest,
            num_channel=num_channel,
            warp=warp,
        )
        observed[name] = sky_images

        if vignetting is not None:
            coordinates = na.SpectralPositionalVectorArray(
                wavelength=wavelength_rest, position=sky
            )
            predicted[name] = na.value(vignetting(coordinates)).ndarray

    anchor_index = coalign.ANCHOR
    print(f"\nsmoothing sigma {SIGMA} sky px; ratios against channel "
          f"{anchor_index}\n", flush=True)

    for name in coalign.LINES:
        print(f"=== {name} ===", flush=True)
        anchor = observed[name][anchor_index]
        for c in range(num_channel):
            if c == anchor_index:
                continue
            other = observed[name][c]
            valid = np.isfinite(anchor) & np.isfinite(other) & (anchor > 0)
            if valid.sum() < 100:
                print(f"  channel {c}: too little overlap", flush=True)
                continue

            # smooth numerator and denominator separately before dividing, so
            # that photon noise does not bias the ratio the way it would if
            # the noisy quotient were smoothed instead
            a = np.where(valid, anchor, 0.0)
            b = np.where(valid, other, 0.0)
            w = valid.astype(float)
            a_s = scipy.ndimage.gaussian_filter(a, SIGMA, mode="nearest")
            b_s = scipy.ndimage.gaussian_filter(b, SIGMA, mode="nearest")
            w_s = scipy.ndimage.gaussian_filter(w, SIGMA, mode="nearest")
            good = w_s > 0.5
            ratio = np.where(good, b_s / np.where(a_s == 0, np.nan, a_s), np.nan)

            r = ratio[good & np.isfinite(ratio)]
            r = r / np.median(r)
            print(f"  channel {c}: observed ratio spread "
                  f"{100 * r.std():5.2f}% rms, "
                  f"{100 * (np.percentile(r, 99) - np.percentile(r, 1)):5.2f}% "
                  f"1-99 range", flush=True)

            if name in predicted:
                p = predicted[name]
                pr = (p[c] / p[anchor_index])[good & np.isfinite(ratio)]
                pr = pr / np.median(pr)
                residual = r / pr
                print(f"{'':13s} modeled ratio spread {100 * pr.std():5.2f}% rms",
                      flush=True)
                print(f"{'':13s} after removing the model "
                      f"{100 * residual.std():5.2f}% rms  <- what is unexplained",
                      flush=True)


if __name__ == "__main__":
    main()
