#!/usr/bin/env python3
"""
Check that the coalignment recovers a shift it was given, to sub-pixel.

The anchor measured against itself returns zero, which shows the method
does not invent a shift but says nothing about whether it measures a real
one accurately — the quantity that decides whether the fitted correction
is worth applying.

So a known warp is injected instead. The anchor's own image is projected
onto the sky grid twice: once through the plain coordinates, and once
through coordinates displaced by an affine-plus-dispersion field of chosen
coefficients. The second is then treated as if it were another channel and
put through the ordinary measurement and fit. Because both images come
from the same data, the only difference between them is the injected warp,
and the truth is known exactly.

The coefficients are in units of the sky grid, matching the fitted
correction, so the errors reported here are directly comparable with the
0.06 to 0.17 pixel residuals of the real fit.
"""

import os

import numpy as np
import astropy.units as u
import astropy.constants
import named_arrays as na

import esis
import coalign

TIME = int(os.environ.get("ESIS_TIME", "15"))
DEGREE = int(os.environ.get("ESIS_DEGREE", "2"))
NUM_SKY = int(os.environ.get("ESIS_NUM_SKY", "512"))
NUM_TILE = int(os.environ.get("ESIS_NUM_TILE", "6"))
INTERPOLATION = os.environ.get("ESIS_INTERPOLATION", "nearest")
NUM_PASS = int(os.environ.get("ESIS_NUM_PASS", "1"))

# the injected truth, in sky-grid pixels: a translation, a gradient along
# each sky axis, and a dispersion term, all of the size the real fit found.
# ESIS_TRUTH_SCALE multiplies all eight, so the same test can be repeated at
# a different size or with the sign reversed: an estimator biased low gives
# back the same fractional shortfall every time, while a single unlucky
# realization does not.
SCALE = float(os.environ.get("ESIS_TRUTH_SCALE", "1.0"))
TRUTH_X = {
    k: SCALE * v
    for k, v in dict(constant=0.60, x=0.45, y=-0.30, wavelength=0.50).items()
}
TRUTH_Y = {
    k: SCALE * v
    for k, v in dict(constant=-0.40, x=-0.25, y=0.35, wavelength=-0.20).items()
}

# the channel index the warped copy is presented as; any non-anchor index
# works, since the fit treats each channel independently
CHANNEL_TEST = 0


def main() -> None:
    """Inject a known warp, measure it back, and report the error."""
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

    names = list(coalign.LINES)
    wavelength_mean = np.mean([w.to_value(u.AA) for w in coalign.LINES.values()])
    wavelength_half = abs(coalign.LINES[names[1]].to_value(u.AA) - wavelength_mean)

    truth_x = np.array([TRUTH_X[k] for k in ("constant", "x", "y", "wavelength")])
    truth_y = np.array([TRUTH_Y[k] for k in ("constant", "x", "y", "wavelength")])

    def field(
        wavelength_rest: u.Quantity,
        coefficient_x: np.ndarray,
        coefficient_y: np.ndarray,
    ) -> tuple:
        """Evaluate one displacement field over the sky grid."""
        cx = 2 * (sky.x - center_x) / span_x
        cy = 2 * (sky.y - center_y) / span_y
        dw = (wavelength_rest.to_value(u.AA) - wavelength_mean) / wavelength_half
        basis = (1, cx, cy, dw)
        return (
            sum(c * b for c, b in zip(coefficient_x, basis)),
            sum(c * b for c, b in zip(coefficient_y, basis)),
        )

    def warped(
        wavelength_rest: u.Quantity,
        accumulated_x: np.ndarray,
        accumulated_y: np.ndarray,
    ) -> na.Cartesian2dVectorArray:
        """
        Displace the sky coordinates by the injected field, less what is fitted.

        Pass zero sees the injected warp whole.  Later passes see only what
        the accumulated correction has failed to remove, which is the small
        residual the iteration is supposed to converge on.
        """
        dx_true, dy_true = field(wavelength_rest, truth_x, truth_y)
        dx_fit, dy_fit = field(wavelength_rest, accumulated_x, accumulated_y)
        return na.Cartesian2dVectorArray(
            x=sky.x + (dx_true - dx_fit) * pixel_x,
            y=sky.y + (dy_true - dy_fit) * pixel_y,
        )

    accumulated_x = np.zeros(4)
    accumulated_y = np.zeros(4)

    for index_pass in range(NUM_PASS):
        measurements = {}
        for name, wavelength_rest in coalign.LINES.items():
            plain = coalign.project(
                distortion=distortion,
                image=image,
                sky=sky,
                wavelength=wavelength_rest,
                num_channel=num_channel,
                interpolation=INTERPOLATION,
            )
            shifted = coalign.project(
                distortion=distortion,
                image=image,
                sky=sky,
                wavelength=wavelength_rest,
                num_channel=num_channel,
                warp={
                    coalign.ANCHOR: warped(
                        wavelength_rest, accumulated_x, accumulated_y
                    )
                },
                interpolation=INTERPOLATION,
            )

            # the anchor's own image, plain, against the anchor's own image
            # warped: everything else about the two is identical
            pair = list(plain)
            pair[CHANNEL_TEST] = shifted[coalign.ANCHOR]
            measurements[name] = coalign.measure_shifts(pair, NUM_TILE)

        cx, cy, dw, dx, dy = [], [], [], [], []
        for name in coalign.LINES:
            for _cx, _cy, _dx, _dy in measurements[name][CHANNEL_TEST]:
                cx.append(_cx)
                cy.append(_cy)
                dw.append(
                    (coalign.LINES[name].to_value(u.AA) - wavelength_mean)
                    / wavelength_half
                )
                dx.append(_dx)
                dy.append(_dy)

        matrix = coalign.design_matrix(np.array(cx), np.array(cy), np.array(dw))
        bx, by, rms_x, rms_y, kept = coalign.fit_model(
            matrix, np.array(dx), np.array(dy)
        )
        accumulated_x = accumulated_x + bx
        accumulated_y = accumulated_y + by
        worst_remaining = max(
            np.abs(accumulated_x - truth_x).max(),
            np.abs(accumulated_y - truth_y).max(),
        )
        print(
            f"pass {index_pass}: {kept}/{len(dx)} tiles,"
            f" rms {rms_x:.3f}/{rms_y:.3f} px,"
            f" worst remaining error {worst_remaining:.3f} px",
            flush=True,
        )

    bx, by = accumulated_x, accumulated_y

    # the measurement returns the shift of the warped image relative to the
    # plain one, whose sign is set by the convention of the correlation; the
    # constant term fixes it, and the same sign must then hold for every
    # other coefficient or the model is not being recovered at all
    sign = np.sign(bx[0] * truth_x[0]) if bx[0] else 1.0

    print(f"\nfit kept {kept}/{len(dx)} tiles, rms {rms_x:.3f}/{rms_y:.3f} px")
    print(f"sign convention: measured = {sign:+.0f} x injected\n")
    print(f"{'term':>12s} {'injected':>10s} {'recovered':>10s} {'error':>10s}")
    worst = 0.0
    for label, truth, fitted in (
        ("dx const", truth_x[0], sign * bx[0]),
        ("dx X", truth_x[1], sign * bx[1]),
        ("dx Y", truth_x[2], sign * bx[2]),
        ("dx W", truth_x[3], sign * bx[3]),
        ("dy const", truth_y[0], sign * by[0]),
        ("dy X", truth_y[1], sign * by[1]),
        ("dy Y", truth_y[2], sign * by[2]),
        ("dy W", truth_y[3], sign * by[3]),
    ):
        error = fitted - truth
        worst = max(worst, abs(error))
        print(f"{label:>12s} {truth:10.3f} {fitted:10.3f} {error:+10.3f}")

    print(f"\ntruth scale {SCALE:+g}, {NUM_TILE}x{NUM_TILE} tiles, {INTERPOLATION}")
    print(f"worst coefficient error {worst:.3f} sky pixels", flush=True)
    print(
        "for comparison, the real fit left 0.06 to 0.17 px of residual"
        " disagreement between the channels",
        flush=True,
    )


if __name__ == "__main__":
    main()
