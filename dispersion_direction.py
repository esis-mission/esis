#!/usr/bin/env python3
"""
Find the sky direction that each channel confuses with a wavelength shift.

A slitless spectrograph cannot tell a displacement on the sky from a shift
in wavelength: both move the image on the sensor. For one channel the two
are degenerate along a single sky direction, the one whose sensor
displacement is parallel to that channel's dispersion.

Knowing that direction for every channel says whether a velocity ramp along
a fixed sky axis can be blamed on dispersion at all. If the four channels
are degenerate along four different sky directions, a ramp shared by the
whole reconstruction cannot be a per-channel dispersion error, and the
explanation has to be looked for elsewhere.

The Jacobian is evaluated by finite differences on the same linearized
distortion the inversion uses, at the center of the field.
"""

import numpy as np
import astropy.units as u
import astropy.constants
import named_arrays as na

import esis
import coalign

DEGREE = 2


def main() -> None:
    """Report the degenerate sky direction of every channel."""
    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system

    wavelength_line = na.stack(list(coalign.LINES.values()), axis="line")
    velocity = na.linspace(-100, 100, axis="wavelength", num=3) * u.km / u.s
    wavelength = (wavelength_line * (1 + velocity / astropy.constants.c)).to(u.AA)
    wavelength = wavelength.combine_axes(
        axes=("line", "wavelength"),
        axis_new="wavelength",
    )

    print("linearizing...", flush=True)
    distortion = system.linearize(wavelength=wavelength, degree=DEGREE).distortion

    sky = coalign.sky_grid(system, 3)
    centre_x = (sky.start.x + sky.stop.x) / 2
    centre_y = (sky.start.y + sky.stop.y) / 2
    span = float(na.value(sky.stop.x - sky.start.x).ndarray)

    wavelength_0 = coalign.LINES["O V"]
    step_sky = 0.02 * span * u.deg
    step_wavelength = 0.05 * u.AA

    def sensor(dx, dy, dw):
        """Map one perturbed sky/wavelength point onto every sensor."""
        coordinates = na.SpectralPositionalVectorArray(
            wavelength=wavelength_0 + dw,
            position=na.Cartesian2dVectorArray(
                x=centre_x + dx,
                y=centre_y + dy,
            ),
        )
        result = distortion.distort(coordinates).position
        return np.stack(
            [
                np.asarray(na.value(result.x).ndarray),
                np.asarray(na.value(result.y).ndarray),
            ]
        )

    # the sensor coordinates come back as plain numbers, so the step sizes
    # are divided out as plain numbers too: the Jacobian is then in sensor
    # pixels per degree and per angstrom, and the direction it yields is in
    # degrees of sky per angstrom
    zero = 0 * u.deg
    s_0 = sensor(zero, zero, 0 * u.AA)
    d_wavelength = (sensor(zero, zero, step_wavelength) - s_0) / step_wavelength.value
    d_x = (sensor(step_sky, zero, 0 * u.AA) - s_0) / step_sky.value
    d_y = (sensor(zero, step_sky, 0 * u.AA) - s_0) / step_sky.value

    num_channel = s_0.shape[-1]
    print(
        "\nthe sky direction each channel confuses with wavelength,"
        " as an azimuth measured from +field_x toward +field_y:",
        flush=True,
    )
    azimuths = []
    for c in range(num_channel):
        jacobian = np.stack([d_x[:, c], d_y[:, c]], axis=-1)
        direction = np.linalg.solve(jacobian, d_wavelength[:, c])
        azimuth = np.degrees(np.arctan2(direction[1], direction[0]))
        azimuths.append(azimuth)
        magnitude = np.linalg.norm(direction)
        print(
            f"  channel {c}: azimuth {azimuth:+7.1f} deg,"
            f" {magnitude * 3600:.1f} arcsec per AA",
            flush=True,
        )

    # a direction and its opposite are the same degeneracy, so fold the
    # azimuths onto a half turn before asking whether they agree
    folded = np.mod(np.array(azimuths), 180.0)
    spread = folded.max() - folded.min()
    print(
        f"\nfolded onto a half turn: {np.array2string(folded, precision=1)} deg,"
        f" spread {spread:.1f} deg",
        flush=True,
    )
    print(
        "a small spread would mean the channels share one degenerate axis, so a"
        " ramp along it could be dispersion; a large one means they do not.",
        flush=True,
    )


if __name__ == "__main__":
    main()
