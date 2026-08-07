#!/usr/bin/env python3
"""
How big would a given scene grid be, without building its weights?

The weights build is the expensive step, so choosing a grid by trial is
slow.  Every scene cell becomes a quadrilateral on the detector, and the
number of weights it produces is the number of pixels that quadrilateral
touches, which follows from the distortion model alone.  Sampling that
over a coarse subset of the field gives the triple count, and so the
resident memory, in seconds rather than an hour.

The estimate is calibrated against the one configuration whose weights
have actually been built.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse

import numpy as np

import astropy.units as u
import named_arrays as na

import esis
from esis.data._level_4 import _caching

import tied_config

#: The measured triple count of the configuration used in production,
#: after the field-stop support and shadow mask have removed their share.
CALIBRATION = dict(pitch=0.75, num_velocity=24, triples=2.209e9)

#: Bytes per triple once the redundant variance copy is dropped: two
#: int32 indices and one float32 value, in each direction.
BYTES_PER_TRIPLE = 12


def candidates(
    linear,
    position,
    wavelength,
    stride: int,
) -> float:
    """
    Count the detector pixels the scene cells touch, on a subsampled field.

    Parameters
    ----------
    linear
        The linearized optical system.
    position
        The scene grid vertices.
    wavelength
        The wavelengths to evaluate at.
    stride
        Sample every `stride` cells along each field axis.
    """
    x = position.x.ndarray
    y = position.y.ndarray
    x = x[::stride]
    y = y[::stride]

    coordinates = na.SpectralPositionalVectorArray(
        wavelength=wavelength,
        position=na.Cartesian2dVectorArray(
            x=na.ScalarArray(x, axes="field_x"),
            y=na.ScalarArray(y, axes="field_y"),
        ),
    )
    sensor = linear.distortion.distort(coordinates).position

    order = ("channel", "wavelength", "field_x", "field_y")

    def _nd(a):
        """
        Extract an ndarray with the canonical axis order.

        Parameters
        ----------
        a
            The array to extract.
        """
        src = [a.axes.index(ax) for ax in order if ax in a.axes]
        return np.moveaxis(np.asarray(a.ndarray), src, range(len(src)))

    px = _nd(sensor.x.to(u.pix)).astype(np.float64)
    py = _nd(sensor.y.to(u.pix)).astype(np.float64)

    # each subsampled cell stands for stride**2 real cells, but spans
    # stride times more detector pixels along each axis, so shrink its
    # extent back to what one real cell would touch
    lo_x = np.minimum.reduce(
        [px[..., :-1, :-1], px[..., 1:, :-1], px[..., 1:, 1:], px[..., :-1, 1:]]
    )
    hi_x = np.maximum.reduce(
        [px[..., :-1, :-1], px[..., 1:, :-1], px[..., 1:, 1:], px[..., :-1, 1:]]
    )
    lo_y = np.minimum.reduce(
        [py[..., :-1, :-1], py[..., 1:, :-1], py[..., 1:, 1:], py[..., :-1, 1:]]
    )
    hi_y = np.maximum.reduce(
        [py[..., :-1, :-1], py[..., 1:, :-1], py[..., 1:, 1:], py[..., :-1, 1:]]
    )

    span_x = (hi_x - lo_x) / stride
    span_y = (hi_y - lo_y) / stride
    per_cell = (span_x + 1) * (span_y + 1)

    # average over the sampled field, then sum over every channel and
    # velocity cell, and scale back up to the full field
    num_full = (position.x.ndarray.size - 1) * (position.y.ndarray.size - 1)
    return float(per_cell.mean(axis=(-2, -1)).sum()) * num_full


def main() -> None:
    """Size the candidate configurations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--budget", type=float, default=23.0, help="GiB of VRAM")
    parser.add_argument("--chunk", type=int, default=50_000_000)
    parser.add_argument(
        "--config",
        nargs="+",
        default=["0.75,24", "1.0,24", "1.0,14", "0.75,14", "1.5,24", "1.25,24"],
        help="pitch,num_velocity pairs",
    )
    args = parser.parse_args()

    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system
    key = _caching.key_system(system)
    code = _caching.code_state()

    _, _, union = tied_config.grids(CALIBRATION["num_velocity"])
    linear = _caching.linear_system(
        system, key=key, wavelength=union, degree=2, code=code
    )
    print("linear system ready", flush=True)

    windows = tied_config.windows()
    rows = []
    for spec in args.config:
        pitch, num_velocity = spec.split(",")
        pitch = float(pitch)
        num_velocity = int(num_velocity)

        position, _, _, num_field = tied_config.position_grid(system, pitch * u.arcsec)
        _, member_grids, _ = tied_config.grids(num_velocity)

        total = 0.0
        for w, (_, members, _, _) in enumerate(windows):
            for wavelength_0, _ in members:
                grid = member_grids[float(wavelength_0.to_value(u.AA))]
                grid = grid.cell_centers("wavelength")
                total += candidates(linear, position, grid, args.stride)

        rows.append((pitch, num_velocity, num_field, total))
        print(
            f'  {pitch:g}", {num_velocity} bins: num_field={num_field},'
            f" raw candidates {total / 1e9:.2f} G",
            flush=True,
        )

    reference = [
        r
        for r in rows
        if r[0] == CALIBRATION["pitch"] and r[1] == CALIBRATION["num_velocity"]
    ]
    if not reference:
        print("\n(no calibration point in this set; counts are raw)")
        return
    scale = CALIBRATION["triples"] / reference[0][3]
    print(f"\ncalibration against the built configuration: x{scale:.3f}\n")

    print(
        f"{'pitch':>7}{'bins':>6}{'km/s':>7}{'grid':>7}{'triples':>9}"
        f"{'weights':>9}{'buffers':>9}{'total':>8}   verdict"
    )
    for pitch, num_velocity, num_field, raw in rows:
        triples = raw * scale
        weights = 2 * triples * BYTES_PER_TRIPLE / 2**30

        # working set: the scene, one backprojection per channel, the
        # detector images and their variance, and the transient gather
        num_scene = 5 * num_velocity * num_field**2
        buffers = (
            num_scene * 4  # the scene
            + 4 * num_scene * 4  # a backprojection per channel
            + 2 * 4 * 2048 * 1040 * 4  # images and variance
            + 3 * args.chunk * 4  # the gather and its scaled products
        ) / 2**30

        total = weights + buffers
        velocity = 2 * tied_config.LIMIT_VELOCITY.to_value(u.km / u.s) / num_velocity
        if total < 0.85 * args.budget:
            verdict = "comfortable"
        elif total < args.budget:
            verdict = "tight"
        else:
            verdict = "no"
        print(
            f'{pitch:>6g}"{num_velocity:>6d}{velocity:>7.1f}{num_field:>7d}'
            f"{triples / 1e9:>8.2f}G{weights:>9.1f}{buffers:>9.1f}{total:>8.1f}"
            f"   {verdict}"
        )


if __name__ == "__main__":
    main()
