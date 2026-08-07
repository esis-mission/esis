#!/usr/bin/env python3
"""
Render the movies from a saved tied-window run.

Rebuilds the coordinates the way ``export_fits`` does, measures how far
the scene slides across the grid over the flight, and renders the movies
with that drift undone.  Re-rendering from the saved cube costs a few
minutes, against most of an hour to repeat the inversion.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse
import pathlib
import time as _time

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

import astropy.constants
import astropy.units as u
import named_arrays as na

import esis

import tied_config


def build(path: str, pitch: float, num_velocity: int):
    """
    Rebuild the Level-4 product from a saved run.

    Parameters
    ----------
    path
        The saved ``npz``.
    pitch
        The scene pitch in arcsec.
    num_velocity
        The velocity bins per window.
    """
    z = np.load(path, mmap_mode="r")
    solutions = z["solutions"]
    num_time, num_cell_total, num_x, _ = solutions.shape

    windows = tied_config.windows()
    num_line = len(windows)
    assert num_cell_total == num_line * num_velocity

    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    position, _, _, num_field = tied_config.position_grid(
        instrument.system, pitch * u.arcsec
    )
    assert num_field == num_x

    a = esis.flights.f1.data.level_1()
    time = a.inputs.time
    if a.axis_channel in time.shape:
        time = time.mean(a.axis_channel)

    num_cell_standard = num_line * (num_velocity + 1) - 1
    limit = tied_config.LIMIT_VELOCITY.to_value(u.km / u.s)
    velocity = np.linspace(-limit, limit, num_velocity + 1)
    rest = u.Quantity([w[1][0][0] for w in windows])

    vertices = []
    for i in range(num_line):
        vertices.extend(
            rest[i].to_value(u.AA)
            * (1 + velocity / astropy.constants.c.to_value(u.km / u.s))
        )

    outputs = np.zeros((num_time, num_cell_standard, num_x, num_x), dtype=np.float32)
    for i in range(num_line):
        start = i * (num_velocity + 1)
        outputs[:, start : start + num_velocity] = solutions[
            :, i * num_velocity : (i + 1) * num_velocity
        ]

    unit = u.ph / (u.AA * u.s * u.cm**2 * u.deg**2)
    return esis.data.Level_4(
        inputs=na.TemporalSpectralPositionalVectorArray(
            time=time,
            wavelength=na.ScalarArray(np.array(vertices) * u.AA, axes="wavelength"),
            position=position,
        ),
        outputs=na.ScalarArray(
            outputs * unit, axes=("time", "wavelength", "field_x", "field_y")
        ),
        instrument=instrument,
        wavelength_center=na.ScalarArray(rest, axes="line"),
        label_line=[w[0] for w in windows],
        members_line=[[(u.Quantity(w0), float(r)) for w0, r in w[1]] for w in windows],
        num_velocity=num_velocity,
        mean_chi_squared=na.ScalarArray(
            np.asarray(z["chi2_final"]), axes=("time", "channel")
        ),
        num_iteration=na.ScalarArray(np.asarray(z["iterations"]), axes=("time",)),
        factor_norm=na.ScalarArray(
            np.asarray(z["factor_norm"]), axes=("time", "channel")
        ),
    )


def main() -> None:
    """Render the movies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pitch", type=float, default=0.75)
    parser.add_argument("--num-velocity", type=int, default=24)
    parser.add_argument("--limit-velocity", type=float, default=100.0)
    parser.add_argument("--no-coregister", action="store_true")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = _time.perf_counter()
    level_4 = build(args.npz, args.pitch, args.num_velocity)
    print(f"product rebuilt: {_time.perf_counter() - t0:.0f} s", flush=True)

    drift = None
    if not args.no_coregister:
        drift = level_4.drift()
        offset = drift.to_value(u.arcsec)
        print(
            f"measured drift: x {offset[:, 0].min():+.2f} to {offset[:, 0].max():+.2f},"
            f" y {offset[:, 1].min():+.2f} to {offset[:, 1].max():+.2f} arcsec",
            flush=True,
        )
        # the fitted pointing put the excursion at -3.7 to +2.9 arcsec in x
        # when it was measured against AIA, so these should agree
        print(
            f"  excursion: x {np.ptp(offset[:, 0]):.2f},"
            f" y {np.ptp(offset[:, 1]):.2f} arcsec",
            flush=True,
        )

    limit = args.limit_velocity * u.km / u.s
    index = level_4.num_line - 1

    for name, animation in (
        (
            "full_fov_intensity",
            level_4.animate_intensity(index_line=index, drift=drift),
        ),
        (
            "full_fov_doppler",
            level_4.animate_doppler(
                index_line=index, limit_velocity=limit, drift=drift
            ),
        ),
    ):
        animation.save(out / f"{name}.gif", writer="pillow", fps=5)
        plt.close(animation._fig)
        print(f"saved {name}: {_time.perf_counter() - t0:.0f} s", flush=True)

    position_event = level_4.locate_event()
    print(f"event located at {position_event}", flush=True)

    try:
        context = esis.flights.f1.data.aia_context()
        import sunpy.visualization.colormaps  # noqa: F401

        cmaps = {"AIA 304": "sdoaia304", "AIA 171": "sdoaia171", "AIA 193": "sdoaia193"}
    except Exception as e:  # noqa: BLE001
        print(f"aia context unavailable ({e})", flush=True)
        context, cmaps = None, None

    animation = level_4.animate_event(
        position=position_event,
        context=context,
        cmaps_context=cmaps,
        limit_velocity=limit,
        drift=drift,
    )
    animation.save(out / "event_e.gif", writer="pillow", fps=5)
    plt.close(animation._fig)
    print(f"TOTAL: {_time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
