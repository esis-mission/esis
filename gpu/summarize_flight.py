#!/usr/bin/env python3
"""
Summarize the two arms over the whole flight, robustly.

The per-frame chi squared has occasional enormous spikes — a cosmic ray or
a bad readout can put one channel of one frame two orders of magnitude
above its neighbours — so a mean over frames is meaningless. Everything
here is reported as a median over frames, with the spikes listed
separately rather than averaged in.

    python summarize_flight.py baseline.npz corrected.npz
"""

import sys

import numpy as np

LIMIT_VELOCITY = 210.0
NUM_VELOCITY = 24


def velocity_maps(data) -> tuple[np.ndarray, np.ndarray]:
    """Intensity and mean velocity of every window, as (frame, window, x, y)."""
    solutions = data["solutions"]
    num_frame = solutions.shape[0]
    num_window = solutions.shape[1] // NUM_VELOCITY
    num_x, num_y = solutions.shape[2], solutions.shape[3]

    edges = np.linspace(-LIMIT_VELOCITY, LIMIT_VELOCITY, NUM_VELOCITY + 1)
    velocity = 0.5 * (edges[:-1] + edges[1:])

    intensity = np.empty((num_frame, num_window, num_x, num_y), dtype=np.float32)
    mean = np.empty_like(intensity)
    for f in range(num_frame):
        cube = np.asarray(solutions[f]).reshape(
            num_window, NUM_VELOCITY, num_x, num_y
        )
        total = cube.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean[f] = (cube * velocity[None, :, None, None]).sum(axis=1) / total
        intensity[f] = total
    return intensity, mean


def plane_gradient(value, weight):
    """Intensity-weighted plane fit; gradients along field_x then field_y."""
    num_x, num_y = value.shape
    index_x, index_y = np.mgrid[0:num_x, 0:num_y]
    coordinate_x = 2 * index_x / (num_x - 1) - 1
    coordinate_y = 2 * index_y / (num_y - 1) - 1

    good = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
    if good.sum() < 100:
        return np.nan, np.nan
    a = np.stack(
        [np.ones(good.sum()), coordinate_x[good], coordinate_y[good]], axis=-1
    )
    w = np.sqrt(weight[good])
    solution, *_ = np.linalg.lstsq(a * w[:, None], value[good] * w, rcond=None)
    return 2 * solution[1], 2 * solution[2]


def main() -> None:
    """Print the robust comparison."""
    arms = {
        "baseline": np.load(sys.argv[1], mmap_mode="r"),
        "corrected": np.load(sys.argv[2], mmap_mode="r"),
    }
    labels = [str(s) for s in arms["baseline"]["labels"]]
    frames = np.asarray(arms["baseline"]["frames"])

    chi2 = {k: np.asarray(v["chi2_final"]) for k, v in arms.items()}
    num_channel = chi2["baseline"].shape[1]

    print("=== chi squared, median over the flight ===")
    print(f"{'channel':>8s} {'baseline':>10s} {'corrected':>10s} {'change':>9s}")
    for c in range(num_channel):
        b = np.median(chi2["baseline"][:, c])
        k = np.median(chi2["corrected"][:, c])
        print(f"{c:>8d} {b:10.4f} {k:10.4f} {100 * (k - b) / b:+8.2f}%")

    print("\n=== frames whose chi squared exceeds 3x that channel's median ===")
    for c in range(num_channel):
        column = chi2["baseline"][:, c]
        bad = np.where(column > 3 * np.median(column))[0]
        if bad.size:
            listed = ", ".join(
                f"{frames[i]} ({column[i]:.1f})" for i in bad
            )
            print(f"  channel {c}: {listed}")

    print("\n=== velocity ramp, median over the flight (km/s) ===")
    maps = {k: velocity_maps(v) for k, v in arms.items()}
    print(
        f"{'line':>14s} {'baseline':>9s} {'corrected':>10s} {'change':>9s}"
        f"   {'x base':>7s} {'x corr':>7s}   {'y base':>7s} {'y corr':>7s}"
    )
    for w, label in enumerate(labels):
        component = {}
        for name in arms:
            intensity, mean = maps[name]
            g = np.array(
                [
                    plane_gradient(mean[f, w], intensity[f, w])
                    for f in range(len(frames))
                ]
            )
            component[name] = g
        magnitude = {
            k: np.median(np.hypot(g[:, 0], g[:, 1])) for k, g in component.items()
        }
        change = 100 * (magnitude["corrected"] - magnitude["baseline"])
        change /= magnitude["baseline"]
        print(
            f"{label:>14s} {magnitude['baseline']:9.2f}"
            f" {magnitude['corrected']:10.2f} {change:+8.1f}%"
            f"   {np.median(component['baseline'][:, 0]):7.2f}"
            f" {np.median(component['corrected'][:, 0]):7.2f}"
            f"   {np.median(component['baseline'][:, 1]):7.2f}"
            f" {np.median(component['corrected'][:, 1]):7.2f}"
        )


if __name__ == "__main__":
    main()
