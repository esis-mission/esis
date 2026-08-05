#!/usr/bin/env python3
"""
Compare the baseline and corrected production inversions.

Two questions. Does the reconstruction fit the data better — the per-channel
chi squared. And does the spurious velocity trend across the field of view
shrink — a plane fitted to the intensity-weighted mean velocity of each
window, whose gradient is the red-to-blue ramp the coalignment is meant to
remove.

The gradient is quoted in km/s across the full field, which is the number
to compare against the ~17.5 km/s that one detector pixel of inter-channel
disagreement is worth.
"""

import pathlib

import numpy as np

DIRECTORY = pathlib.Path(__file__).resolve().parent
LIMIT_VELOCITY = 210.0
NUM_VELOCITY = 24


def load(name: str) -> dict:
    """Load one arm's saved inversion."""
    return np.load(DIRECTORY / name, allow_pickle=True)


def velocity_maps(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the intensity and mean velocity of every window and frame.

    Returns arrays shaped (frame, window, field_x, field_y) plus the
    velocity axis.
    """
    solutions = data["solutions"]
    num_frame = solutions.shape[0]
    num_window = solutions.shape[1] // NUM_VELOCITY
    num_x, num_y = solutions.shape[2], solutions.shape[3]

    cube = solutions.reshape(num_frame, num_window, NUM_VELOCITY, num_x, num_y)

    edges = np.linspace(-LIMIT_VELOCITY, LIMIT_VELOCITY, NUM_VELOCITY + 1)
    velocity = 0.5 * (edges[:-1] + edges[1:])

    intensity = cube.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (cube * velocity[None, None, :, None, None]).sum(axis=2) / intensity

    return intensity, mean, velocity


def plane_gradient(
    value: np.ndarray,
    weight: np.ndarray,
    inside: np.ndarray,
) -> tuple[float, float, float]:
    """
    Fit an intensity-weighted plane to a map and return its gradient.

    ``value`` is indexed ``(field_x, field_y)``, and the two gradients come
    back in that order — along ``field_x`` first, then along ``field_y``.
    Each is expressed as the change across the full width of the field, so
    it is directly the size of the red-to-blue ramp along that axis.
    """
    num_x, num_y = value.shape
    index_x, index_y = np.mgrid[0:num_x, 0:num_y]
    # normalize the coordinates to [-1, 1] so a coefficient is half the
    # change across the field
    coordinate_x = 2 * index_x / (num_x - 1) - 1
    coordinate_y = 2 * index_y / (num_y - 1) - 1

    good = inside & np.isfinite(value) & np.isfinite(weight) & (weight > 0)
    if good.sum() < 100:
        return np.nan, np.nan, np.nan

    a = np.stack(
        [np.ones(good.sum()), coordinate_x[good], coordinate_y[good]],
        axis=-1,
    )
    w = np.sqrt(weight[good])
    solution, *_ = np.linalg.lstsq(a * w[:, None], value[good] * w, rcond=None)

    # the full-field change is twice each coefficient
    return solution[0], 2 * solution[1], 2 * solution[2]


def main() -> None:
    """Report the chi squared and velocity trends of both arms."""
    arms = {
        "baseline": load("baseline_p0.75_nv24_fs1.03.npz"),
        "corrected": load("corrected_p0.75_nv24_fs1.03.npz"),
    }

    print("=== chi squared per channel ===", flush=True)
    frames = arms["baseline"]["frames"]
    for i, frame in enumerate(frames):
        b = arms["baseline"]["chi2_final"][i]
        c = arms["corrected"]["chi2_final"][i]
        change = 100 * (c - b) / b
        print(f"frame {frame}", flush=True)
        print(f"  baseline  {np.array2string(b, precision=4)}", flush=True)
        print(f"  corrected {np.array2string(c, precision=4)}", flush=True)
        print(f"  change    {np.array2string(change, precision=2)} %", flush=True)

    print("\n=== velocity trend across the field ===", flush=True)
    labels = [str(s) for s in arms["baseline"]["labels"]]
    inside = arms["baseline"]["inside"]

    maps = {name: velocity_maps(data) for name, data in arms.items()}

    for w, label in enumerate(labels):
        print(f"\n{label}", flush=True)
        for i, frame in enumerate(frames):
            row = f"  frame {frame}:"
            for name in ("baseline", "corrected"):
                intensity, mean, _ = maps[name]
                offset, ramp_x, ramp_y = plane_gradient(
                    mean[i, w], intensity[i, w], inside
                )
                magnitude = np.hypot(ramp_x, ramp_y)
                row += (
                    f"  {name}: offset {offset:+6.2f}"
                    f" ramp (x {ramp_x:+6.2f}, y {ramp_y:+6.2f})"
                    f" |{magnitude:5.2f}| km/s"
                )
            print(row, flush=True)


if __name__ == "__main__":
    main()
