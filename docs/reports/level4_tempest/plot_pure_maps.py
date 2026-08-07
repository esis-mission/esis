#!/usr/bin/env python3
"""
Spectrally pure intensity maps from the co-added flight.

One panel per tied window, each the velocity-integrated intensity of a
single ion, from the inversion of the whole flight summed into one deep
exposure.  The faint lines are the point: O III and O IV reach useful
signal over only about a sixth of the field in a single exposure.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

import esis
import tied_config


def main() -> None:
    """Render the maps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        default=str(pathlib.Path.home() / "esis_local_runs/local_p0.75_nv6_coadd.npz"),
    )
    parser.add_argument("--pitch", type=float, default=0.75)
    parser.add_argument("--out", default="pure_maps.png")
    args = parser.parse_args()

    z = np.load(args.npz)
    solutions = z["solutions"][0]
    inside = np.asarray(z["inside"])
    labels = [str(x) for x in np.asarray(z["labels"])]
    num_velocity = solutions.shape[0] // len(labels)

    system = esis.flights.f1.optics.distortion_fit(num_distribution=0).system
    position, _, _, _ = tied_config.position_grid(system, args.pitch * u.arcsec)
    x = position.x.ndarray.to_value(u.arcsec)
    y = position.y.ndarray.to_value(u.arcsec)
    extent = (x[0], x[-1], y[0], y[-1])

    print(f"chi2: {np.asarray(z['chi2_final'])[0].round(2)}")
    print(f"iterations: {int(np.asarray(z['iterations'])[0])}")

    fig, axs = plt.subplots(
        1, len(labels), figsize=(4.1 * len(labels), 4.8), constrained_layout=True
    )
    for i, (ax, label) in enumerate(zip(axs, labels)):
        cube = solutions[i * num_velocity : (i + 1) * num_velocity]
        image = cube.sum(axis=0).astype(np.float64)
        interior = image[inside]
        vmax = np.percentile(interior, 99.5)
        ax.imshow(
            image.T,
            origin="lower",
            extent=extent,
            norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=vmax),
            cmap="magma",
        )
        ax.set_title(label)
        ax.set_xlabel("helioprojective x (arcsec)")
        ax.set_aspect("equal")
        contrast = interior.std() / interior.mean()
        print(f"  {label:<14} peak {vmax:.3g}  contrast {contrast:.3f}")
    axs[0].set_ylabel("helioprojective y (arcsec)")
    fig.suptitle(
        "Spectrally pure intensity, whole flight co-added"
        f" ({args.pitch:g} arcsec, {num_velocity} velocity bins)"
    )
    fig.savefig(args.out, dpi=110)
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
