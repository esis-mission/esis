#!/usr/bin/env python3
"""
Do the spectrally pure maps behave like their formation temperatures say?

Correlates each co-added line map against co-temporal AIA imagery.  If the
tying and de-blending recover real emission, the correlations should sort
themselves by temperature: the chromospheric lines with 304, and Mg X --
which peaks at log T 6.05, computed from CHIANTI -- with 193, whose Fe XII
response peaks at 6.2 and is the only AIA channel overlapping Mg X at all.

A correlation matrix that ordered itself the wrong way, or that made every
line look alike, would say the windows are sharing flux rather than
separating it.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage

import astropy.units as u

import esis
import tied_config

#: Peak formation temperature of each AIA channel, log10 K, from the
#: standard channel response functions.
TEMPERATURE_AIA = {"AIA 304": 4.7, "AIA 171": 5.85, "AIA 193": 6.2}

#: Peak of each window's contribution function, log10 K.
TEMPERATURE_LINE = {
    "He I 584": 4.3,
    "O III 600": 4.95,
    "O IV 608+610": 5.18,
    "Mg X 610+625": 6.05,
    "O V 630": 5.37,
}


def main() -> None:
    """Run the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        default=str(pathlib.Path.home() / "esis_local_runs/local_p0.75_nv6_coadd.npz"),
    )
    parser.add_argument("--pitch", type=float, default=0.75)
    parser.add_argument("--smooth", type=float, default=2.0, help="arcsec")
    parser.add_argument(
        "--out", default=str(pathlib.Path.home() / "esis_local_runs/aia_compare.png")
    )
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
    xc = (x[:-1] + x[1:]) / 2
    yc = (y[:-1] + y[1:]) / 2
    extent = (x[0], x[-1], y[0], y[-1])

    print("loading the AIA context", flush=True)
    context = esis.flights.f1.data.aia_context()

    # both instruments smoothed to a common scale before correlating, so the
    # answer is about structure rather than about resolution
    sigma = args.smooth / args.pitch

    maps = {}
    for i, label in enumerate(labels):
        cube = solutions[i * num_velocity : (i + 1) * num_velocity]
        maps[label] = scipy.ndimage.gaussian_filter(
            cube.sum(axis=0).astype(np.float64), sigma
        )

    aia = {}
    for label, function in context.items():
        values = np.asarray(function.outputs.ndarray, dtype=np.float64)
        axes = function.outputs.axes
        values = np.moveaxis(
            values,
            [axes.index(a) for a in ("time", "detector_x", "detector_y")],
            (0, 1, 2),
        )
        deep = values.mean(axis=0)
        ax = np.asarray(function.inputs.position.x.ndarray.to_value(u.arcsec))
        ay = np.asarray(function.inputs.position.y.ndarray.to_value(u.arcsec))
        ax = (ax[:-1] + ax[1:]) / 2 if ax.size == deep.shape[0] + 1 else ax
        ay = (ay[:-1] + ay[1:]) / 2 if ay.size == deep.shape[1] + 1 else ay
        interpolate = scipy.interpolate.RegularGridInterpolator(
            (ax, ay), deep, bounds_error=False, fill_value=np.nan
        )
        grid = np.stack(np.meshgrid(xc, yc, indexing="ij"), axis=-1)
        resampled = interpolate(grid)
        aia[label] = scipy.ndimage.gaussian_filter(np.nan_to_num(resampled), sigma)
        print(f"  {label}: resampled onto the scene grid", flush=True)

    # correlate on the interior, away from the support boundary
    interior = scipy.ndimage.binary_erosion(inside, iterations=int(20 / args.pitch))
    print(f"\ncorrelating over {interior.sum()} cells\n")

    order_line = sorted(labels, key=lambda k: TEMPERATURE_LINE[k])
    order_aia = sorted(aia, key=lambda k: TEMPERATURE_AIA[k])

    matrix = np.zeros((len(order_line), len(order_aia)))
    header = f"{'line':<14}{'log T':>7}" + "".join(f"{a:>12}" for a in order_aia)
    print(header)
    for i, line in enumerate(order_line):
        row = f"{line:<14}{TEMPERATURE_LINE[line]:>7.2f}"
        for j, channel in enumerate(order_aia):
            a = maps[line][interior]
            b = aia[channel][interior]
            matrix[i, j] = np.corrcoef(a, b)[0, 1]
            row += f"{matrix[i, j]:>12.3f}"
        best = order_aia[int(np.argmax(matrix[i]))]
        print(row + f"   best: {best}")

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    spec = fig.add_gridspec(2, 3, height_ratios=[1.35, 1])

    ax = fig.add_subplot(spec[0, 0])
    image = maps["Mg X 610+625"]
    ax.imshow(
        image.T,
        origin="lower",
        extent=extent,
        norm=matplotlib.colors.PowerNorm(
            0.5, vmin=0, vmax=np.percentile(image[inside], 99.5)
        ),
        cmap="magma",
    )
    ax.set_title("ESIS Mg X 610+625 — co-added, log T 6.05")
    ax.set_ylabel("helioprojective y (arcsec)")
    ax.set_aspect("equal")

    for k, channel in enumerate(("AIA 193", "AIA 171")):
        ax = fig.add_subplot(spec[0, k + 1])
        image = np.where(inside, aia[channel], np.nan)
        ax.imshow(
            image.T,
            origin="lower",
            extent=extent,
            norm=matplotlib.colors.PowerNorm(
                0.5, vmin=0, vmax=np.nanpercentile(image, 99.5)
            ),
            cmap="magma",
        )
        ax.set_title(f"{channel} — flight average, log T {TEMPERATURE_AIA[channel]}")
        ax.set_aspect("equal")

    ax = fig.add_subplot(spec[1, 0])
    picture = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=max(0.5, matrix.max()))
    ax.set_xticks(range(len(order_aia)))
    ax.set_xticklabels([a.replace("AIA ", "") for a in order_aia])
    ax.set_yticks(range(len(order_line)))
    ax.set_yticklabels(order_line, fontsize=8)
    ax.set_xlabel("AIA channel (cooler to hotter)")
    for i in range(len(order_line)):
        for j in range(len(order_aia)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="w" if matrix[i, j] < 0.6 * matrix.max() else "k",
                fontsize=8,
            )
    fig.colorbar(picture, ax=ax, label="Pearson correlation")
    ax.set_title("correlation, ordered by formation temperature")

    ax = fig.add_subplot(spec[1, 1:])
    for line in order_line:
        ax.plot(
            [TEMPERATURE_AIA[c] for c in order_aia],
            [matrix[order_line.index(line), order_aia.index(c)] for c in order_aia],
            marker="o",
            label=f"{line} ({TEMPERATURE_LINE[line]:.2f})",
        )
    ax.set_xlabel("AIA channel peak, log T")
    ax.set_ylabel("correlation with the ESIS map")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title("each line's affinity across the AIA channels")

    fig.savefig(args.out, dpi=110)
    plt.close(fig)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
