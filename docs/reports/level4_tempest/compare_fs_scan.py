#!/usr/bin/env python3
"""
Edge-artifact diagnostic across field-stop support sizes.

For each scan product, compute the O V window intensity as a function of
distance inward from the support boundary. A support that is too small
shows a pile-up: flux that belongs outside is crammed into the boundary
cells, so the profile spikes at small distance. At the correct size the
profile should approach the interior level smoothly.

The metric reported is the ratio of the mean intensity in the outermost
ring (0-2 arcsec from the edge) to the interior level (10-30 arcsec),
alongside the frame-15 chi-squared, so the geometric and statistical
evidence can be read together.

Experimental tooling: untracked, not part of the esis package.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

DIR = pathlib.Path("/home/group/charleskankelborg/level4_gpu_tied")
OUT = pathlib.Path("/home/group/charleskankelborg/level4_compare")
PITCH = 1.5
NV = 14
NUM_WINDOW = 5

SCALES = [
    (1.000, "gpu_tied_p1.5_nv14.npz"),
    (1.005, "gpu_tied_p1.5_nv14_fs1.005.npz"),
    (1.010, "gpu_tied_p1.5_nv14_fs1.01.npz"),
    (1.015, "gpu_tied_p1.5_nv14_fs1.015.npz"),
    (1.020, "gpu_tied_p1.5_nv14_fs1.02.npz"),
    (1.025, "gpu_tied_p1.5_nv14_fs1.025.npz"),
    (1.030, "gpu_tied_p1.5_nv14_fs1.03.npz"),
    (1.040, "gpu_tied_p1.5_nv14_fs1.04.npz"),
    (1.050, "gpu_tied_p1.5_nv14_fs1.05.npz"),
    (1.075, "gpu_tied_p1.5_nv14_fs1.075.npz"),
    (1.100, "gpu_tied_p1.5_nv14_fs1.1.npz"),
    (1.150, "gpu_tied_p1.5_nv14_fs1.15.npz"),
]

BINS = np.arange(0, 32, 2)


def main() -> None:
    """Compute and plot the edge profiles."""
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    print(f"{'scale':>6} {'inside':>7} {'edge/interior':>14} {'chi2 mean':>10}")

    rows = []
    for scale, name in SCALES:
        path = DIR / name
        if not path.exists():
            continue
        z = np.load(path, mmap_mode="r")
        inside = np.asarray(z["inside"])
        cube = np.asarray(
            z["solutions"][0, (NUM_WINDOW - 1) * NV : NUM_WINDOW * NV],
            dtype=np.float64,
        )
        image = cube.sum(axis=0)
        chi2 = float(np.asarray(z["chi2_final"])[0].mean())

        distance = scipy.ndimage.distance_transform_edt(inside) * PITCH
        profile = np.array(
            [
                image[inside & (distance >= lo) & (distance < hi)].mean()
                for lo, hi in zip(BINS[:-1], BINS[1:])
            ]
        )
        interior = image[inside & (distance >= 10) & (distance < 30)].mean()
        ratio = profile[0] / interior
        centers = (BINS[:-1] + BINS[1:]) / 2
        ax.plot(centers, profile / interior, marker="o", ms=3, label=f"{scale:.3f}")
        print(f"{scale:6.3f} {inside.mean():7.3f} {ratio:14.3f} {chi2:10.3f}")
        rows.append((scale, inside.mean(), ratio, chi2))

    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("distance inward from support boundary (arcsec)")
    ax.set_ylabel("mean O V intensity / interior level")
    ax.set_title("Field-stop support size: edge pile-up (frame 15, 1.5 arcsec)")
    ax.legend(title="fs scale", ncols=2, fontsize=8)
    fig.savefig(OUT / "fs_scan_edge_profile.png", dpi=120)
    plt.close(fig)

    if rows:
        scale, _, ratio, chi2 = np.array(rows).T
        fig, ax1 = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        ax1.plot(scale, ratio, "o-", color="tab:red", label="edge/interior")
        ax1.axhline(1.0, color="tab:red", lw=0.8, ls=":")
        ax1.set_xlabel("field-stop scale")
        ax1.set_ylabel("edge pile-up ratio", color="tab:red")
        ax2 = ax1.twinx()
        ax2.plot(scale, chi2, "s-", color="tab:blue", label="chi2")
        ax2.set_ylabel("mean chi-squared", color="tab:blue")
        fig.savefig(OUT / "fs_scan_knee.png", dpi=120)
        plt.close(fig)
    print("figures saved", flush=True)


if __name__ == "__main__":
    main()
