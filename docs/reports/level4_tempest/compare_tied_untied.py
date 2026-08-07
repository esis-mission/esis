#!/usr/bin/env python3
"""
Quantitative comparison of the tied+FS+pedestal GPU product vs the untied one.

Metrics, per shared window pair (untied index -> tied index):
- window-integrated intensity maps at selected frames: Pearson correlation
  and total-flux ratio inside the field-stop support
- speckle metric (RMS of the high-pass map over its mean, inside the
  support) -- quantifies the weak-window noise the ties should suppress
- intensity-weighted mean-velocity (Doppler) maps: RMS difference where
  the intensity is significant
- edge profile: mean intensity vs distance to the octagon boundary, for
  the field-stop size question
- event-region spectra at Event E, overlaid per window

Optionally (--chain path given) also compares a chain-start tied product
against the seed-start one: the null-space gap retest on the constrained
model.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

PATH_UNTIED = "/home/group/charleskankelborg/level4_gpu/gpu_full_run.npz"
PATH_TIED = "/home/group/charleskankelborg/level4_gpu_tied/gpu_tied_p0.75_nv24.npz"
OUT = pathlib.Path("/home/group/charleskankelborg/level4_compare")

NV_UNTIED = 23
STRIDE_UNTIED = 24  # 23 cells + 1 gap cell per window
NV_TIED = 24

# (label, untied window index, tied window index)
PAIRS = [
    ("He I 584", 0, 0),
    ("O III 600", 1, 1),
    ("O IV 608", 2, 2),
    ("Mg X 610", 3, 3),
    ("Mg X 625", 4, 3),
    ("O V 630", 5, 4),
]

EVENT = (47.78, -87.84)  # arcsec
BOX = 6.75  # arcsec half-width of the event box


def window_untied(sol, w):
    """
    Slice one untied window from a frame cube.

    Parameters
    ----------
    sol
        The frame solution, shape (wavelength, x, y).
    w
        The window index.
    """
    return sol[w * STRIDE_UNTIED : w * STRIDE_UNTIED + NV_UNTIED]


def window_tied(sol, w):
    """
    Slice one tied window from a frame cube.

    Parameters
    ----------
    sol
        The frame solution, shape (wavelength, x, y).
    w
        The window index.
    """
    return sol[w * NV_TIED : (w + 1) * NV_TIED]


def speckle(image, inside):
    """
    High-pass RMS over mean, inside the support.

    Parameters
    ----------
    image
        The intensity map.
    inside
        The support mask.
    """
    smooth = scipy.ndimage.gaussian_filter(image, 4)
    return float(
        np.sqrt(np.mean(np.square((image - smooth)[inside]))) / image[inside].mean()
    )


def doppler(cube, velocity_center, threshold):
    """
    Intensity-weighted mean velocity map.

    Parameters
    ----------
    cube
        The window cube, shape (velocity, x, y).
    velocity_center
        The velocity bin centers.
    threshold
        Intensity floor below which the map is masked (NaN).
    """
    intensity = cube.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.tensordot(velocity_center, cube, axes=(0, 0)) / intensity
    v[intensity < threshold] = np.nan
    return v, intensity


def main() -> None:
    """Run the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tied", default=PATH_TIED)
    parser.add_argument("--untied", default=PATH_UNTIED)
    parser.add_argument("--chain", default=None, help="chain-start tied npz")
    parser.add_argument("--frames", default="5,15,25")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    frames_check = [int(x) for x in args.frames.split(",")]

    z_u = np.load(args.untied, mmap_mode="r")
    z_t = np.load(args.tied, mmap_mode="r")
    inside = np.asarray(z_t["inside"])
    num_field = inside.shape[0]

    # cell-center grid (0.75 arcsec pitch, centered like tied_config)
    import astropy.units as u

    import esis
    import tied_config

    system = esis.flights.f1.optics.distortion_fit(num_distribution=0).system
    position, center, extent, nf = tied_config.position_grid(system, 0.75 * u.arcsec)
    assert nf == num_field
    xv = position.x.ndarray.to_value(u.arcsec)
    yv = position.y.ndarray.to_value(u.arcsec)
    xc = (xv[:-1] + xv[1:]) / 2
    yc = (yv[:-1] + yv[1:]) / 2
    XX, YY = np.meshgrid(xc, yc, indexing="ij")

    vel_u = np.linspace(-201.25, 201.25, NV_UNTIED + 1)
    vel_u = (vel_u[:-1] + vel_u[1:]) / 2
    vel_t = np.linspace(-210.0, 210.0, NV_TIED + 1)
    vel_t = (vel_t[:-1] + vel_t[1:]) / 2

    distance = scipy.ndimage.distance_transform_edt(inside)
    bins_edge = np.arange(0, 42, 2)

    box = (np.abs(XX - EVENT[0]) < BOX) & (np.abs(YY - EVENT[1]) < BOX)

    for f in frames_check:
        sol_u = np.asarray(z_u["solutions"][f], dtype=np.float64)
        sol_t = np.asarray(z_t["solutions"][f], dtype=np.float64)
        print(f"=== frame {f} ===", flush=True)
        for label, wu, wt in PAIRS:
            cube_u = window_untied(sol_u, wu)
            cube_t = window_tied(sol_t, wt)
            i_u = cube_u.sum(axis=0)
            i_t = cube_t.sum(axis=0)
            corr = np.corrcoef(i_u[inside], i_t[inside])[0, 1]
            flux_ratio = i_t[inside].sum() / i_u[inside].sum()
            s_u = speckle(i_u, inside)
            s_t = speckle(i_t, inside)
            thresh = 5 * np.median(i_u[inside])
            v_u, _ = doppler(cube_u, vel_u, thresh)
            v_t, _ = doppler(cube_t, vel_t, thresh)
            both = np.isfinite(v_u) & np.isfinite(v_t) & inside
            dv = float(np.sqrt(np.nanmean(np.square((v_t - v_u)[both]))))
            print(
                f"  {label:10s}: corr {corr:.4f}, flux t/u {flux_ratio:.3f},"
                f" speckle u {s_u:.3f} t {s_t:.3f}, dv rms {dv:6.1f} km/s",
                flush=True,
            )

    # --- figures at frame 15 -------------------------------------------------
    f = 15
    sol_u = np.asarray(z_u["solutions"][f], dtype=np.float64)
    sol_t = np.asarray(z_t["solutions"][f], dtype=np.float64)

    fig, axs = plt.subplots(
        2, len(PAIRS), figsize=(4 * len(PAIRS), 8), constrained_layout=True
    )
    for k, (label, wu, wt) in enumerate(PAIRS):
        for row, image in enumerate(
            [window_untied(sol_u, wu).sum(axis=0), window_tied(sol_t, wt).sum(axis=0)]
        ):
            ax = axs[row, k]
            ax.imshow(
                image.T,
                origin="lower",
                extent=(xv[0], xv[-1], yv[0], yv[-1]),
                norm=matplotlib.colors.PowerNorm(
                    0.5, vmin=0, vmax=np.percentile(image[inside], 99.5)
                ),
                cmap="viridis",
            )
            ax.set_title(f"{label} {'untied' if row == 0 else 'tied'}")
    fig.savefig(OUT / "intensity_frame15.png", dpi=110)
    plt.close(fig)

    fig, axs = plt.subplots(
        1, len(PAIRS), figsize=(4 * len(PAIRS), 4.2), constrained_layout=True
    )
    for k, (label, wu, wt) in enumerate(PAIRS):
        i_u = window_untied(sol_u, wu).sum(axis=0)
        i_t = window_tied(sol_t, wt).sum(axis=0)
        profile_u = [
            i_u[inside & (distance >= lo) & (distance < hi)].mean()
            for lo, hi in zip(bins_edge[:-1], bins_edge[1:])
        ]
        profile_t = [
            i_t[inside & (distance >= lo) & (distance < hi)].mean()
            for lo, hi in zip(bins_edge[:-1], bins_edge[1:])
        ]
        centers = (bins_edge[:-1] + bins_edge[1:]) / 2 * 0.75
        axs[k].plot(centers, profile_u, label="untied")
        axs[k].plot(centers, profile_t, label="tied")
        axs[k].set_title(label)
        axs[k].set_xlabel("distance to octagon edge (arcsec)")
    axs[0].set_ylabel("mean intensity")
    axs[0].legend()
    fig.savefig(OUT / "edge_profile_frame15.png", dpi=110)
    plt.close(fig)

    fig, axs = plt.subplots(
        1, len(PAIRS), figsize=(4 * len(PAIRS), 4.2), constrained_layout=True
    )
    for k, (label, wu, wt) in enumerate(PAIRS):
        spectrum_u = window_untied(sol_u, wu)[:, box].sum(axis=1)
        spectrum_t = window_tied(sol_t, wt)[:, box].sum(axis=1)
        axs[k].stairs(spectrum_u, np.linspace(-201.25, 201.25, NV_UNTIED + 1))
        axs[k].stairs(spectrum_t, np.linspace(-210, 210, NV_TIED + 1))
        axs[k].set_title(label)
        axs[k].set_xlabel("velocity (km/s)")
    axs[0].set_ylabel("event-box intensity")
    axs[0].legend(["untied", "tied"])
    fig.savefig(OUT / "event_spectra_frame15.png", dpi=110)
    plt.close(fig)
    print("figures saved", flush=True)

    # --- chain vs seed on the constrained model ------------------------------
    if args.chain:
        z_c = np.load(args.chain, mmap_mode="r")
        order = list(np.asarray(z_c["frames"]))
        print("=== chain vs seed (tied model) ===", flush=True)
        for f in range(z_t["solutions"].shape[0]):
            a = np.asarray(z_t["solutions"][f], dtype=np.float64)
            b = np.asarray(z_c["solutions"][order.index(f)], dtype=np.float64)
            rel = np.abs(a - b).sum() / a.sum()
            chi_s = z_t["chi2_final"][f].mean()
            chi_c = z_c["chi2_final"][order.index(f)].mean()
            print(
                f"  frame {f:2d}: rel |dsol| {rel:.4f},"
                f" chi2 seed {chi_s:.3f} chain {chi_c:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
