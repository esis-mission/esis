#!/usr/bin/env python3
"""
Compare Doppler colour limits on the real reconstruction.

Renders the O V Doppler map at a mid-flight frame and at the event, under
three limits: the historical fixed 80 km/s, the limit measured from the
visible velocities, and a fixed 120 km/s -- roughly 1.6x the O V sound
speed, so the sonic point sits inside the colour range rather than at its
edge.

Experimental tooling: untracked, not part of the esis package.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

import astropy.constants as const
import astropy.units as u

PATH = "/home/group/charleskankelborg/level4_gpu_tied/gpu_tied_p0.75_nv24_fs1.03.npz"
OUT = pathlib.Path("/home/group/charleskankelborg/level4_compare")
NV = 24
INDEX_OV = 4
FRAME = 15
EVENT = (47.78, -87.84)

# O V 629.73 peaks near log T = 5.37 in ionisation equilibrium
SOUND_SPEED = float(
    np.sqrt(5 / 3 * const.k_B * (10**5.37 * u.K) / (0.6 * const.m_p)).to_value(
        u.km / u.s
    )
)


def main() -> None:
    """Render the comparison."""
    import astropy.units as u

    import esis
    import tied_config

    OUT.mkdir(exist_ok=True)
    z = np.load(PATH, mmap_mode="r")
    inside = np.asarray(z["inside"])
    factor_photon = np.asarray(z["factor_photon"])

    sl = slice(INDEX_OV * NV, (INDEX_OV + 1) * NV)
    cube = np.asarray(z["solutions"][FRAME, sl], dtype=np.float64)
    photons = (cube * factor_photon[sl]).sum(axis=0)

    velocity_bin = np.linspace(-210.0, 210.0, NV + 1)
    velocity_bin = (velocity_bin[:-1] + velocity_bin[1:]) / 2

    intensity = cube.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        velocity = np.tensordot(velocity_bin, cube, axes=(0, 0)) / intensity

    system = esis.flights.f1.optics.distortion_fit(num_distribution=0).system
    position, _, _, _ = tied_config.position_grid(system, 0.75 * u.arcsec)
    x = position.x.ndarray.to_value(u.arcsec)
    y = position.y.ndarray.to_value(u.arcsec)

    alpha_reference = np.nanpercentile(intensity, 99)
    visible = (intensity / alpha_reference) > 0.2
    visible &= np.isfinite(velocity) & inside
    derived = float(np.percentile(np.abs(velocity[visible]), 99))
    print(f"O V sound speed at logT=5.37: {SOUND_SPEED:.1f} km/s")
    print(f"derived limit (99th pct of visible |v|): {derived:.1f} km/s")
    print(f"visible pixels: {100 * visible.mean():.1f}% of the grid")
    for limit in (80.0, derived, 120.0):
        clipped = np.abs(velocity[visible]) > limit
        print(
            f"  limit {limit:6.1f} km/s -> clips {100 * clipped.mean():.2f}% of visible"
        )

    cmap = matplotlib.colormaps["RdBu_r"]
    limits = [
        (f"derived {derived:.0f}", derived),
        ("fixed 40", 40.0),
        ("fixed 80", 80.0),
        ("fixed 120", 120.0),
    ]

    for name, box in (("full", None), ("event", 40.0)):
        if box is None:
            sx = sy = slice(None)
            extent = (x[0], x[-1], y[0], y[-1])
        else:
            xc = (x[:-1] + x[1:]) / 2
            yc = (y[:-1] + y[1:]) / 2
            sx = slice(
                int(np.searchsorted(xc, EVENT[0] - box)),
                int(np.searchsorted(xc, EVENT[0] + box)),
            )
            sy = slice(
                int(np.searchsorted(yc, EVENT[1] - box)),
                int(np.searchsorted(yc, EVENT[1] + box)),
            )
            extent = (xc[sx][0], xc[sx][-1], yc[sy][0], yc[sy][-1])

        fig, axs = plt.subplots(
            1, 4, figsize=(21, 5.4), constrained_layout=True, sharey=True
        )
        for ax, (label, limit) in zip(axs, limits):
            norm = matplotlib.colors.Normalize(vmin=-limit, vmax=limit)
            rgba = cmap(norm(velocity[sx, sy].T))
            alpha = intensity[sx, sy].T / alpha_reference
            rgba[..., 3] = np.clip(np.nan_to_num(alpha), 0, 1)
            ax.imshow(rgba, extent=extent, origin="lower")
            ax.set_title(f"O V 630 — {label} km/s")
            ax.set_xlabel("helioprojective x (arcsec)")
            ax.set_aspect("equal")
            bar = fig.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                orientation="horizontal",
                fraction=0.05,
                label="mean Doppler velocity (km/s)",
            )
            for sign in (-1, 1):
                if abs(SOUND_SPEED) < limit:
                    bar.ax.axvline(sign * SOUND_SPEED, color="k", lw=1.2, ls="--")
        axs[0].set_ylabel("helioprojective y (arcsec)")
        fig.suptitle(
            f"Doppler colour limit, frame {FRAME}"
            f" (dashed: O V sound speed {SOUND_SPEED:.0f} km/s)"
        )
        path = OUT / f"doppler_limits_{name}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
