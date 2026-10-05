#!/usr/bin/env python3
"""
Draw the figures of the distortion-fit summary from a run's tables.

    python figures.py flight <directory> <out.pdf|png>   # the flight in one figure
    python figures.py frame <directory> <out.pdf|png>    # model and data at frame 15
    python figures.py tiles <directory> <out.pdf|png>    # tile shifts, frames 0 and 15

``flight`` needs only the committed tables (``distortion_pointing.ecsv``,
``defocus.ecsv``, ``acceptance.ecsv``, ``window_drift.ecsv``), so the
documentation can build it; ``frame`` needs the ``blink_015.npz`` a
``blink.py render`` leaves in the directory.
"""

import pathlib
import sys

import numpy as np
import astropy.table
import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CHANNEL_COLORS = ("#b8541f", "#2f5fa8", "#3c8d5a", "#7b4fa3")


def _read(directory: pathlib.Path, name: str) -> None | astropy.table.QTable:
    path = directory / name
    if not path.exists():
        return None
    return astropy.table.QTable.read(path, format="ascii.ecsv")


def flight(directory: pathlib.Path, out: pathlib.Path) -> None:
    """Draw the pointing, drift, defocus, channel offsets and coalignment in flight."""
    pointing = _read(directory, "distortion_pointing.ecsv")
    defocus = _read(directory, "defocus.ecsv")
    acceptance = _read(directory, "acceptance.ecsv")
    coregistration = _read(directory, "coregistration.ecsv")
    frames = np.asarray(pointing["frame"])
    t = 10 * (frames - frames[0]) / 60  # minutes from the first frame, 10 s cadence

    fig, axes = plt.subplots(
        5, 1, figsize=(3.5, 7.6), sharex=True, constrained_layout=True
    )

    ax = axes[0]
    ax.plot(t, pointing["pitch"].to_value(u.arcsec), color="k", lw=1.2, label="pitch")
    ax.plot(
        t, pointing["yaw"].to_value(u.arcsec), color="k", lw=1.2, ls="--", label="yaw"
    )
    ax.set_ylabel("pointing\n[arcsec]")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    ax = axes[1]
    for c in range(pointing["drift_x"].shape[1]):
        dx = pointing["drift_x"][:, c].to_value(u.pix)
        dy = pointing["drift_y"][:, c].to_value(u.pix)
        ax.plot(t, np.hypot(dx, dy), color=CHANNEL_COLORS[c], lw=1.2, label=f"ch{c}")
    ax.set_ylabel("window drift\n[px]")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center")

    ax = axes[2]
    if defocus is not None:
        ax.plot(
            10 * (np.asarray(defocus["frame"]) - frames[0]) / 60,
            defocus["z_primary_measured"].to_value(u.um),
            ".",
            color="0.4",
            ms=4,
            label="per frame",
        )
        ax.plot(
            t, pointing["z_primary"].to_value(u.um), color="k", lw=1.2, label="applied"
        )
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_ylabel("primary defocus\n[µm]")

    ax = axes[3]
    if "pitch_channel" in pointing.colnames:
        for c in range(pointing["pitch_channel"].shape[1]):
            kwargs = dict(color=CHANNEL_COLORS[c], lw=1.2)
            ax.plot(t, pointing["pitch_channel"][:, c].to_value(u.arcsec), **kwargs)
            ax.plot(
                t, pointing["yaw_channel"][:, c].to_value(u.arcsec), ls="--", **kwargs
            )
        ax.plot([], [], color="k", lw=1.2, label="pitch")
        ax.plot([], [], color="k", lw=1.2, ls="--", label="yaw")
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center")
    ax.set_ylabel("channel offset\n[arcsec]")

    ax = axes[4]
    if coregistration is not None:
        # the channels as the stages before the co-registration left them
        tt = 10 * (np.asarray(coregistration["frame"]) - frames[0]) / 60
        before = np.hypot(
            coregistration["shift_x"].to_value(u.pix),
            coregistration["shift_y"].to_value(u.pix),
        )
        for c in range(before.shape[1]):
            if np.any(before[:, c] > 0):
                ax.plot(tt, before[:, c], color=CHANNEL_COLORS[c], lw=0.8, ls="--")
    if acceptance is not None:
        anchor = acceptance.meta.get("anchor", 1)
        for c in sorted(set(int(v) for v in acceptance["channel"])):
            if c == anchor:
                continue
            rows = acceptance[acceptance["channel"] == c]
            tt = 10 * (np.asarray(rows["frame"]) - frames[0]) / 60
            shift = np.mean(
                [
                    rows[k].to_value(u.pix)
                    for k in rows.colnames
                    if k.startswith("shift_")
                ],
                axis=0,
            )
            ax.plot(
                tt, shift, color=CHANNEL_COLORS[c], lw=1.2, label=f"ch{c} vs ch{anchor}"
            )
        ax.axhline(0.1, color="0.6", lw=0.8, ls=":")
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center")
    ax.set_ylabel("channel shift\n[px]")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time since first frame [min]")
    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.yaxis.label.set_size(8)
    fig.savefig(out, dpi=300)
    print(f"-> {out}")


def _highpass(a: np.ndarray, valid: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """Remove the smooth part of an image inside its valid pixels, and standardize."""
    import scipy.ndimage

    weights = valid.astype(float)
    smooth = scipy.ndimage.gaussian_filter(np.where(valid, a, 0.0), sigma)
    norm = scipy.ndimage.gaussian_filter(weights, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        detail = a - np.where(norm > 1e-3, smooth / norm, 0.0)
    values = detail[valid]
    return (detail - values.mean()) / values.std()


def frame(directory: pathlib.Path, out: pathlib.Path, t: int = 15) -> None:
    """Draw model, data and their difference per channel at one frame."""
    with np.load(directory / f"blink_{t:03d}.npz") as f:
        model = f["model"].astype(float)
        data = f["data"].astype(float)
        inside = f["inside"].astype(float) > 0.5
    num_channel = model.shape[0]
    fig, axes = plt.subplots(
        3, num_channel, figsize=(7.2, 2.9), constrained_layout=True, squeeze=False
    )
    for c in range(num_channel):
        rows = np.where(inside[c].any(axis=1))[0]
        cols = np.where(inside[c].any(axis=0))[0]
        window = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        m, d, ok = model[c][window], data[c][window], inside[c][window]
        # the difference of the detail: the proxy scene has its own line
        # ratios and gradients, which are not what the fit is judged on
        residual = _highpass(d, ok) - _highpass(m, ok)
        for k, (a, title, cmap, lim) in enumerate(
            (
                (d, "data", "gray", (-2.5, 2.5)),
                (m, "model", "gray", (-2.5, 2.5)),
                (residual, "data − model\ndetail", "RdBu_r", (-1.5, 1.5)),
            )
        ):
            ax = axes[k, c]
            ax.imshow(
                np.where(ok, a, np.nan),
                origin="lower",
                cmap=cmap,
                vmin=lim[0],
                vmax=lim[1],
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_title(f"channel {c}", fontsize=9)
            if c == 0:
                ax.set_ylabel(title, fontsize=8)
    fig.savefig(out, dpi=300)
    print(f"-> {out}")


def tiles(directory: pathlib.Path, out: pathlib.Path, frames=(0, 15)) -> None:
    """Draw every channel's tile shifts against the anchor as arrows over the field."""
    table = _read(directory, "acceptance_tiles.ecsv")
    modes = _read(directory, "acceptance_modes.ecsv")
    anchor = table.meta.get("anchor", 1)
    num_sky = table.meta.get("num_sky", 1600)
    lines = list(dict.fromkeys(str(v) for v in table["line"]))
    channels = [c for c in sorted(set(int(v) for v in table["channel"])) if c != anchor]
    fig, axes = plt.subplots(
        len(frames) * len(lines),
        len(channels),
        figsize=(2.3 * len(channels), 2.3 * len(frames) * len(lines)),
        constrained_layout=True,
        squeeze=False,
    )
    scale = 0.5  # arrow length per pixel of shift, in units of the field's width
    for r, (t, line) in enumerate((t, line) for t in frames for line in lines):
        for k, c in enumerate(channels):
            ax = axes[r, k]
            rows = table[
                (np.asarray(table["frame"]) == t)
                & (np.asarray(table["channel"]) == c)
                & (np.asarray(table["line"]) == line)
            ]
            x = np.asarray(rows["i"]) / (num_sky - 1)
            y = np.asarray(rows["j"]) / (num_sky - 1)
            dx = rows["dx"].to_value(u.pix)
            dy = rows["dy"].to_value(u.pix)
            ax.quiver(
                x,
                y,
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=1 / scale,
                width=0.012,
                color=CHANNEL_COLORS[c],
            )
            ax.quiver(
                [0.08],
                [0.08],
                [0.5],
                [0.0],
                angles="xy",
                scale_units="xy",
                scale=1 / scale,
                width=0.012,
                color="0.4",
            )
            ax.text(0.08, 0.13, "0.5 px", fontsize=6, color="0.4")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if modes is not None:
                m = modes[
                    (np.asarray(modes["frame"]) == t)
                    & (np.asarray(modes["channel"]) == c)
                    & (np.asarray(modes["line"]) == line)
                ]
                if len(m):
                    m = m[0]
                    ax.set_title(
                        f"ch{c} vs ch{anchor}, {line}, frame {t}\n"
                        f"shift {float(m['translation'].value):.2f}, "
                        f"mag {float(m['magnification'].value):.2f}, "
                        f"rot {float(m['rotation'].value):.2f}, "
                        f"noise {float(m['residual'].value):.2f} px",
                        fontsize=6,
                    )
    fig.savefig(out, dpi=300)
    print(f"-> {out}")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "flight":
        flight(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
    elif command == "frame":
        frame(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
    elif command == "tiles":
        tiles(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
