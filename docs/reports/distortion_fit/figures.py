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
import esis
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
    fig, axes = plt.subplots(
        5, 1, figsize=(3.5, 7.6), sharex=True, constrained_layout=True
    )
    esis.flights.f1.optics.plot_distortion_flight(axes, directory, CHANNEL_COLORS)
    for ax, loc, ncol in zip(
        axes,
        ("upper right", "upper center", "upper right", "upper center", "upper center"),
        (2, 4, 1, 2, 3),
    ):
        ax.legend(frameon=False, fontsize=7, ncol=ncol, loc=loc)
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
    axes = esis.flights.f1.optics.plot_coalignment_tiles(
        frames=frames, directory=directory, colors=CHANNEL_COLORS
    )
    fig = axes[0, 0].figure
    fig.set_size_inches(2.3 * axes.shape[1], 2.3 * axes.shape[0])
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
