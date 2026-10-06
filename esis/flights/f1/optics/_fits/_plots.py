"""Read and draw the committed distortion fit, for the report and the paper."""

from __future__ import annotations
import pathlib
import numpy as np
import astropy.units as u
import astropy.table
import matplotlib.pyplot as plt

__all__ = [
    "distortion_fit_table",
    "plot_distortion_flight",
    "plot_coalignment_tiles",
]

_directory_data = pathlib.Path(__file__).parent.parent / "_instruments" / "_data"

_tables = dict(
    reference="distortion_reference.ecsv",
    pointing="distortion_pointing.ecsv",
    acceptance="acceptance.ecsv",
    acceptance_modes="acceptance_modes.ecsv",
    acceptance_tiles="acceptance_tiles.ecsv",
    window_edges="window_edges.ecsv",
    window_drift="window_drift.ecsv",
    defocus="defocus.ecsv",
    coregistration="coregistration.ecsv",
)
"""The tables the distortion fit committed, by the name this module reads them under."""

CHANNEL_COLORS = ("#b8541f", "#2f5fa8", "#3c8d5a", "#7b4fa3")
"""One colour per channel, the same in every figure."""


def distortion_fit_table(
    name: str,
    directory: None | str | pathlib.Path = None,
) -> astropy.table.QTable:
    """
    Read one of the tables the distortion fit of the flight committed.

    Parameters
    ----------
    name
        Which table: ``reference`` (the fitted parameters of every channel),
        ``pointing`` (the per-frame pointing, defocus, window drift and
        channel offsets), ``acceptance`` (the held-out scores and the
        coalignment metric of every frame), ``acceptance_modes`` and
        ``acceptance_tiles`` (the decomposition of the tile shifts and the
        tiles themselves), ``window_edges`` and ``window_drift`` (the
        measured edges and their motion), ``defocus`` (the defocus of the
        primary through the flight) or ``coregistration`` (the channels
        against one another, and the offsets that register them).
    directory
        The directory to read from.  If :obj:`None`, the committed tables.

    Raises
    ------
    ValueError
        If `name` is not one of the tables.
    """
    if name not in _tables:
        raise ValueError(f"no table named {name!r}; one of {sorted(_tables)}")
    directory = _directory_data if directory is None else pathlib.Path(directory)
    return astropy.table.QTable.read(directory / _tables[name], format="ascii.ecsv")


def _minutes(
    frames: np.ndarray, first: int, cadence: u.Quantity = 10 * u.s
) -> np.ndarray:
    """Convert frame indices to minutes since the first frame."""
    return (np.asarray(frames, dtype=float) - first) * cadence.to_value(u.min)


def plot_distortion_flight(
    axes: None | np.ndarray = None,
    directory: None | str | pathlib.Path = None,
    colors: tuple[str, ...] = CHANNEL_COLORS,
    threshold: None | u.Quantity = 0.1 * u.pix,
) -> np.ndarray:
    """
    Draw the time-dependent terms of the fit and the coalignment through the flight.

    Five panels against time since the first frame: the pointing of the
    payload; the drift of each channel's windows; the measured and the
    applied defocus of the primary; the offset of each channel's own
    pointing from the co-registration; and the shift of every channel's sky
    against the anchor's, before the co-registration dashed and after it
    solid.

    Parameters
    ----------
    axes
        Five axes to draw into, one per panel.  If :obj:`None`, a new figure
        is made.
    directory
        The directory to read the tables from.  If :obj:`None`, the
        committed tables.
    colors
        One colour per channel.
    threshold
        A coalignment to mark on the last panel, or :obj:`None`.

    Returns
    -------
    The axes drawn into.
    """
    pointing = distortion_fit_table("pointing", directory)
    acceptance = distortion_fit_table("acceptance", directory)
    defocus = distortion_fit_table("defocus", directory)
    coregistration = distortion_fit_table("coregistration", directory)
    frames = np.asarray(pointing["frame"])
    first = int(frames[0])
    t = _minutes(frames, first)
    if axes is None:
        _, axes = plt.subplots(5, 1, sharex=True, constrained_layout=True)
    axes = np.asarray(axes)

    ax = axes[0]
    ax.plot(t, pointing["pitch"].to_value(u.arcsec), color="k", label="pitch")
    ax.plot(t, pointing["yaw"].to_value(u.arcsec), color="k", ls="--", label="yaw")
    ax.set_ylabel("pointing\n[arcsec]")

    ax = axes[1]
    for c in range(pointing["drift_x"].shape[1]):
        dx = pointing["drift_x"][:, c].to_value(u.pix)
        dy = pointing["drift_y"][:, c].to_value(u.pix)
        ax.plot(t, np.hypot(dx, dy), color=colors[c], label=f"ch{c}")
    ax.set_ylabel("window drift\n[pix]")

    ax = axes[2]
    ax.plot(
        _minutes(defocus["frame"], first),
        defocus["z_primary_measured"].to_value(u.um),
        ".",
        color="0.4",
        label="per frame",
    )
    ax.plot(t, pointing["z_primary"].to_value(u.um), color="k", label="applied")
    ax.set_ylabel("primary defocus\n[µm]")

    ax = axes[3]
    for c in range(pointing["pitch_channel"].shape[1]):
        ax.plot(t, pointing["pitch_channel"][:, c].to_value(u.arcsec), color=colors[c])
        ax.plot(
            t,
            pointing["yaw_channel"][:, c].to_value(u.arcsec),
            color=colors[c],
            ls="--",
        )
    ax.plot([], [], color="k", label="pitch")
    ax.plot([], [], color="k", ls="--", label="yaw")
    ax.set_ylabel("channel offset\n[arcsec]")

    ax = axes[4]
    anchor = int(acceptance.meta.get("anchor", 1))
    before = np.hypot(
        coregistration["shift_x"].to_value(u.pix),
        coregistration["shift_y"].to_value(u.pix),
    )
    tt = _minutes(coregistration["frame"], first)
    for c in range(before.shape[1]):
        if c != anchor:
            ax.plot(tt, before[:, c], color=colors[c], ls="--", lw=0.8)
    for c in sorted(set(int(v) for v in acceptance["channel"])):
        if c == anchor:
            continue
        rows = acceptance[np.asarray(acceptance["channel"]) == c]
        shift = np.mean(
            [rows[k].to_value(u.pix) for k in rows.colnames if k.startswith("shift_")],
            axis=0,
        )
        ax.plot(
            _minutes(rows["frame"], first),
            shift,
            color=colors[c],
            label=f"ch{c} vs ch{anchor}",
        )
    if threshold is not None:
        ax.axhline(threshold.to_value(u.pix), color="0.6", lw=0.8, ls=":")
    ax.set_ylabel("channel shift\n[pix]")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time since first frame [min]")
    return axes


def plot_coalignment_tiles(
    axes: None | np.ndarray = None,
    frames: tuple[int, ...] = (0, 15),
    directory: None | str | pathlib.Path = None,
    colors: tuple[str, ...] = CHANNEL_COLORS,
    scale: float = 0.5,
) -> np.ndarray:
    """
    Draw every channel's tile shifts against the anchor as arrows over the field.

    One panel per frame, line and channel: the shift of each tile of the
    channel's sky against the anchor's, as an arrow at the tile's place in
    the field, with a half-pixel arrow for scale.  The panel's title gives
    the translation, magnification and rotation of the field of shifts and
    what is left once the linear modes are removed, from the acceptance.

    Parameters
    ----------
    axes
        The axes to draw into, indexed ``[frame and line, channel]``.  If
        :obj:`None`, a new figure is made.
    frames
        The frames to draw.
    directory
        The directory to read the tables from.  If :obj:`None`, the
        committed tables.
    colors
        One colour per channel.
    scale
        The length of an arrow per pixel of shift, as a fraction of the
        field's width.

    Returns
    -------
    The axes drawn into.
    """
    table = distortion_fit_table("acceptance_tiles", directory)
    modes = distortion_fit_table("acceptance_modes", directory)
    anchor = int(table.meta.get("anchor", 1))
    num_sky = int(table.meta.get("num_sky", 1600))
    lines = list(dict.fromkeys(str(v) for v in table["line"]))
    channels = [c for c in sorted(set(int(v) for v in table["channel"])) if c != anchor]
    rows_all = [(t, line) for t in frames for line in lines]
    if axes is None:
        _, axes = plt.subplots(
            len(rows_all), len(channels), constrained_layout=True, squeeze=False
        )
    axes = np.asarray(axes).reshape(len(rows_all), len(channels))
    for r, (t, line) in enumerate(rows_all):
        for k, c in enumerate(channels):
            ax = axes[r, k]
            rows = table[
                (np.asarray(table["frame"]) == t)
                & (np.asarray(table["channel"]) == c)
                & (np.asarray(table["line"]) == line)
            ]
            kwargs = dict(angles="xy", scale_units="xy", scale=1 / scale, width=0.012)
            ax.quiver(
                np.asarray(rows["i"]) / (num_sky - 1),
                np.asarray(rows["j"]) / (num_sky - 1),
                rows["dx"].to_value(u.pix),
                rows["dy"].to_value(u.pix),
                color=colors[c],
                **kwargs,
            )
            ax.quiver([0.08], [0.08], [0.5], [0.0], color="0.4", **kwargs)
            ax.text(0.08, 0.13, "0.5 px", fontsize=6, color="0.4")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
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
    return axes
