#!/usr/bin/env python3
"""
Measure what the distortion fit cannot explain, from the Level-1 frames alone.

    python diagnostics.py image <directory> <out.npz>     # image against window
    python diagnostics.py edges <directory> <out.npz>     # profile of every edge
    python diagnostics.py report-image <npz> <directory> <figure>
    python diagnostics.py report-edges <npz> <figure>

``<directory>`` holds a run's tables (``distortion_reference.ecsv``,
``window_drift.ecsv``, ``distortion_pointing.ecsv``).

``image`` asks whether each channel's image moves with its window or inside
it.  Against frame 15 it measures, on the detector, the motion of the solar
image, by correlating the interior of the He I and O V windows with the same
pixels at frame 15, their edges eroded away; and the motion of the window,
from the edge table.  Both are taken to the sky through the channel's own
Jacobian, where the pointing moves every channel's image, and not its
window, by one vector; the field stop moves every window, and not its
image, by one vector; a grating or a camera moves the image and the window
of its channel together; and whatever is left moves one channel's image
alone.

``edges`` asks whether the edges are the fiducials the fit takes them for.
For every frame, channel, line and side it fits the error-function step of
:func:`esis.optics.measure_edges` along the rows or columns that cross the
edge, keeping its width, and stacks the profiles about their crossing,
normalized from the dark side to the lit side, which shows the edge's shape
free of the step's symmetry.
"""

import pathlib
import sys
import warnings

import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.special
import astropy.table
import astropy.units as u
import named_arrays as na
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import esis  # noqa: E402
from esis.flights.f1.optics._fits import _fits  # noqa: E402
from esis.optics._distortions import _alignment, _outline  # noqa: E402

REFERENCE = 15
"""The frame every motion is measured against, the one the reference was fit to."""

LINES = {"He I": 0, "O V": 2}
"""The lines whose windows are measured, and their index along the wavelength axis."""

SIDES = ("left", "right", "top", "bottom")
"""The sides of a window, in the order of the edge table."""

CHANNEL_COLORS = ("#b8541f", "#2f5fa8", "#3c8d5a", "#7b4fa3")

ANCHOR = 1
"""The channel the others are compared with."""

LAST = 26
"""The last frame bright enough for its edges to be measured."""


def _windows(directory: pathlib.Path):
    """Linearize the committed fit: the outline and the Jacobian of every window."""
    model = esis.optics.DistortionParameters.from_file(
        directory / "distortion_reference.ecsv"
    ).to_instrument(_fits._base(None))
    origin = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([0.0, 10.0, 0.0]) * u.arcsec, axes="s"),
        y=na.ScalarArray(np.array([0.0, 0.0, 10.0]) * u.arcsec, axes="s"),
    )
    footprints, jacobians = {}, {}
    num_channel = model.camera.channel.shape["channel"]
    for c in range(num_channel):
        channel = model[dict(channel=c)]
        linear = channel.system.linearize(
            wavelength=channel.wavelength, degree=2, field_stop=True
        )
        for name, k in LINES.items():
            wavelength = channel.wavelength[dict(wavelength=k)]
            footprints[c, name] = linear.footprint(wavelength)
            x, y = _alignment.sensor_coordinates(
                linear.distortion, origin, wavelength.ndarray, ("s", "s")
            )
            x, y = (np.asarray(v).ravel() for v in (x, y))
            # sensor pixels (x, y) per arcsecond of sky (x, y)
            jacobians[c, name] = (
                np.array([[x[1] - x[0], x[2] - x[0]], [y[1] - y[0], y[2] - y[0]]])
                / 10.0
            )
    return num_channel, footprints, jacobians


def _frames(level_1, t: int, num_channel: int) -> list[np.ndarray]:
    """Read every channel's frame as a plain ``[y, x]`` array."""
    return _fits._frames_by_axis(level_1[dict(time=t)].outputs.value, num_channel)


def _motion(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Measure the motion of the features from tile `a` to tile `b`, as (x, y)."""
    total = np.zeros(2)
    current = b
    for _ in range(3):
        # `b` shows at r what `a` shows at r + s: the features moved by -s
        s = np.array(_alignment.shift_fft(a, current))
        if not np.all(np.isfinite(s)):
            return np.array([np.nan, np.nan])
        total -= s
        # undo the motion found so far and measure what is left, which
        # removes the bias of the parabola through the correlation's peak
        current = scipy.ndimage.shift(b, -total, order=3, mode="nearest")
    return total[::-1]


def image(directory: pathlib.Path, out: pathlib.Path, tile: int = 160) -> None:
    """Measure the motion of every channel's image and window against frame 15."""
    level_1 = esis.flights.f1.data.level_1()
    num_time = level_1.shape["time"]
    num_channel, footprints, jacobians = _windows(directory)
    reference = _frames(level_1, REFERENCE, num_channel)

    slices = {}
    for (c, name), footprint in footprints.items():
        mask = esis.optics.window_mask(footprint, reference[c].shape)
        # away from the edges, so that the edge itself takes no part
        mask = scipy.ndimage.binary_erosion(mask, iterations=30)
        slices[c, name] = [
            (slice(i, i + tile), slice(j, j + tile))
            for i in range(0, mask.shape[0] - tile + 1, tile // 2)
            for j in range(0, mask.shape[1] - tile + 1, tile // 2)
            if mask[i : i + tile, j : j + tile].all()
        ]

    drift = astropy.table.QTable.read(
        directory / "window_drift.ecsv", format="ascii.ecsv"
    )
    smooth = _fits._drift_smooth(drift, num_channel, 3)
    measured = set(int(t) for t in drift["frame"])

    motion = np.full((num_time, num_channel, len(LINES), 2), np.nan)
    window = np.full((num_time, num_channel, 2), np.nan)
    for t in range(num_time):
        current = _frames(level_1, t, num_channel)
        for c in range(num_channel):
            a = np.where(np.isfinite(reference[c]), reference[c], 0.0)
            b = np.where(np.isfinite(current[c]), current[c], 0.0)
            for k, name in enumerate(LINES):
                rows = np.array([_motion(a[s], b[s]) for s in slices[c, name]])
                rows = rows[np.isfinite(rows).all(axis=1)]
                if len(rows) >= 4:
                    motion[t, c, k] = np.median(rows, axis=0)
            if t in measured:
                window[t, c] = _fits._drift_shift(drift, c, t) - smooth(c, REFERENCE)
        print(f"frame {t}: image {np.round(motion[t, :, 0], 2).tolist()}", flush=True)

    np.savez(
        out,
        image=motion,
        window=window,
        jacobian=np.array(
            [[jacobians[c, name] for name in LINES] for c in range(num_channel)]
        ),
        pixels_per_arcsec=float(np.hypot(*jacobians[ANCHOR, "He I"][:, 0])),
    )
    print(f"-> {out}")


def _trend(frames: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Fit a line through the measured frames; the slope per fifteen frames."""
    v = np.asarray(v, dtype=float)
    good = (frames <= LAST) & np.isfinite(v).reshape(len(frames), -1).all(axis=1)
    if good.sum() < 8:
        return np.full(v.shape[1:], np.nan)
    return np.polyfit(frames[good] - REFERENCE, v[good], 1)[0] * 15


def report_image(
    path: pathlib.Path, directory: pathlib.Path, out: pathlib.Path
) -> None:
    """Print and draw the image's motion against the window's, channel by channel."""
    z = np.load(path)
    jacobian, scale = z["jacobian"], float(z["pixels_per_arcsec"])
    num_time, num_channel = z["image"].shape[:2]
    frames = np.arange(num_time)
    others = [c for c in range(num_channel) if c != ANCHOR]

    def sky(v, c, k=0):
        return np.linalg.solve(jacobian[c, k], np.asarray(v).T).T * scale

    # on the sky, in the anchor's pixels: frame, channel, (x, y)
    moved = np.nanmean(
        [
            np.stack([sky(z["image"][:, c, k], c, k) for c in range(num_channel)], 1)
            for k in range(len(LINES))
        ],
        axis=0,
    )
    window = np.stack([sky(z["window"][:, c], c) for c in range(num_channel)], 1)

    print("trends over the measured frames, pixels on the sky per 15 frames,")
    print(f"each channel against channel {ANCHOR}:")
    for c in others:
        di = moved[:, c] - moved[:, ANCHOR]
        dw = window[:, c] - window[:, ANCHOR]
        ti, tw, td = (_trend(frames, v) for v in (di, dw, di - dw))
        print(
            f"  channel {c}: image ({ti[0]:+.2f}, {ti[1]:+.2f}), "
            f"window ({tw[0]:+.2f}, {tw[1]:+.2f}), "
            f"image - window ({td[0]:+.2f}, {td[1]:+.2f})"
        )
    pointing = astropy.table.QTable.read(
        directory / "distortion_pointing.ecsv", format="ascii.ecsv"
    )
    common = np.nanmean(moved - window, axis=1) / scale
    row = pointing[np.asarray(pointing["frame"]) == 0][0]
    print(
        f"frame 0, the motion common to the channels: ({common[0, 0]:+.2f}, "
        f"{common[0, 1]:+.2f}) arcsec of image against window; the fitted "
        f"pointing is pitch {row['pitch'].to_value(u.arcsec):+.2f}, "
        f"yaw {row['yaw'].to_value(u.arcsec):+.2f} arcsec"
    )

    fig, axes = plt.subplots(
        3, 2, figsize=(9, 8.5), sharex=True, constrained_layout=True
    )
    for j, name in enumerate(("sky x", "sky y")):
        ax = axes[0, j]
        for c in range(num_channel):
            color = CHANNEL_COLORS[c]
            ax.plot(frames, moved[:, c, j], color=color, label=f"ch{c} image")
            ax.plot(frames, window[:, c, j], ".", color=color, ms=4)
        ax.set_title(
            f"motion against frame {REFERENCE}, {name}\n"
            "lines: image interior; dots: measured edges",
            fontsize=9,
        )
        ax = axes[1, j]
        for c in others:
            color = CHANNEL_COLORS[c]
            di = moved[:, c, j] - moved[:, ANCHOR, j]
            dw = window[:, c, j] - window[:, ANCHOR, j]
            ax.plot(frames, di, color=color, label=f"ch{c} - ch{ANCHOR}, image")
            ax.plot(frames, dw, ".", color=color, ms=5, label="edges")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(
            f"channel against channel {ANCHOR}, {name}\n"
            "an image that rides with its window puts line and dots together",
            fontsize=9,
        )
        ax = axes[2, j]
        for c in others:
            v = (moved[:, c, j] - moved[:, ANCHOR, j]) - (
                window[:, c, j] - window[:, ANCHOR, j]
            )
            ax.plot(frames, v, "o-", ms=3, color=CHANNEL_COLORS[c], label=f"ch{c}")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(f"image minus window against channel {ANCHOR}, {name}", fontsize=9)
        ax.set_xlabel("frame")
    for ax in axes[:, 0]:
        ax.set_ylabel("pixels on the sky")
        ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.savefig(out, dpi=160)
    print(f"-> {out}")


def _step(x, a, b, x0, sigma):
    return a + b * scipy.special.erf((x - x0) / (np.sqrt(2) * sigma))


OFFSETS = np.arange(-12, 12.01, 0.25)
"""The distances from the crossing at which the stacked edge profile is sampled."""


def _fit_edge(profile: np.ndarray, guess: float, half: int = 30):
    """Fit the step of :func:`esis.optics.measure_edges`, keeping every parameter."""
    lo, hi = int(guess - half), int(guess + half)
    if lo < 2 or hi > profile.size - 2:
        return None
    x = np.arange(lo, hi)
    y = profile[lo:hi]
    if not np.all(np.isfinite(y)):
        return None
    b0 = (y[-8:].mean() - y[:8].mean()) / 2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = scipy.optimize.curve_fit(
                _step, x, y, p0=[y.mean(), b0, guess, 2.0], maxfev=4000
            )
    except Exception:
        return None
    if abs(p[3]) > 12 or abs(p[2] - guess) > half / 2 or abs(p[1]) < 0.2 * abs(p[0]):
        return None
    # the profile from the dark side (0) to the lit side (1), about the crossing
    normalized = (y - (p[0] - abs(p[1]))) / (2 * abs(p[1]))
    distance = x - p[2]
    if p[1] < 0:
        distance, normalized = -distance[::-1], normalized[::-1]
    return p[2] + 0.5, abs(p[3]), np.interp(OFFSETS, distance, normalized)


def edges(directory: pathlib.Path, out: pathlib.Path) -> None:
    """Measure where every window edge is, how sharp and what shape, in every frame."""
    level_1 = esis.flights.f1.data.level_1()
    num_time = level_1.shape["time"]
    num_channel, footprints, _ = _windows(directory)
    polygons = {key: _outline._polygon(f) for key, f in footprints.items()}

    shape = (num_time, num_channel, len(LINES), len(SIDES))
    position = np.full(shape, np.nan)
    sigma = np.full(shape, np.nan)
    profile = np.full(shape + (OFFSETS.size,), np.nan)
    reference = {}
    order = [REFERENCE] + [t for t in range(num_time) if t != REFERENCE]
    for t in order:
        frames = _frames(level_1, t, num_channel)
        for c in range(num_channel):
            frame = scipy.ndimage.median_filter(
                np.asarray(frames[c], dtype=float), size=(3, 1)
            )
            for j, name in enumerate(LINES):
                px, py = polygons[c, name]
                found = _crossings(frame, px, py)
                for s in range(len(SIDES)):
                    rows = found[s]
                    if len(rows) < 20:
                        continue
                    index = [r[0] for r in rows]
                    where = dict(zip(index, (r[1] for r in rows)))
                    if t == REFERENCE:
                        reference[c, j, s] = where
                    known = reference.get((c, j, s), {})
                    # each crossing against the same row or column at frame 15
                    delta = [where[i] - known[i] for i in index if i in known]
                    if len(delta) < 20:
                        continue
                    position[t, c, j, s] = np.median(delta)
                    sigma[t, c, j, s] = np.median([r[2] for r in rows])
                    profile[t, c, j, s] = np.nanmedian([r[3] for r in rows], axis=0)
        print(f"frame {t}: sigma {np.round(sigma[t, :, 0], 2).tolist()}", flush=True)

    np.savez(out, position=position, sigma=sigma, profile=profile, offsets=OFFSETS)
    print(f"-> {out}")


def _crossings(frame: np.ndarray, px: np.ndarray, py: np.ndarray) -> dict:
    """Fit every row and column crossing a side of a window, as measure_edges does."""
    fraction, skip, stride, margin = 0.35, 25, 2, 40
    xc, yc = px.mean(), py.mean()
    rx, ry = 0.5 * np.ptp(px), 0.5 * np.ptp(py)
    found = {s: [] for s in range(len(SIDES))}
    for dy in range(-int(fraction * ry), int(fraction * ry)):
        y = int(round(yc + dy))
        if abs(dy) < skip or y < 0 or y >= frame.shape[0]:
            continue
        for s, guess in zip((0, 1), _outline._crossings(px, py, y)):
            if np.isfinite(guess) and margin <= guess <= frame.shape[1] - margin:
                r = _fit_edge(frame[y], guess)
                if r is not None:
                    found[s].append((y,) + r)
    for dx in range(-int(fraction * rx), int(fraction * rx), stride):
        x = int(round(xc + dx))
        if x < 0 or x >= frame.shape[1]:
            continue
        for s, guess in zip((2, 3), _outline._crossings(py, px, x)):
            if np.isfinite(guess) and margin <= guess <= frame.shape[0] - margin:
                r = _fit_edge(frame[:, x], guess)
                if r is not None:
                    found[s].append((x,) + r)
    return found


def _level(profile: np.ndarray, offsets: np.ndarray, level: float) -> float:
    """Find where a stacked profile crosses a level, nearest the fitted crossing."""
    if not np.all(np.isfinite(profile)):
        return np.nan
    i = np.flatnonzero((profile[:-1] < level) & (profile[1:] >= level))
    if i.size == 0:
        return np.nan
    i = i[np.argmin(np.abs(offsets[i]))]
    step = (level - profile[i]) / (profile[i + 1] - profile[i])
    return float(offsets[i] + step * (offsets[i + 1] - offsets[i]))


def report_edges(path: pathlib.Path, out: pathlib.Path) -> None:
    """Print and draw the motion, the sharpness and the shape of every edge."""
    z = np.load(path)
    position, sigma, profile, offsets = (
        z[k] for k in ("position", "sigma", "profile", "offsets")
    )
    num_time, num_channel = position.shape[:2]
    frames = np.arange(num_time)
    names = list(LINES)

    skew = np.full(position.shape, np.nan)
    for index in np.ndindex(position.shape):
        lo, mid, hi = (_level(profile[index], offsets, v) for v in (0.1, 0.5, 0.9))
        skew[index] = (hi - mid) - (mid - lo)

    print("per edge: motion per 15 frames [px]; sigma early, middle, late [px];")
    print("skew of the stacked profile, 50-90% minus 10-50% [px]")
    for c in range(num_channel):
        for j, name in enumerate(names):
            parts = []
            for s, side in enumerate(SIDES):
                v = sigma[:, c, j, s]
                if not np.isfinite(v).any():
                    parts.append(f"{side} not measured")
                    continue
                parts.append(
                    f"{side} {_trend(frames, position[:, c, j, s]):+.2f}; "
                    f"{np.nanmean(v[:3]):.1f} {np.nanmean(v[14:17]):.1f} "
                    f"{np.nanmean(v[24:27]):.1f}; {np.nanmedian(skew[:, c, j, s]):+.1f}"
                )
            print(f"  channel {c} {name}: " + " | ".join(parts))
    print(
        "the two lines on the motion of the same side, O V minus He I [px / 15 frames]:"
    )
    for c in range(num_channel):
        d = [
            _trend(frames, position[:, c, 1, s]) - _trend(frames, position[:, c, 0, s])
            for s in range(len(SIDES))
        ]
        print(
            f"  channel {c}: "
            + ", ".join(f"{side} {v:+.2f}" for side, v in zip(SIDES, d))
        )

    styles = dict(zip(names, ("-", "--")))
    fig, axes = plt.subplots(
        3, num_channel, figsize=(3.4 * num_channel, 8.6), constrained_layout=True
    )
    for c in range(num_channel):
        for j, name in enumerate(names):
            for s, side in enumerate(SIDES):
                kwargs = dict(color=CHANNEL_COLORS[s], lw=1.2, ls=styles[name])
                axes[0, c].plot(
                    frames, position[:, c, j, s], label=f"{side}, {name}", **kwargs
                )
                axes[1, c].plot(frames, sigma[:, c, j, s], **kwargs)
        axes[0, c].set_title(
            f"channel {c}: edge motion against frame {REFERENCE}", fontsize=9
        )
        axes[1, c].set_title("edge sharpness (error-function sigma)", fontsize=9)
        axes[1, c].set_xlabel("frame")
        for s in range(len(SIDES)):
            for t0, ls in ((0, ":"), (14, "-"), (24, "--")):
                mean = np.nanmean(profile[t0 : t0 + 3, c, 0, s], axis=0)
                axes[2, c].plot(offsets, mean, color=CHANNEL_COLORS[s], lw=1.0, ls=ls)
        axes[2, c].set_xlim(-8, 8)
        axes[2, c].set_title(
            "He I stacked edge profile, dark to lit\n"
            "dotted: frames 0-2, solid: 14-16, dashed: 24-26",
            fontsize=8,
        )
        axes[2, c].set_xlabel("pixels from the fitted crossing")
    axes[0, 0].set_ylabel("sensor pixels")
    axes[1, 0].set_ylabel("pixels")
    axes[0, 0].legend(fontsize=6, frameon=False, ncol=2)
    fig.savefig(out, dpi=160)
    print(f"-> {out}")


if __name__ == "__main__":
    command = sys.argv[1]
    paths = [pathlib.Path(a) for a in sys.argv[2:]]
    if command == "image":
        image(*paths)
    elif command == "edges":
        edges(*paths)
    elif command == "report-image":
        report_image(*paths)
    elif command == "report-edges":
        report_edges(*paths)
    else:
        raise SystemExit(__doc__)
