#!/usr/bin/env python3
"""
Put the channels on the sky through the fit, to blink and difference them.

    python coalign.py render <directory> [frame ...]   # one npz per frame
    python coalign.py page <directory> <out.html>      # a self-contained page

The render reads every channel's Level-1 frame onto a common sky grid
through its fitted distortion at each alignment line, inside the window
that line illuminates, with the frame's pointing and window drift applied,
as the internal alignment sees the channels.  It normalizes each image to a
mean of one over the pixels every channel covers, measures the median tile
shift between every pair of channels, and records the direction on the sky
in which each channel disperses.  The page blinks any two channels and shows
their difference, plain or high-passed, with the dispersion directions
drawn: a Doppler-shifted feature is displaced along that arrow, so in a
difference it shows as a paired excess and deficit while stationary
structure cancels.

Environment: ESIS_NUM_SCENE (sampling of the AIA scene whose extent sets the
sky grid, default 401) and ESIS_BLOCK (block-averaging of the render,
default 3) for the render, whose frames default to the reference frame;
ESIS_PAGE_JPEG (a quality; empty for lossless PNG) and ESIS_PAGE_BLOCK (a
further block-averaging factor) for the page, trading fidelity for the
number of frames that fit it.  ESIS_COREGISTERED=1 renders through the
co-registered copy of the pointing table (``reproduce.py coregister``),
with ``_coregistered`` in the file names.
"""

import base64
import io
import json
import logging
import os
import pathlib
import sys

import numpy as np
import astropy.table
import astropy.units as u
import named_arrays as na
import esis
from esis.flights.f1.optics._fits import _fits
from esis.optics._distortions import _alignment

logging.getLogger("numba").setLevel(logging.WARNING)

BLOCK = int(os.environ.get("ESIS_BLOCK", "3"))
"""The block-averaging factor of the rendered images; 3 is 1.6 arcsec, two pixels."""

NUM_SKY = 1600
"""The samples along each axis of the sky grid, as in the alignment."""

NUM_SCENE = int(os.environ.get("ESIS_NUM_SCENE", "401"))
"""The sampling of the AIA scene, whose extent sets the sky grid."""

ANCHOR = 1
"""The channel the alignment anchors on, the default to difference against."""

SUFFIX = (
    "_coregistered"
    if os.environ.get("ESIS_COREGISTERED", "") not in ("", "0", "false", "no")
    else ""
)
"""Marks a render through the co-registered pointing table, in its file names."""

HIGHPASS_SIGMA = 40.0
"""
The Gaussian sigma of the page's high-pass, in sky samples of 0.52 arcsec.

About 21 arcsec: smaller than a supergranule, so only the vignetting and
effective-area gradients are removed.  The shift measurement uses the
alignment's own, sharper filter.
"""

PAGE_JPEG = os.environ.get("ESIS_PAGE_JPEG", "")
"""The JPEG quality of the page's images, or empty for lossless PNG."""

PAGE_BLOCK = int(os.environ.get("ESIS_PAGE_BLOCK", "1"))
"""A further block-averaging factor applied when the page is built."""


def _block(
    a: np.ndarray, valid: np.ndarray, block: int
) -> tuple[np.ndarray, np.ndarray]:
    """Block-average an image over its valid pixels, and the fraction valid."""
    ny, nx = (a.shape[0] // block) * block, (a.shape[1] // block) * block
    shape = (ny // block, block, nx // block, block)
    v = valid[:ny, :nx].reshape(shape).astype(float)
    s = np.where(valid[:ny, :nx], a[:ny, :nx], 0.0).reshape(shape)
    count = v.sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s.sum(axis=(1, 3)) / count
    return np.where(count > 0, mean, 0.0), count / block**2


def _parameters(
    reference: esis.optics.DistortionParameters,
    pointing: None | astropy.table.QTable,
    t: int,
    c: int,
) -> esis.optics.DistortionParameters:
    """Apply a frame's pointing and drift to one channel's reference parameters."""
    if pointing is None or t not in np.asarray(pointing["frame"]):
        return _fits._channel(reference, c)
    return _fits.frame_parameters(reference, pointing, t, c)


def render(directory: pathlib.Path, frames: tuple[int, ...]) -> None:
    """Sample every channel's frame on the sky at each line and save it."""
    reference = esis.optics.DistortionParameters.from_file(
        directory / "distortion_reference.ecsv"
    )
    path_pointing = directory / f"distortion_pointing{SUFFIX}.ecsv"
    pointing = (
        astropy.table.QTable.read(path_pointing, format="ascii.ecsv")
        if path_pointing.exists()
        else None
    )
    instrument = _fits._base(None)
    num_channel = instrument.camera.channel.shape["channel"]
    wavelengths = _fits._wavelengths_alignment()
    times = _fits._frame_times()
    axes = ("detector_y", "detector_x")
    axis_sky = ("sky_x", "sky_y")
    for t in frames:
        scene, observation = _fits._frame(t, NUM_SCENE)
        sky = _alignment.sky_grid(scene.inputs.position, NUM_SKY, axis_sky)
        position = scene.inputs.position
        extent = [
            float(position.x.min().ndarray.to_value(u.arcsec)),
            float(position.x.max().ndarray.to_value(u.arcsec)),
            float(position.y.min().ndarray.to_value(u.arcsec)),
            float(position.y.max().ndarray.to_value(u.arcsec)),
        ]
        step = (extent[1] - extent[0]) / (NUM_SKY - 1)
        frames_l1 = [
            np.asarray(
                na.value(observation[dict(channel=c)]).ndarray_aligned(axes),
                dtype=float,
            )
            for c in range(num_channel)
        ]
        linears = []
        for c in range(num_channel):
            p = _parameters(reference, pointing, t, c)
            model = p.to_instrument(instrument[dict(channel=c)])
            linears.append(
                model.system.linearize(
                    wavelength=model.wavelength, degree=2, field_stop=True
                )
            )
        distortions = [system.distortion for system in linears]

        images, highpassed, valid, scales = [], [], [], []
        shifts, dispersions = {}, {}
        for name, wavelength in wavelengths.items():
            sampled, coordinates = [], []
            for c in range(num_channel):
                xc, yc = _alignment.sensor_coordinates(
                    distortions[c], sky, wavelength, axis_sky
                )
                coordinates.append((xc, yc))
                # only the window this line illuminates, as the alignment reads it
                mask = _alignment.window_mask(
                    linears[c].footprint(wavelength), frames_l1[c].shape
                )
                frame = np.where(mask, frames_l1[c], np.nan)
                sampled.append(_alignment.sample_on_sky(frame, xc, yc))
            # detector pixels per sky sample, from the anchor's mapping
            xa, ya = coordinates[ANCHOR]
            scales.append(
                float(
                    np.nanmedian(
                        np.hypot(np.gradient(xa, axis=0), np.gradient(ya, axis=0))
                    )
                )
            )
            common = np.all([np.isfinite(s) for s in sampled], axis=0)
            # for the shift measurement, as the alignment sees the channels
            standardized = []
            for s in sampled:
                values = s[common]
                standardized.append((s - values.mean()) / values.std())
            # for the page: every channel at a mean of one over the pixels
            # they all cover, so that a difference is a fraction of the mean
            normalized = [s / s[common].mean() for s in sampled]
            filtered = []
            for s in normalized:
                own = np.isfinite(s)
                h = _alignment.highpass(np.where(own, s, 1.0), sigma=HIGHPASS_SIGMA)
                filtered.append(np.where(own, h, np.nan))
            # the direction on the sky along which this channel disperses
            dispersion = []
            for c in range(num_channel):
                origin = na.Cartesian2dVectorArray(
                    x=na.ScalarArray(np.zeros(1) * u.arcsec, axes="s"),
                    y=na.ScalarArray(np.zeros(1) * u.arcsec, axes="s"),
                )
                where = [
                    distortions[c]
                    .distort(
                        na.SpectralPositionalVectorArray(
                            wavelength=wavelength + k * u.AA, position=origin
                        )
                    )
                    .position
                    for k in (0, 0.1)
                ]
                shift = np.array(
                    [
                        float(na.value(where[1].x - where[0].x).ndarray[0]),
                        float(na.value(where[1].y - where[0].y).ndarray[0]),
                    ]
                )
                x10, y10 = coordinates[c]
                # detector pixels per sky sample along each sky axis
                jac = np.array(
                    [
                        [
                            np.nanmedian(np.gradient(x10, axis=0)),
                            np.nanmedian(np.gradient(x10, axis=1)),
                        ],
                        [
                            np.nanmedian(np.gradient(y10, axis=0)),
                            np.nanmedian(np.gradient(y10, axis=1)),
                        ],
                    ]
                )
                on_sky = np.linalg.solve(jac, shift)
                dispersion.append((on_sky / np.linalg.norm(on_sky)).tolist())
            # the tile shifts between every pair, in detector pixels
            table = {}
            for a in range(num_channel):
                # the NaNs outside the windows tell the tiles what is valid
                measured = _alignment.measure_shifts(standardized, 6, a)
                for b, rows in measured.items():
                    if rows:
                        r = np.array(rows)
                        table[f"{b}-{a}"] = [
                            float(np.median(r[:, 2]) * scales[-1]),
                            float(np.median(r[:, 3]) * scales[-1]),
                        ]
            shifts[name] = table
            dispersions[name] = dispersion
            images.append(normalized)
            highpassed.append(filtered)
            valid.append([np.isfinite(s) for s in sampled])

        # crop to the pixels any channel covers at any line; the sky images
        # are indexed [sky_x, sky_y] and are stored transposed, [y, x]
        covered = np.any([np.any(v, axis=0) for v in valid], axis=0)
        rows = np.where(covered.any(axis=1))[0]
        cols = np.where(covered.any(axis=0))[0]
        window = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        crop = [
            extent[0] + step * rows[0],
            extent[0] + step * rows[-1],
            extent[2] + step * cols[0],
            extent[2] + step * cols[-1],
        ]

        def pack(stack):
            out, frac = [], []
            for line, per_channel in enumerate(stack):
                o, f = [], []
                for c, a in enumerate(per_channel):
                    mean, fraction = _block(
                        a[window].T, valid[line][c][window].T, BLOCK
                    )
                    o.append(mean)
                    f.append(fraction)
                out.append(o)
                frac.append(f)
            return np.array(out, dtype=np.float16), np.array(frac, dtype=np.float16)

        image_blocks, fraction = pack(images)
        highpass_blocks, _ = pack(highpassed)
        row = None
        if pointing is not None:
            rows_t = pointing[np.asarray(pointing["frame"]) == t]
            row = rows_t[0] if len(rows_t) else None
        np.savez_compressed(
            directory / f"coalign{SUFFIX}_{t:03d}.npz",
            images=image_blocks,
            highpass=highpass_blocks,
            valid=fraction,
            meta=json.dumps(
                dict(
                    frame=t,
                    time=times[t],
                    lines=list(wavelengths),
                    shifts=shifts,
                    dispersion=dispersions,
                    limits=dict(images=[0.0, 3.0], highpass=[-1.0, 1.0]),
                    highpass_sigma=HIGHPASS_SIGMA,
                    scale=scales,
                    extent=crop,
                    block=BLOCK,
                    step=step,
                    anchor=ANCHOR,
                    pointing=(
                        [
                            float(row["pitch"].to_value("arcsec")),
                            float(row["yaw"].to_value("arcsec")),
                            float(row["roll"].to_value("deg")),
                        ]
                        if row is not None
                        else None
                    ),
                )
            ),
        )
        summary = "; ".join(
            f"{name}: "
            + ", ".join(
                f"ch{c} {table[key][0]:+.2f},{table[key][1]:+.2f}"
                for c in range(num_channel)
                if (key := f"{c}-{ANCHOR}") in table
            )
            for name, table in shifts.items()
        )
        print(f"frame {t}: median shifts vs channel {ANCHOR} [px] {summary}")


def _png(a: np.ndarray, limits: tuple[float, float]) -> str:
    """Encode an image over the given range as a base64 grey PNG, or a JPEG."""
    from PIL import Image

    lower, upper = limits
    grey = np.clip((a - lower) / (upper - lower), 0, 1)
    image = Image.fromarray((255 * grey[::-1]).astype(np.uint8))
    buffer = io.BytesIO()
    if PAGE_JPEG:
        image.save(buffer, format="JPEG", quality=int(PAGE_JPEG))
    else:
        image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


def _reblock(
    a: np.ndarray, valid: np.ndarray, block: int
) -> tuple[np.ndarray, np.ndarray]:
    """Block-average a stored image further, over its valid pixels."""
    if block == 1:
        return a, valid
    return _block(a, valid > 0.5, block)


def _mask(valid: np.ndarray) -> str:
    """Encode where an image is valid as a base64 black-and-white PNG."""
    from PIL import Image

    image = Image.fromarray(np.where(valid[::-1] > 0.5, 255, 0).astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


def page(directory: pathlib.Path, out: pathlib.Path) -> None:
    """Write a self-contained page that blinks and differences the channels."""
    frames = {}
    times = None  # loaded only for renders that predate the stored times
    for path in sorted(directory.glob(f"coalign{SUFFIX}_[0-9]*.npz")):
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
            images = f["images"].astype(float)
            valid = f["valid"].astype(float)
        num_line, num_channel = images.shape[:2]
        if "time" not in meta:
            times = _fits._frame_times() if times is None else times
            meta["time"] = times[meta["frame"]]
        entry = dict(meta, block=meta["block"] * PAGE_BLOCK)
        encoded = {"images": [], "masks": []}
        for k in range(num_line):
            rows = {key: [] for key in encoded}
            for c in range(num_channel):
                a, v = _reblock(images[k, c], valid[k, c], PAGE_BLOCK)
                rows["images"].append(_png(a, meta["limits"]["images"]))
                rows["masks"].append(_mask(v))
            for key in encoded:
                encoded[key].append(rows[key])
        entry.update(encoded)
        frames[meta["frame"]] = entry
    payload = json.dumps(frames)
    html = (
        pathlib.Path(__file__)
        .with_name("coalign_template.html")
        .read_text(encoding="utf-8")
    )
    out.write_text(html.replace("/*FRAMES*/", payload), encoding="utf-8")
    print(f"{len(frames)} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "render":
        frames = tuple(int(a) for a in sys.argv[3:]) or (15,)
        render(pathlib.Path(sys.argv[2]), frames)
    elif command == "page":
        page(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
