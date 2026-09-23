#!/usr/bin/env python3
"""
Render the model against the data for every frame of the flight, to blink.

    python blink.py render <directory> [device]   # one npz per frame
    python blink.py page <directory> <out.html>   # a self-contained page

The render images the AIA proxy scene through the fitted channel with each
frame's pointing and window drift applied, standardizes the image and the
Level-1 frame alike, and block-averages both.  The page embeds them as
JPEGs and blinks between model and data under the keyboard.
"""

import base64
import copy
import io
import json
import logging
import pathlib
import sys

import numpy as np
import astropy.table
import esis
from esis.flights.f1.optics._fits import _fits

logging.getLogger("numba").setLevel(logging.WARNING)

BLOCK = 4
"""The block-averaging factor of the rendered images."""


def _standardize(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shift and scale an array to zero mean and unit deviation inside a mask."""
    values = a[mask]
    return (a - values.mean()) / values.std()


def _block(a: np.ndarray, block: int) -> np.ndarray:
    ny, nx = (a.shape[0] // block) * block, (a.shape[1] // block) * block
    return a[:ny, :nx].reshape(ny // block, block, nx // block, block).mean(axis=(1, 3))


def render(directory: pathlib.Path, device: None | str) -> None:
    """Image every frame through the fit and save it beside the data."""
    reference = esis.optics.DistortionParameters.from_file(
        directory / "distortion_reference.ecsv"
    )
    pointing = astropy.table.QTable.read(
        directory / "distortion_pointing.ecsv", format="ascii.ecsv"
    )
    instrument = _fits._base(None)
    num_channel = instrument.camera.channel.shape["channel"]
    axes = ("detector_y", "detector_x")
    for row in pointing:
        t = int(row["frame"])
        scene, observation = _fits._frame(t, 401)
        model, data, lit = [], [], []
        for c in range(num_channel):
            p = copy.copy(reference[dict(channel=c)])
            p.pitch = p.pitch + row["pitch"]
            p.yaw = p.yaw + row["yaw"]
            p.roll = p.roll + row["roll"]
            for name in ("yaw_grating", "pitch_grating"):
                if name in pointing.colnames:
                    setattr(p, name, getattr(p, name) + row[name][c])
            channel = instrument[dict(channel=c)]
            merit = esis.optics.LinearMerit(
                instrument=channel,
                parameters=p,
                scene=scene,
                observation=observation[dict(channel=c)],
                device=device,
            )
            image = np.asarray(merit.image(p).ndarray_aligned(axes), dtype=float)
            frame = np.asarray(merit.observation.ndarray_aligned(axes), dtype=float)
            inside = image > 1e-4 * image.max()
            model.append(_block(_standardize(image, inside), BLOCK))
            data.append(_block(_standardize(frame, inside), BLOCK))
            lit.append(_block(inside.astype(float), BLOCK))
        np.savez_compressed(
            directory / f"blink_{t:03d}.npz",
            model=np.stack(model).astype(np.float16),
            data=np.stack(data).astype(np.float16),
            inside=np.stack(lit).astype(np.float16),
            pointing=np.array(
                [
                    row["pitch"].to_value("arcsec"),
                    row["yaw"].to_value("arcsec"),
                    row["roll"].to_value("deg"),
                ]
            ),
        )
        print(f"frame {t}: rendered", flush=True)


def _jpeg(a: np.ndarray, inside: np.ndarray, limit: float = 3.0) -> str:
    """Encode a standardized image as a base64 JPEG, grey outside the windows."""
    from PIL import Image

    scaled = np.clip((a + limit) / (2 * limit), 0, 1)
    scaled = np.where(inside > 0.5, scaled, 0.5)
    image = Image.fromarray((255 * scaled[::-1]).astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode()


def page(directory: pathlib.Path, out: pathlib.Path) -> None:
    """Write a self-contained page that blinks the model against the data."""
    frames = {}
    for path in sorted(directory.glob("blink_*.npz")):
        t = int(path.stem.split("_")[1])
        with np.load(path) as f:
            model = f["model"].astype(float)
            data = f["data"].astype(float)
            inside = f["inside"].astype(float)
            pointing = f["pointing"].tolist()
        frames[t] = dict(
            pointing=pointing,
            model=[_jpeg(model[c], inside[c]) for c in range(model.shape[0])],
            data=[_jpeg(data[c], inside[c]) for c in range(data.shape[0])],
        )
    payload = json.dumps(frames)
    html = (
        pathlib.Path(__file__)
        .with_name("blink_template.html")
        .read_text(encoding="utf-8")
    )
    out.write_text(html.replace("/*FRAMES*/", payload), encoding="utf-8")
    print(f"{len(frames)} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "render":
        render(pathlib.Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
    elif command == "page":
        page(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
