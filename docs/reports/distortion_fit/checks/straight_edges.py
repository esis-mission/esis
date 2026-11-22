#!/usr/bin/env python3
"""
Refit a finished pass with the drift of the windows read from their straight edges.

A check of the choice the fit makes in
:func:`esis.flights.f1.optics.measure_window_edges`, which reads the drift
of the windows from a step fit to every row and column rather than from the
straight edge fit to each side.  This reads it from the straight edges
instead, refers it to the reference frame the same way, and refits the
sector focus and the pointing of every frame with it, from the pass's
reference; the registration of the channels the acceptance measures is then
compared.  On the committed fit the straight edges leave the channels
registered worse than the steps, at 0.117 against 0.106 px rms at He I and
0.108 against 0.074 px at O V, and the two sides of one window move apart
by 0.14--0.39 px rms with them against 0.06--0.20 px with the steps.
From this directory::

    python straight_edges.py <pass directory> <output directory>

with the options of ``reproduce.py`` read from the same environment
variables.  The pointing of the 30 frames is fit in this process, so this
takes as long as those stages of a pass.
"""

from __future__ import annotations
import pathlib
import shutil
import sys

import numpy as np
import astropy.table
import named_arrays as na
import esis
from esis.flights.f1.optics._fits import _fits, _record

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import reproduce  # noqa: E402


def drift_from_edges(directory: pathlib.Path) -> astropy.table.QTable:
    """
    Measure the drift of the windows from the straight edges of their sides.

    Parameters
    ----------
    directory
        The pass, whose absolute fits of the channels place the outlines the
        edges are searched near, and whose drift table names the frames.
    """
    steps = astropy.table.QTable.read(
        directory / "window_drift.ecsv", format="ascii.ecsv"
    )
    frames = steps.meta["frames"]
    lines = steps.meta["lines"]
    parameters = reproduce._channels(directory)
    instrument = _fits._base(None)
    level_1 = esis.flights.f1.data.level_1()
    axes = ("time", "channel", "detector_y", "detector_x")
    data = np.asarray(na.value(level_1.outputs).ndarray_aligned(axes), dtype=float)
    scene, observation = _fits._frame(
        reproduce.FRAME_REFERENCE, reproduce.options.num_scene
    )
    tables = []
    for c in range(reproduce.NUM_CHANNEL):
        channel = instrument[dict(channel=c)]
        merit = esis.optics.LinearMerit(
            instrument=channel,
            parameters=parameters[c],
            scene=scene,
            observation=observation[dict(channel=c)],
            seed=reproduce.options.seed,
        )
        linear = merit.linearize(parameters[c].to_instrument(channel))
        footprints = _fits._footprints(linear, channel.wavelength)
        for t in frames:
            table = esis.optics.measure_edges(data[t, c], footprints)
            table = table[np.isin(table["line"], lines)]
            table["frame"] = t
            table["channel"] = c
            tables.append(table)
    drift = _fits.window_drift(astropy.table.vstack(tables), reproduce.NUM_CHANNEL)
    edges = astropy.table.QTable.read(
        directory / "window_edges.ecsv", format="ascii.ecsv"
    )
    _, drift = _fits._refer_to_frame(
        edges, drift, reproduce.FRAME_REFERENCE, reproduce.DRIFT_DEGREE
    )
    drift.meta.update(steps.meta)
    drift.meta["description"] = (
        "The motion of each window's sides from the reference frame, per "
        "frame, from the crossings of the straight edge fit to each side."
    )
    return drift


def main(source: pathlib.Path, directory: pathlib.Path) -> None:
    """Refit the pass with the drift from the straight edges, and compare."""
    directory.mkdir(parents=True, exist_ok=True)
    names = [f"channel_{c}.ecsv" for c in range(reproduce.NUM_CHANNEL)]
    names += ["distortion_reference.ecsv", "window_edges.ecsv"]
    if (source / "field_stop_input.ecsv").exists():
        names.append("field_stop_input.ecsv")
    for name in names:
        shutil.copyfile(source / name, directory / name)
    reproduce._field_stop(directory)
    path = directory / "window_drift.ecsv"
    if not path.exists():
        drift_from_edges(source).write(path, format="ascii.ecsv")
    stage = reproduce._stage
    stage("defocus", directory / "defocus.ecsv", reproduce.defocus, directory)
    for t in range(reproduce.NUM_FRAME):
        output = directory / f"pointing_{t:02d}.ecsv"
        stage(f"pointing {t}", output, reproduce.pointing, t, directory)
    stage("gather", directory / "distortion_pointing.ecsv", reproduce.gather, directory)
    stage("accept", directory / "acceptance.ecsv", reproduce.accept, directory)

    for name, d in (("steps", source), ("straight edges", directory)):
        rms = _record.registration_rms(
            astropy.table.QTable.read(d / "acceptance.ecsv", format="ascii.ecsv"),
            astropy.table.QTable.read(d / "window_drift.ecsv", format="ascii.ecsv"),
        )
        print(
            f"drift from the {name}: registration "
            + ", ".join(f"{line} {v:.3f} px" for line, v in rms.items())
        )


if __name__ == "__main__":
    reproduce.configure()
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
