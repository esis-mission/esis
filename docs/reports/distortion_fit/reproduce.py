#!/usr/bin/env python3
"""
Reproduce the committed distortion fit of ESIS-I from the model and the data.

The fit is the stages of
:func:`esis.flights.f1.optics.fit_distortion_reference`.  The first, the
absolute fit of each channel, is hours on one GPU per channel, so the
stages are split across jobs.  Run, in order::

    python reproduce.py channel <c> <directory>   # absolute fit of channel c
    python reproduce.py edges <directory>         # window edges of every frame
    python reproduce.py combine <directory>       # outline + shared + alignment
    python reproduce.py pointing <t> <directory>  # per-frame pointing of frame t
    python reproduce.py gather <directory>        # pointing rows -> ECSV
    python reproduce.py accept <directory>        # score the fit on held-out frames

Environment: ESIS_DEVICE (default ``cuda``; empty for the host),
ESIS_WORKERS (default 6), ESIS_MERIT (``correlation`` or
``least_squares``), and to depart from the committed configuration,
ESIS_FREE_ABSOLUTE and ESIS_FREE_SHARED (colon-separated field names).
"""

import logging
import os
import pathlib
import re
import sys

import astropy.table
import esis
from esis.flights.f1.optics._fits import _fits

# a dependency configures the root logger at INFO, which lets numba report
# every device allocation
logging.getLogger("numba").setLevel(logging.WARNING)

DEVICE = os.environ.get("ESIS_DEVICE", "cuda") or None
WORKERS = int(os.environ.get("ESIS_WORKERS", "6"))

# ESIS_MERIT selects the comparison the fit maximizes: "correlation" (the
# default) or "least_squares", which also fits the degradation of each channel
MERIT = os.environ.get("ESIS_MERIT", "correlation")


def _names(variable: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read a colon-separated list of field names from the environment."""
    value = os.environ.get(variable)
    if not value:
        return default
    # a comma would be split by Slurm's --export, so colons separate the names
    return tuple(n for n in re.split(r"[:,\s]+", value) if n)


FREE_ABSOLUTE = _names("ESIS_FREE_ABSOLUTE", _fits._FREE_ABSOLUTE)
FREE_SHARED = _names("ESIS_FREE_SHARED", _fits._FREE_SHARED)


def _channels(directory: pathlib.Path) -> list[esis.optics.DistortionParameters]:
    return [
        esis.optics.DistortionParameters.from_file(directory / f"channel_{c}.ecsv")
        for c in range(4)
    ]


def channel(c: int, directory: pathlib.Path) -> None:
    """Fit one channel absolutely and save it as a one-row ECSV."""
    instrument = _fits._base(None)
    scene, observation = _fits._frame(15, 401)
    log = _fits._logger(directory, f"channel_{c}")
    channel = instrument[dict(channel=c)]
    p0 = esis.optics.DistortionParameters.from_instrument(channel)
    merit = esis.optics.LinearMerit(
        instrument=channel,
        parameters=p0,
        scene=scene,
        observation=observation[dict(channel=c)],
        device=DEVICE,
        merit=MERIT,
    )
    log("free: " + ", ".join(FREE_ABSOLUTE))
    fitted = esis.optics.fit_distortion(
        objective=merit,
        parameters=p0,
        bounds=esis.flights.f1.optics.distortion_fit_bounds(p0),
        workers=WORKERS,
        free=FREE_ABSOLUTE,
        popsize=15,
        maxiter=80,
        tol=0.0,
        seed=0,
        log=log,
    )
    fitted.to_file(
        directory / f"channel_{c}.ecsv",
        metadata=dict(
            channel=c,
            stage="absolute",
            free=list(FREE_ABSOLUTE),
            merit=MERIT,
            popsize=15,
            maxiter=80,
            tol=0.0,
            seed=0,
            versions=_fits._versions(),
        ),
    )
    log(
        f"channel {c}: correlation {merit.correlation(fitted):.4f}, "
        f"least-squares score {merit.score(fitted):.4f}"
    )


def edges(directory: pathlib.Path) -> None:
    """Measure the window edges of every frame near the absolute fits' outlines."""
    _fits.measure_window_edges(
        parameters=_channels(directory),
        directory=directory,
        path=directory / "window_edges.ecsv",
        path_drift=directory / "window_drift.ecsv",
    )


def combine(directory: pathlib.Path) -> None:
    """Run the outline, shared and internal stages from the saved per-channel fits."""
    path_edges = directory / "window_edges.ecsv"
    _fits.fit_distortion_reference(
        device=DEVICE,
        workers=WORKERS,
        merit=MERIT,
        channels=(),
        parameters=_channels(directory),
        free_absolute=FREE_ABSOLUTE,
        free_shared=FREE_SHARED,
        edges=path_edges if path_edges.exists() else None,
        path=directory / "distortion_reference.ecsv",
        directory=directory,
    )


def pointing(t: int, directory: pathlib.Path) -> None:
    """Fit the pointing of one frame and save it as a one-row ECSV."""
    parameters = esis.optics.DistortionParameters.from_file(
        directory / "distortion_reference.ecsv"
    )
    path_drift = directory / "window_drift.ecsv"
    _fits.fit_distortion_pointing(
        instrument=parameters.to_instrument(_fits._base(None)),
        device=DEVICE,
        merit=MERIT,
        frames=(t,),
        drift=path_drift if path_drift.exists() else None,
        path=directory / f"pointing_{t:02d}.ecsv",
        directory=directory,
    )


def gather(directory: pathlib.Path) -> None:
    """Gather the per-frame pointing rows into the committed table."""
    tables = [
        astropy.table.QTable.read(p, format="ascii.ecsv")
        for p in sorted(directory.glob("pointing_*.ecsv"))
    ]
    table = astropy.table.vstack(tables, metadata_conflicts="silent")
    table.sort("frame")
    table.meta.update(tables[0].meta)
    table = _fits.pointing_relative(table, frame=15)
    table.write(
        directory / "distortion_pointing.ecsv", format="ascii.ecsv", overwrite=True
    )
    print(f"{len(table)} frames -> {directory / 'distortion_pointing.ecsv'}")


def accept(directory: pathlib.Path) -> None:
    """Score the reference on frames across the flight, without an inversion."""
    path_edges = directory / "window_edges.ecsv"
    _fits.acceptance(
        reference=esis.optics.DistortionParameters.from_file(
            directory / "distortion_reference.ecsv"
        ),
        pointing=astropy.table.QTable.read(
            directory / "distortion_pointing.ecsv", format="ascii.ecsv"
        ),
        edges=(
            astropy.table.QTable.read(path_edges, format="ascii.ecsv")
            if path_edges.exists()
            else None
        ),
        device=DEVICE,
        merit=MERIT,
        path=directory / "acceptance.ecsv",
        directory=directory,
    )


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "channel":
        channel(int(sys.argv[2]), pathlib.Path(sys.argv[3]))
    elif command == "edges":
        edges(pathlib.Path(sys.argv[2]))
    elif command == "combine":
        combine(pathlib.Path(sys.argv[2]))
    elif command == "pointing":
        pointing(int(sys.argv[2]), pathlib.Path(sys.argv[3]))
    elif command == "gather":
        gather(pathlib.Path(sys.argv[2]))
    elif command == "accept":
        accept(pathlib.Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
