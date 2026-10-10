#!/usr/bin/env python3
"""
Reproduce the committed distortion fit of ESIS-I from the model and the data.

The stages of :func:`esis.flights.f1.optics.fit_distortion_reference` are
split across jobs, since the absolute fit alone is hours per channel.  The
chain, in order::

    python reproduce.py channel <c> <directory>   # absolute fit of channel c
    python reproduce.py edges <directory>         # window edges of every frame
    python reproduce.py combine <directory>       # outline + shared + alignment
    python reproduce.py defocus <directory>       # focus of each sector, per frame
    python reproduce.py pointing <t> <directory>  # pointing of frame t
    python reproduce.py gather <directory>        # pointing rows -> ECSV
    python reproduce.py accept <directory>        # score the fit on held-out frames

Three more commands work on a finished reference::

    python reproduce.py polish <directory>      # shared + alignment, saved start
    python reproduce.py align <directory>       # alignment only, in place
    python reproduce.py coregister <directory>  # empirical channel offsets, a check

``polish`` starts from ``start_reference.ecsv`` in the directory and carries
its outline record into the result.  ``coregister`` measures what the focus
of each sector leaves and writes ``coregistration.ecsv`` beside a copy of
the pointing table with the offsets applied,
``distortion_pointing_coregistered.ecsv``; with ESIS_COREGISTERED=1,
``accept`` (and ``coalign.py``) read that copy and write their results with
a ``_coregistered`` suffix, so the two models can be compared.

Environment: ESIS_DEVICE (default ``cuda``; ``cpu`` or empty for the host),
ESIS_WORKERS (default 6), ESIS_NUM_SCENE (sampling of the AIA scene,
default 401), ESIS_SEED (seed of the capture, default 0), ESIS_PRIMARY=0 to
hold the primary at nominal, ESIS_DRIFT_DEGREE (polynomial that smooths the
window drift, default 3; empty for none), ESIS_DEFOCUS_DEGREE (polynomial
through each sector's focus, default 2), ESIS_COREGISTRATION_DEGREE (the
same for ``coregister``, default 2), ESIS_COREGISTERED=1 to score the
co-registered copy of the pointing table, and to depart from the committed
configuration, ESIS_MERIT (``correlation`` or ``least_squares``),
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

DEVICE = os.environ.get("ESIS_DEVICE", "cuda")
DEVICE = None if DEVICE.strip().lower() in ("", "cpu", "none") else DEVICE
WORKERS = int(os.environ.get("ESIS_WORKERS", "6"))
MERIT = os.environ.get("ESIS_MERIT", "correlation")
PRIMARY = os.environ.get("ESIS_PRIMARY", "1") not in ("0", "false", "no")
# 401 samples is 2.1 arcsec, about 2.8 ESIS pixels
NUM_SCENE = int(os.environ.get("ESIS_NUM_SCENE", "401"))
SEED = int(os.environ.get("ESIS_SEED", "0"))
DRIFT_DEGREE = os.environ.get("ESIS_DRIFT_DEGREE", "3")
DRIFT_DEGREE = int(DRIFT_DEGREE) if DRIFT_DEGREE else None
COREGISTRATION_DEGREE = int(os.environ.get("ESIS_COREGISTRATION_DEGREE", "2"))
DEFOCUS_DEGREE = int(os.environ.get("ESIS_DEFOCUS_DEGREE", "2"))
COREGISTERED = os.environ.get("ESIS_COREGISTERED", "") not in ("", "0", "false", "no")
SUFFIX = "_coregistered" if COREGISTERED else ""


def _pointing(directory: pathlib.Path) -> pathlib.Path:
    """Name the pointing table to score: the chain's, or the co-registered copy."""
    return directory / f"distortion_pointing{SUFFIX}.ecsv"


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
    scene, observation = _fits._frame(15, NUM_SCENE)
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
        seed=SEED,
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
            seed=SEED,
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
        num_scene=NUM_SCENE,
        device=DEVICE,
        workers=WORKERS,
        merit=MERIT,
        channels=(),
        parameters=_channels(directory),
        free_absolute=FREE_ABSOLUTE,
        free_shared=FREE_SHARED,
        edges=path_edges if path_edges.exists() else None,
        primary=PRIMARY,
        path=directory / "distortion_reference.ecsv",
        directory=directory,
    )


def polish(directory: pathlib.Path) -> None:
    """Rerun the shared polish and the alignment from a saved reference."""
    path_start = directory / "start_reference.ecsv"
    start = astropy.table.QTable.read(path_start, format="ascii.ecsv")
    parameters = esis.optics.DistortionParameters.from_file(path_start)
    path = directory / "distortion_reference.ecsv"
    _fits.fit_distortion_reference(
        num_scene=NUM_SCENE,
        device=DEVICE,
        workers=WORKERS,
        merit=MERIT,
        channels=(),
        parameters=[_fits._channel(parameters, c) for c in range(4)],
        free_absolute=FREE_ABSOLUTE,
        free_shared=FREE_SHARED,
        edges=None,
        primary=False,
        path=path,
        directory=directory,
    )
    # the windows were placed by the start's outline stage: carry its record
    result = astropy.table.QTable.read(path, format="ascii.ecsv")
    for key in ("absolute", "outline"):
        result.meta["stages"][key] = start.meta.get("stages", {}).get(key)
        result.meta["scores"][key] = start.meta.get("scores", {}).get(key)
    result.meta["stages"]["primary"] = start.meta.get("stages", {}).get("primary")
    result.meta["start"] = dict(
        provenance=start.meta.get("provenance"),
        date=start.meta.get("date"),
        commits=start.meta.get("commits"),
        polish=f"shared polish and alignment rerun at num_scene={NUM_SCENE}",
    )
    result.write(path, format="ascii.ecsv", overwrite=True)


def align(directory: pathlib.Path) -> None:
    """Rerun only the internal alignment of the directory's reference, in place."""
    path = directory / "distortion_reference.ecsv"
    _fits.realign_distortion_reference(
        reference=path,
        num_scene=NUM_SCENE,
        free_shared=FREE_SHARED,
        path=path,
        directory=directory,
    )


def defocus(directory: pathlib.Path) -> None:
    """Measure the focus of each sector of the primary through the flight."""
    path_drift = directory / "window_drift.ecsv"
    _fits.fit_defocus_history(
        reference=directory / "distortion_reference.ecsv",
        num_scene=NUM_SCENE,
        drift=path_drift if path_drift.exists() else None,
        drift_degree=DRIFT_DEGREE,
        degree=DEFOCUS_DEGREE,
        path=directory / "defocus.ecsv",
        directory=directory,
    )


def pointing(t: int, directory: pathlib.Path) -> None:
    """Fit the pointing of one frame and save it as a one-row ECSV."""
    parameters = esis.optics.DistortionParameters.from_file(
        directory / "distortion_reference.ecsv"
    )
    path_drift = directory / "window_drift.ecsv"
    path_defocus = directory / "defocus.ecsv"
    _fits.fit_distortion_pointing(
        instrument=parameters.to_instrument(_fits._base(None)),
        num_scene=NUM_SCENE,
        device=DEVICE,
        merit=MERIT,
        frames=(t,),
        drift=path_drift if path_drift.exists() else None,
        drift_degree=DRIFT_DEGREE,
        defocus=path_defocus if path_defocus.exists() else None,
        relative=False,
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
    table.meta["environment"] = _fits._environment()
    table = _fits.pointing_relative(table, frame=15)
    table.write(
        directory / "distortion_pointing.ecsv", format="ascii.ecsv", overwrite=True
    )
    print(f"{len(table)} frames -> {directory / 'distortion_pointing.ecsv'}")


def coregister(directory: pathlib.Path) -> None:
    """Measure the channel offsets the sector focus leaves; the chain is untouched."""
    pointing = astropy.table.QTable.read(
        directory / "distortion_pointing.ecsv", format="ascii.ecsv"
    )
    table = _fits.fit_coregistration(
        reference=directory / "distortion_reference.ecsv",
        pointing=pointing,
        num_scene=NUM_SCENE,
        degree=COREGISTRATION_DEGREE,
        path=directory / "coregistration.ecsv",
        directory=directory,
    )
    path = directory / "distortion_pointing_coregistered.ecsv"
    _fits.apply_coregistration(pointing, table).write(
        path, format="ascii.ecsv", overwrite=True
    )
    print(f"channel offsets -> {directory / 'coregistration.ecsv'}, applied in {path}")


def accept(directory: pathlib.Path) -> None:
    """Score the reference on frames across the flight, without an inversion."""
    path_edges = directory / "window_edges.ecsv"
    _fits.acceptance(
        reference=esis.optics.DistortionParameters.from_file(
            directory / "distortion_reference.ecsv"
        ),
        pointing=astropy.table.QTable.read(_pointing(directory), format="ascii.ecsv"),
        edges=(
            astropy.table.QTable.read(path_edges, format="ascii.ecsv")
            if path_edges.exists()
            else None
        ),
        num_scene=NUM_SCENE,
        device=DEVICE,
        merit=MERIT,
        path=directory / f"acceptance{SUFFIX}.ecsv",
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
    elif command == "polish":
        polish(pathlib.Path(sys.argv[2]))
    elif command == "align":
        align(pathlib.Path(sys.argv[2]))
    elif command == "defocus":
        defocus(pathlib.Path(sys.argv[2]))
    elif command == "pointing":
        pointing(int(sys.argv[2]), pathlib.Path(sys.argv[3]))
    elif command == "gather":
        gather(pathlib.Path(sys.argv[2]))
    elif command == "coregister":
        coregister(pathlib.Path(sys.argv[2]))
    elif command == "accept":
        accept(pathlib.Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
