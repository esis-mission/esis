#!/usr/bin/env python3
"""
Submit the parallel Level-4 inversion to SLURM on tempest.

Run this on the tempest login node, inside the esis virtual environment::

    python submit_level_4.py production
    python submit_level_4.py compare-chain
    python submit_level_4.py compare-weights --frames 0,7,15,22,29

Modes
-----
``production``
    The shared-weights parallel inversion: one job builds the regridding
    weights of the reference frame (and inverts it), a 30-wide array job
    inverts every frame independently from the Gaussian seed, and a gather
    job assembles the full :class:`esis.data.Level_4` product.
``compare-chain``
    The sequential warm-start chain (:func:`esis.flights.f1.data.level_4`),
    the method the parallel inversion is compared against.
    Shares the weights of the production run.
``compare-weights``
    Re-invert a subset of frames with weights rebuilt from the fitted
    per-frame pointing (``weights_shared=False``), to compare against the
    shared-weights results.  Each frame builds its own weights, which are
    hundreds of gigabytes of cache per frame — hence the subset.
"""

import argparse
import pathlib
import subprocess
import time

ACCOUNT = "group-charleskankelborg"
CACHE = "/home/group/charleskankelborg/esis-cache"
VENV = "~/venvs/esis/bin/activate"

PREAMBLE = f"""
source {VENV}
export ESIS_CACHE_DIR={CACHE}
set -x
"""


def sbatch(
    name: str,
    command: str,
    directory: pathlib.Path,
    partition: str = "unsafe",
    mem: str = "500G",
    cpus: int = 64,
    hours: int = 8,
    array: None | str = None,
    dependency: None | str = None,
    requeue: bool = False,
) -> str:
    """
    Write a batch script and submit it, returning the job id.

    Parameters
    ----------
    name
        The job name and the stem of the script and log filenames.
    command
        The shell command the job runs.
    directory
        The directory the script and logs are written to.
    partition
        The partition to submit to.
    mem
        The memory request, passed to ``--mem``.
    cpus
        The number of CPUs per task.
    hours
        The time limit in hours.
    array
        An optional array specification, e.g. ``"0-29"``.
    dependency
        An optional dependency specification, e.g. ``"afterok:12345"``.
    requeue
        Whether to requeue the job if it is preempted.
    """
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --account={ACCOUNT}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --time={hours}:00:00",
        f"#SBATCH --output={directory}/{name}-%A-%a.log",
    ]
    if array is not None:
        lines.append(f"#SBATCH --array={array}")
    if dependency is not None:
        lines.append(f"#SBATCH --dependency={dependency}")
    if requeue:
        lines.append("#SBATCH --requeue")
    lines.append(PREAMBLE)
    lines.append(command)

    script = directory / f"{name}.sbatch"
    script.write_text("\n".join(lines) + "\n")

    result = subprocess.run(
        ["sbatch", "--parsable", str(script)],
        capture_output=True,
        text=True,
        check=True,
    )
    job = result.stdout.strip().split(";")[0]
    print(f"{name}: job {job}")
    return job


def main() -> None:
    """Parse the command line and submit the requested mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["production", "compare-chain", "compare-weights"],
    )
    parser.add_argument(
        "--frames",
        default="0,7,15,22,29",
        help="comma-separated frame indices for compare-weights",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="number of frames in the flight",
    )
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    directory = pathlib.Path.home() / "level4_runs" / f"{args.mode}_{stamp}"
    directory.mkdir(parents=True)
    pathlib.Path(CACHE).mkdir(parents=True, exist_ok=True)

    python = "python -u -c"

    if args.mode == "production":
        build = sbatch(
            name="l4-build",
            command=(
                f'{python} "import esis;'
                f' esis.flights.f1.data.level_4_frame(15, verbose=True)"'
            ),
            directory=directory,
            partition="fairshare",
            mem="0",
            cpus=384,
            hours=24,
        )
        frames = sbatch(
            name="l4-frames",
            command=(
                f'{python} "import os, esis;'
                f" esis.flights.f1.data.level_4_frame("
                f"int(os.environ['SLURM_ARRAY_TASK_ID']), verbose=True)\""
            ),
            directory=directory,
            array=f"0-{args.num_frames - 1}",
            dependency=f"afterok:{build}",
            requeue=True,
        )
        sbatch(
            name="l4-gather",
            command=(
                f'{python} "import esis;'
                f" a = esis.flights.f1.data.level_4_parallel();"
                f' print(a.outputs.shape)"'
            ),
            directory=directory,
            partition="fairshare",
            mem="500G",
            cpus=16,
            hours=8,
            dependency=f"afterok:{frames}",
        )

    elif args.mode == "compare-chain":
        sbatch(
            name="l4-chain",
            command=(
                f'{python} "import esis;'
                f' esis.flights.f1.data.level_4(verbose=True)"'
            ),
            directory=directory,
            partition="fairshare",
            mem="0",
            cpus=384,
            hours=72,
        )

    elif args.mode == "compare-weights":
        sbatch(
            name="l4-perframe",
            command=(
                f'{python} "import os, esis;'
                f" esis.flights.f1.data.level_4_frame("
                f"int(os.environ['SLURM_ARRAY_TASK_ID']),"
                f' weights_shared=False, verbose=True)"'
            ),
            directory=directory,
            array=args.frames,
            partition="fairshare",
            mem="0",
            cpus=384,
            hours=24,
            requeue=False,
        )

    print(f"scripts and logs: {directory}")


if __name__ == "__main__":
    main()
