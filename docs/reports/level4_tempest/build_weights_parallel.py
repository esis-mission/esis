#!/usr/bin/env python3
"""
Portable per-member parallel weights build.

Builds the tied-configuration member weights as concurrent processes, with
the worker count chosen from the machine's cores and memory — so the same
script runs serially-ish on a laptop, a few wide on rci, and fully fanned
out on a tempest node. ``--slurm`` submits one job per member instead
(maximum tempest fan-out).

The linearization is built serially first: concurrent joblib writes to the
same cache entry race, but distinct member entries are independent.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time

# peak resident memory of one member build at 0.75 arcsec, measured on
# tempest (sacct MaxRSS of the serial build divided by concurrency 1)
GIGABYTES_PER_MEMBER = 60
CORES_PER_MEMBER = 32


def _build_member(pitch: float, num_velocity: int, wavelength: float) -> str:
    """
    Build one member's forward and transpose weights in this process.

    Parameters
    ----------
    pitch
        The scene pitch in arcsec.
    num_velocity
        The number of velocity bins per window.
    wavelength
        The member's rest wavelength in Angstrom, identifying its grid.
    """
    import astropy.units as u
    import named_arrays as na

    import esis
    from esis.data._level_4 import _caching

    import tied_config

    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system
    key = _caching.key_system(system)
    code = _caching.code_state()

    _, member_grids, wavelength_union = tied_config.grids(num_velocity)
    position, _, _, _ = tied_config.position_grid(system, pitch * u.arcsec)

    grid = member_grids[wavelength]
    coordinates = na.SpectralPositionalVectorArray(
        wavelength=grid,
        position=position,
    )
    kwargs = dict(
        key=key,
        wavelength=wavelength_union,
        degree=2,
        code=code,
        coordinates_scene=coordinates,
        axis_wavelength="wavelength",
        axis_field=("field_x", "field_y"),
    )
    t0 = time.perf_counter()
    _caching.weights(system, **kwargs)
    _caching.weights_transpose(system, **kwargs)
    return f"member {wavelength:.2f}: {time.perf_counter() - t0:.0f} s"


def main() -> None:
    """Build all members, in-process pool or SLURM fan-out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitch", type=float, required=True)
    parser.add_argument("--num-velocity", type=int, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--member", type=float, default=None)
    args = parser.parse_args()

    if args.member is not None:
        # worker entry point (also what the SLURM jobs run)
        print(_build_member(args.pitch, args.num_velocity, args.member), flush=True)
        return

    import astropy.units as u

    import esis
    from esis.data._level_4 import _caching

    import tied_config

    # serial prologue: shared cache entries every member depends on
    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system
    key = _caching.key_system(system)
    code = _caching.code_state()
    _, member_grids, wavelength_union = tied_config.grids(args.num_velocity)
    tied_config.position_grid(system, args.pitch * u.arcsec)
    t0 = time.perf_counter()
    _caching.linear_system(
        system, key=key, wavelength=wavelength_union, degree=2, code=code
    )
    print(f"linearize (serial): {time.perf_counter() - t0:.0f} s", flush=True)

    members = sorted(member_grids)

    if args.slurm:
        for wavelength in members:
            script = (
                "#!/bin/bash\n"
                f"#SBATCH --job-name=w{wavelength:.0f}\n"
                "#SBATCH --account=group-charleskankelborg\n"
                "#SBATCH --partition=unsafe\n"
                "#SBATCH --requeue\n"
                f"#SBATCH --mem={GIGABYTES_PER_MEMBER}G\n"
                f"#SBATCH --cpus-per-task={CORES_PER_MEMBER}\n"
                "#SBATCH --time=02:00:00\n"
                f"#SBATCH --output=build-%j-{wavelength:.0f}.log\n"
                f"source {sys.prefix}/bin/activate\n"
                f"export ESIS_CACHE_DIR={os.environ.get('ESIS_CACHE_DIR', '')}\n"
                f"python -u {os.path.abspath(__file__)}"
                f" --pitch {args.pitch} --num-velocity {args.num_velocity}"
                f" --member {wavelength}\n"
            )
            result = subprocess.run(
                ["sbatch", "--parsable"],
                input=script,
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"member {wavelength:.2f}: job {result.stdout.strip()}", flush=True)
        return

    if args.workers is None:
        try:
            import psutil

            gigabytes = psutil.virtual_memory().available / 2**30
        except ImportError:
            gigabytes = 100
        scale = (args.pitch / 0.75) ** -2
        budget = max(1, int(gigabytes / (GIGABYTES_PER_MEMBER * scale)))
        cores = max(1, (os.cpu_count() or 8) // 16)
        args.workers = max(1, min(len(members), budget, cores))
    print(f"workers: {args.workers}", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_build_member, args.pitch, args.num_velocity, wavelength)
            for wavelength in members
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)

    print("BUILD DONE", flush=True)


if __name__ == "__main__":
    main()
