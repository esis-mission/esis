#!/usr/bin/env python3
"""
Add the residual-saving option to the tied GPU runner, in place.

The runner is untracked experimental tooling that lives outside this
repository, so it cannot be transported by git and copying the whole file
over would discard anything the other machine has that this one does not.
This applies the change by anchored replacement instead: it refuses to run
if an anchor is missing, and does nothing if the change is already there.

    python patch_runner.py [path to gpu_mart_tied.py]
"""

import pathlib
import sys

DEFAULT = pathlib.Path.home() / "repos/esis/docs/reports/level4_tempest/gpu_mart_tied.py"

MARKER = "--save-residual"

EDITS = [
    (
        "import argparse\nimport pathlib\nimport time\n",
        "import argparse\nimport pathlib\nimport time\nimport warnings\n",
    ),
    (
        '''    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="run the same apply kernels on the CPU, for a like-for-like baseline",
    )
''',
        '''    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="run the same apply kernels on the CPU, for a like-for-like baseline",
    )
    parser.add_argument(
        "--save-residual",
        action="store_true",
        help=(
            "also write the converged normalized residual of every frame and"
            " channel to a companion npz, binned by --residual-bin"
        ),
    )
    parser.add_argument(
        "--residual-bin",
        type=int,
        default=4,
        help=(
            "bin the saved residual by this factor on each detector axis;"
            " averaging suppresses the photon noise and leaves the systematic"
            " structure, which is what the residual is inspected for"
        ),
    )
''',
    ),
    (
        """    chi2_final = np.empty((len(frames), num_channel))
    iterations = np.empty(len(frames), dtype=int)
""",
        '''    chi2_final = np.empty((len(frames), num_channel))
    iterations = np.empty(len(frames), dtype=int)

    nx_det = shape_detector["detector_x"]
    ny_det = shape_detector["detector_y"]
    bin_det = max(1, args.residual_bin)
    nx_bin, ny_bin = nx_det // bin_det, ny_det // bin_det
    residuals = None
    if args.save_residual:
        residuals = np.empty(
            (len(frames), num_channel, nx_bin, ny_bin), dtype=np.float32
        )

    def bin_detector(flat) -> np.ndarray:
        """
        Average a flat per-detector quantity down onto the binned grid.

        Masked pixels arrive as NaN and are left out of their bin's mean, so
        a bin straddling the edge of the shadow reports the pixels that were
        actually fitted rather than a value diluted towards zero.
        """
        square = flat.reshape(num_channel, nx_det, ny_det)
        square = square[:, : nx_bin * bin_det, : ny_bin * bin_det]
        square = square.reshape(num_channel, nx_bin, bin_det, ny_bin, bin_det)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(square, axis=(2, 4)).astype(np.float32)
''',
    ),
    (
        """        chi2_final[n] = chi2.cpu().numpy()
        iterations[n] = i + 1
""",
        """        chi2_final[n] = chi2.cpu().numpy()
        iterations[n] = i + 1
        if residuals is not None:
            # the loop broke on `images`, so this is the residual of exactly
            # the scene that was stored, in units of its own uncertainty
            residual = ((obs_f - images) / torch.sqrt(width2)).cpu().numpy()
            residual[~det_ok_flat] = np.nan
            residuals[n] = bin_detector(residual)
""",
    ),
    (
        """    print(f"saved {OUT / f'gpu_tied_{tag}.npz'}", flush=True)
""",
        """    print(f"saved {OUT / f'gpu_tied_{tag}.npz'}", flush=True)

    if residuals is not None:
        np.savez(
            OUT / f"gpu_tied_{tag}_residual.npz",
            residual=residuals,
            frames=np.array(frames),
            bin=bin_det,
            shape_detector=np.array([nx_det, ny_det]),
        )
        print(f"saved {OUT / f'gpu_tied_{tag}_residual.npz'}", flush=True)
""",
    ),
]


def main() -> None:
    """Apply every edit, or explain why it cannot be done."""
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    source = path.read_text()

    if MARKER in source:
        print(f"{path} already patched, nothing to do")
        return

    for old, new in EDITS:
        if source.count(old) != 1:
            raise SystemExit(
                f"anchor appears {source.count(old)} times, expected once:\n"
                f"{old[:200]}"
            )
        source = source.replace(old, new)

    backup = path.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(path.read_text())
        print(f"kept the original at {backup}")

    path.write_text(source)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
