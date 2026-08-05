#!/usr/bin/env python3
"""
Run the production GPU inversion with the coalignment correction installed.

A thin wrapper: install the correction, then hand control to the unmodified
`gpu_mart_tied.py` so the corrected arm differs from the baseline in the
distortion model and nothing else.
"""

import pathlib
import runpy
import sys

TOOLING = pathlib.Path.home() / "repos/esis/docs/reports/level4_tempest"


def main() -> None:
    """Install the correction and run the production inversion."""
    sys.path.insert(0, str(TOOLING))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    import correction

    correction.install()

    sys.argv = ["gpu_mart_tied.py"] + sys.argv[1:]
    runpy.run_path(str(TOOLING / "gpu_mart_tied.py"), run_name="__main__")


if __name__ == "__main__":
    main()
