#!/usr/bin/env python3
"""
Did the surviving cosmic rays corrupt the reconstruction, or only chi-squared?

Level-1 removes cosmic rays but is blind within a few pixels of the array
border, and every frame with an inflated chi-squared carries such a hit.
Those pixels lie outside the illuminated field, so the transpose weights
should never read them, and the reconstruction should be untouched even
though the reported chi-squared is not.

Compares an inversion with the border masked against one without: what
chi-squared does, and what the solution does, frame by frame.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse

import numpy as np

DIR = "/home/group/charleskankelborg/level4_gpu_tied"
SPIKING = (11, 13, 14, 18, 19, 24)


def main() -> None:
    """Run the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain", default=f"{DIR}/gpu_tied_p0.75_nv24_fs1.03.npz")
    parser.add_argument(
        "--masked", default=f"{DIR}/gpu_tied_p0.75_nv24_fs1.03_mask4.npz"
    )
    args = parser.parse_args()

    a = np.load(args.plain, mmap_mode="r")
    b = np.load(args.masked, mmap_mode="r")
    inside = np.asarray(a["inside"]).ravel()

    chi_a = np.asarray(a["chi2_final"])
    chi_b = np.asarray(b["chi2_final"])
    num_time = chi_a.shape[0]

    print(f"{'frame':>6} {'chi2 plain':>28} {'chi2 masked':>28} {'rel |dsol|':>11}")
    for t in range(num_time):
        sol_a = np.asarray(a["solutions"][t], dtype=np.float64).reshape(-1, inside.size)
        sol_b = np.asarray(b["solutions"][t], dtype=np.float64).reshape(-1, inside.size)
        sol_a = sol_a[:, inside]
        sol_b = sol_b[:, inside]
        total = np.abs(sol_a).sum()
        rel = np.abs(sol_a - sol_b).sum() / total if total else np.nan
        flag = "  <- spiking" if t in SPIKING else ""
        print(
            f"{t:>6} {np.array2string(chi_a[t], precision=2):>28}"
            f" {np.array2string(chi_b[t], precision=2):>28} {rel:>11.2e}{flag}"
        )

    quiet = [t for t in range(num_time) if t not in SPIKING]
    print()
    print(
        "mean chi2 over quiet frames:"
        f" plain {chi_a[quiet].mean():.3f}, masked {chi_b[quiet].mean():.3f}"
    )
    print(
        "mean chi2 over spiking frames:"
        f" plain {chi_a[list(SPIKING)].mean():.3f},"
        f" masked {chi_b[list(SPIKING)].mean():.3f}"
    )


if __name__ == "__main__":
    main()
