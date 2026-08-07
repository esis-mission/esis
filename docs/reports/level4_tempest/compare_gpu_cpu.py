#!/usr/bin/env python3
"""
Cross-check the 30-frame GPU inversion against the CPU parallel product.

Experimental tooling: untracked, not part of the esis package.
"""

import numpy as np

import esis

OUT = "/home/group/charleskankelborg/level4_gpu"


def main() -> None:
    """Compare per-frame chi-squared and solutions."""
    cpu = esis.flights.f1.data.level_4_parallel()
    gpu = np.load(OUT + "/gpu_full_run.npz", mmap_mode="r")

    solutions_gpu = gpu["solutions"]
    chi2_gpu = np.asarray(gpu["chi2_final"])
    iterations_gpu = np.asarray(gpu["iterations"])

    axis_time = cpu.axis_time
    num_time = cpu.shape[axis_time]

    worst_chi2 = 0.0
    worst_solution = 0.0
    for t in range(num_time):
        chi2_cpu = np.asarray(
            cpu.mean_chi_squared[{axis_time: t}].ndarray, dtype=float
        )
        n_cpu = int(np.asarray(cpu.num_iteration[{axis_time: t}].ndarray))

        sol = cpu.outputs[{axis_time: t}]
        sol_cpu = np.moveaxis(
            np.asarray(sol.ndarray.value, dtype=np.float64),
            [sol.axes.index(ax) for ax in ("wavelength", "field_x", "field_y")],
            range(3),
        )
        sol_gpu = np.asarray(solutions_gpu[t], dtype=np.float64)

        d_chi2 = np.abs(chi2_gpu[t] - chi2_cpu).max()
        d_sol = np.abs(sol_gpu - sol_cpu).mean() / np.abs(sol_cpu).mean()
        worst_chi2 = max(worst_chi2, d_chi2)
        worst_solution = max(worst_solution, d_sol)

        print(
            f"frame {t:3d}: iters gpu/cpu {iterations_gpu[t]:3d}/{n_cpu:3d}"
            f" | max|dchi2| {d_chi2:.2e}"
            f" | mean|dsol|/|cpu| {d_sol:.2e}",
            flush=True,
        )

    print(f"WORST: dchi2 {worst_chi2:.2e}, dsol {worst_solution:.2e}", flush=True)


if __name__ == "__main__":
    main()
