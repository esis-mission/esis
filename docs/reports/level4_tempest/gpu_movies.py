#!/usr/bin/env python3
"""
Re-render the movies of the GPU full run with line labels and colorbars.

Loads the saved GPU solutions and rebuilds an in-memory
:class:`esis.data.Level_4` using the coordinates, units, and mask of the
cached CPU frame-15 product (no weights needed), then renders the
full-field and Event-E movies with the current movie tool.

Experimental tooling: untracked, not part of the esis package.
"""

import time

import numpy as np

import matplotlib

matplotlib.use("Agg")

import named_arrays as na

import esis
from esis.flights.f1.data._level_4._level_4 import _line_labels

OUT = "/home/group/charleskankelborg/level4_gpu"


def main() -> None:
    """Assemble the product and render the movies."""
    t0 = time.perf_counter()
    data = np.load(OUT + "/gpu_full_run.npz", mmap_mode="r")
    solutions = data["solutions"]
    chi2_final = np.asarray(data["chi2_final"])
    iterations = np.asarray(data["iterations"])

    cpu15 = esis.flights.f1.data.level_4_frame(15)
    unit = na.unit(cpu15.outputs)

    a = esis.flights.f1.data.level_1()
    axis_channel = a.axis_channel
    time_values = a.inputs.time
    if axis_channel in time_values.shape:
        time_values = time_values.mean(axis_channel)

    where_shadow = cpu15.where_shadow
    electrons = np.maximum(a.outputs, 0) * ~where_shadow
    mean_channel = electrons.mean((a.axis_x, a.axis_y))
    factor_norm = mean_channel[{axis_channel: 0}] / mean_channel

    outputs = na.ScalarArray(
        ndarray=np.asarray(solutions, dtype=np.float64) * unit,
        axes=("time", "wavelength", "field_x", "field_y"),
    )
    l4 = esis.data.Level_4(
        inputs=na.TemporalSpectralPositionalVectorArray(
            time=time_values,
            wavelength=cpu15.inputs.wavelength,
            position=cpu15.inputs.position,
        ),
        outputs=outputs,
        instrument=cpu15.instrument,
        wavelength_center=cpu15.wavelength_center,
        label_line=_line_labels(),
        num_velocity=cpu15.num_velocity,
        mean_chi_squared=na.ScalarArray(chi2_final, axes=("time", "channel")),
        num_iteration=na.ScalarArray(iterations, axes=("time",)),
        factor_norm=factor_norm,
        where_shadow=where_shadow,
    )
    print(f"assembled: {time.perf_counter() - t0:.0f} s", flush=True)

    index_OV = l4.num_line - 1

    ani = l4.animate_intensity(index_line=index_OV)
    ani.save(OUT + "/full_fov_intensity.gif", writer="pillow", fps=5)
    print("full-fov intensity movie saved", flush=True)

    ani = l4.animate_doppler(index_line=index_OV)
    ani.save(OUT + "/full_fov_doppler.gif", writer="pillow", fps=5)
    print("full-fov doppler movie saved", flush=True)

    position_event = l4.locate_event()
    print(f"event located at {position_event}", flush=True)

    context = None
    cmaps = None
    try:
        context = esis.flights.f1.data.aia_context()
        import sunpy.visualization.colormaps  # noqa: F401

        cmaps = {
            "AIA 304": "sdoaia304",
            "AIA 171": "sdoaia171",
            "AIA 193": "sdoaia193",
        }
    except Exception as e:  # noqa: BLE001 -- context is optional
        print(f"aia context unavailable ({type(e).__name__}: {e})", flush=True)

    ani = l4.animate_event(
        position=position_event,
        context=context,
        cmaps_context=cmaps,
    )
    ani.save(OUT + "/event_e.gif", writer="pillow", fps=5)
    print(f"TOTAL: {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
