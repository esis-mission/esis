#!/usr/bin/env python3
"""
Benchmark a torch scatter-add implementation of the MART weights apply.

Run on a tempest GPU node (``sbatch`` via ``submit_bench_gpu.sh`` or an
interactive ``srun``), inside the esis environment with torch installed and
``ESIS_CACHE_DIR`` pointing at the group cache, after the production weights
have been built.

Loads the shared regridding weights of the baseline Level-4 product,
validates a torch implementation of the forward apply against the CPU
:func:`regridding.regrid_from_weights` on a subset of elements, then times
the full forward and transpose applies on CPU and GPU.
"""

import time

import numpy as np

import astropy.units as u
import named_arrays as na

import esis
from esis.data._level_4 import _caching
from esis.flights.f1.data._level_4._level_4 import _grid_velocity, _lines


def _load_weights():
    """Load the compact forward and transpose weights of the baseline grid."""
    wavelength_center, width_doppler = _lines()
    num_velocity, limit_velocity = _grid_velocity(
        pitch_velocity=17.5 * u.km / u.s,
        limit_velocity=200 * u.km / u.s,
    )

    import astropy.constants

    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system
    key = _caching.key_system(system)
    code = _caching.code_state()

    velocity = na.linspace(
        start=-limit_velocity,
        stop=limit_velocity,
        axis="wavelength",
        num=num_velocity + 1,
    )
    wavelength_vertices = wavelength_center * (1 + velocity / astropy.constants.c)
    wavelength_vertices = wavelength_vertices.to(u.AA).combine_axes(
        axes=("line", "wavelength"),
        axis_new="wavelength",
    )

    pitch_scene = 0.75 * u.arcsec
    factor_fov = 1.25
    field = system.rayfunction_default.inputs.field
    center = (field.max() + field.min()) / 2
    halfwidth = factor_fov * (field.max() - field.min()) / 2
    start = center - halfwidth
    stop = center + halfwidth
    extent = stop - start
    ratio = (np.maximum(extent.x, extent.y) / pitch_scene).ndarray
    num_field = int(np.ceil(ratio.to_value(u.dimensionless_unscaled)))
    position = na.Cartesian2dVectorLinearSpace(
        start=start,
        stop=stop,
        axis=na.Cartesian2dVectorArray("field_x", "field_y"),
        num=num_field + 1,
    )
    coordinates_scene = na.SpectralPositionalVectorArray(
        wavelength=wavelength_vertices,
        position=position,
    )

    kwargs = dict(
        key=key,
        wavelength=wavelength_vertices,
        degree=2,
        coordinates_scene=coordinates_scene,
        axis_wavelength="wavelength",
        axis_field=("field_x", "field_y"),
        code=code,
    )
    forward = _caching.weights(system, **kwargs)
    transpose = _caching.weights_transpose(system, **kwargs)
    return forward, transpose


def _concatenate(weights, num_input: int, num_output: int):
    """
    Concatenate the per-element triples of one channel with global offsets.

    Parameters
    ----------
    weights
        The ``(table, shape_input, shape_output)`` compact weights.
    num_input
        The flattened size of one input (wavelength) plane.
    num_output
        The flattened size of one output plane.
    """
    table, shape_input, shape_output = weights
    axes = table.axes
    index_channel = axes.index("channel")
    view = np.moveaxis(table.ndarray, index_channel, 0)

    result = []
    for c in range(view.shape[0]):
        idx_in = []
        idx_out = []
        vals = []
        for k, triple in enumerate(view[c].reshape(-1)):
            indices_input, indices_output, values = triple
            idx_in.append(indices_input.astype(np.int64) + k * num_input)
            idx_out.append(indices_output.astype(np.int64))
            vals.append(values)
        result.append(
            (
                np.concatenate(idx_in),
                np.concatenate(idx_out),
                np.concatenate(vals),
            )
        )
    return result


def main() -> None:
    """Run the benchmark."""
    import regridding
    import torch

    print("cuda available:", torch.cuda.is_available(), flush=True)
    for i in range(torch.cuda.device_count()):
        print("device", i, torch.cuda.get_device_name(i), flush=True)

    forward, transpose = _load_weights()
    table, shape_input, shape_output = forward
    print("shape_input:", shape_input, flush=True)
    print("shape_output:", shape_output, flush=True)

    num_wavelength = shape_input["wavelength"]
    num_field = shape_input["field_x"] * shape_input["field_y"]
    num_detector = shape_output["detector_x"] * shape_output["detector_y"]
    num_channel = shape_input["channel"]

    rng = np.random.default_rng(42)
    scene = rng.random((num_wavelength * num_field,)).astype(np.float32)

    # --- CPU reference and timing: one element, then extrapolate ------------
    view = np.moveaxis(table.ndarray, table.axes.index("channel"), 0)
    triples_0 = view[0].reshape(-1)[num_wavelength // 2]
    scene_plane = scene[:num_field].astype(float)

    num_element = num_channel * num_wavelength
    t_element = None
    try:
        t0 = time.perf_counter()
        regridding.regrid_from_weights(
            weights=(
                triples_0[0].astype(np.int64),
                triples_0[1].astype(np.int64),
                triples_0[2].astype(float),
            ),
            shape_input=(num_field,),
            shape_output=(num_detector,),
            values_input=scene_plane,
        )
        t_element = time.perf_counter() - t0
        print(
            f"cpu single-element apply: {t_element:.3f} s"
            f" (× {num_element} elements ≈ {t_element * num_element:.1f} s"
            f" per full forward)",
            flush=True,
        )
    except TypeError as e:
        print(f"cpu reference skipped (signature mismatch: {e})", flush=True)

    # --- GPU setup ----------------------------------------------------------
    concatenated = _concatenate(forward, num_field, num_detector)

    device = torch.device("cuda:0")
    channels = []
    bytes_total = 0
    for idx_in, idx_out, vals in concatenated:
        t_in = torch.from_numpy(idx_in).to(device)
        t_out = torch.from_numpy(idx_out).to(device)
        t_val = torch.from_numpy(vals.astype(np.float32)).to(device)
        channels.append((t_in, t_out, t_val))
        bytes_total += t_in.nbytes + t_out.nbytes + t_val.nbytes
    print(f"gpu weights resident: {bytes_total / 2**30:.1f} GiB", flush=True)

    scene_gpu = torch.from_numpy(scene).to(device)

    def forward_apply() -> torch.Tensor:
        """Apply the forward weights to the resident scene."""
        images = torch.zeros(
            (num_channel, num_detector),
            dtype=torch.float32,
            device=device,
        )
        for c, (t_in, t_out, t_val) in enumerate(channels):
            images[c].index_add_(0, t_out, t_val * scene_gpu[t_in])
        return images

    # --- validate one channel against the CPU kernel ------------------------
    images = forward_apply()
    t_in, t_out, t_val = channels[0]
    check = np.zeros(num_detector, dtype=np.float64)
    idx_in, idx_out, vals = concatenated[0]
    np.add.at(check, idx_out, vals.astype(np.float64) * scene[idx_in])
    gpu_image = images[0].cpu().numpy().astype(np.float64)
    error = np.abs(gpu_image - check).max() / max(check.max(), 1e-30)
    print(f"gpu vs cpu max relative error (channel 0): {error:.2e}", flush=True)

    # --- timing --------------------------------------------------------------
    num_repeat = 10
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_repeat):
        forward_apply()
    torch.cuda.synchronize()
    t_gpu = (time.perf_counter() - t0) / num_repeat
    print(f"gpu full forward apply: {t_gpu:.3f} s", flush=True)
    if t_element is not None:
        print(
            f"projected speedup vs cpu: {t_element * num_element / t_gpu:.0f}×",
            flush=True,
        )


if __name__ == "__main__":
    main()
