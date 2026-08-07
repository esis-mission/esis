#!/usr/bin/env python3
"""
Prototype: a full MART inversion of one frame on GPUs.

Mirrors ``esis.data.Level_4.from_level_1`` for a single frame, but runs the
iteration on two GPUs (forward weights resident on one, transpose on the
other), with two noise models for the chi-squared width:

- ``exact``: per-wavelength signal and variance coefficients probed
  numerically from optika's own sensor chain (one ``integrate=False`` CPU
  call), folded into premultiplied value arrays — so shot, Fano, and
  partial-charge-collection noise per wavelength are reproduced without
  reimplementing any formula.
- ``fast``: a single band-mean variance-to-mean ratio applied to the summed
  image — the "simplify for speed" model.

Both modes run, their chi-squared traces and solutions are compared to each
other and to the cached CPU product of the same frame.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse
import time

import numpy as np

import astropy.units as u
import named_arrays as na

import esis
from esis.data._level_4 import _caching
from esis.data._level_4._level_4 import _filter_weights_shadow, _guess
from esis.flights.f1.data._level_4._level_4 import _grid_velocity, _lines


def _setup(index_time: int):
    """
    Mirror the CPU single-frame pipeline up to the inversion.

    Parameters
    ----------
    index_time
        The index of the frame to invert.
    """
    import astropy.constants
    import ctis

    a = esis.flights.f1.data.level_1()
    axis_time = a.axis_time
    axis_channel = a.axis_channel

    where_shadow = a.where_shadow()
    frame = a[{axis_time: slice(index_time, index_time + 1)}]

    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system
    key = _caching.key_system(system)
    code = _caching.code_state()

    wavelength_center, width_doppler = _lines()
    num_velocity, limit_velocity = _grid_velocity(
        pitch_velocity=17.5 * u.km / u.s,
        limit_velocity=200 * u.km / u.s,
    )

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

    linear = _caching.linear_system(
        system, key=key, wavelength=wavelength_vertices, degree=2, code=code
    )

    instrument_mart = ctis.instruments.OptikaInstrument(
        system=linear,
        coordinates_scene=coordinates_scene,
        channel=frame.channel,
        axis_channel=axis_channel,
        axis_wavelength="wavelength",
        axis_scene_xy=("field_x", "field_y"),
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
    instrument_mart.weights = _caching.weights(system, **kwargs)
    instrument_mart.weights_transpose = _caching.weights_transpose(system, **kwargs)

    # replicate the shadow filtering of from_level_1
    shape_detector = {
        axis_channel: a.shape[axis_channel],
        a.axis_x: a.shape[a.axis_x],
        a.axis_y: a.shape[a.axis_y],
    }
    keep = (~where_shadow).broadcast_to(shape_detector)
    src = [keep.axes.index(ax) for ax in shape_detector]
    keep_flat = np.moveaxis(keep.ndarray, src, range(3)).reshape(
        shape_detector[axis_channel], -1
    )
    instrument_mart.weights = _filter_weights_shadow(
        instrument_mart.weights, keep_flat, axis_channel, index_detector=1
    )
    instrument_mart.weights_transpose = _filter_weights_shadow(
        instrument_mart.weights_transpose, keep_flat, axis_channel, index_detector=0
    )

    # replicate the image preprocessing of from_level_1
    electrons = np.maximum(frame.outputs, 0)
    electrons = electrons * ~where_shadow
    mean_channel = electrons.mean((a.axis_x, a.axis_y))
    mean_reference = mean_channel[{axis_channel: 0}]
    electrons = electrons * (mean_reference / mean_channel)
    electrons = electrons[{axis_time: 0}]

    guess = _guess(
        instrument_mart=instrument_mart,
        electrons_reference=electrons,
        width_doppler=width_doppler,
        velocity=velocity,
        num_velocity=num_velocity,
        floor_guess=0.01,
        axis_wavelength="wavelength",
        axis_line="line",
    )

    return instrument_mart, electrons, guess


def _flat_images(electrons, axis_channel, axis_x, axis_y) -> np.ndarray:
    """
    Flatten the observed images to ``(channel, detector)`` C-order.

    Parameters
    ----------
    electrons
        The preprocessed observed images.
    axis_channel
        The channel axis name.
    axis_x
        The horizontal detector axis name.
    axis_y
        The vertical detector axis name.
    """
    src = [electrons.axes.index(ax) for ax in (axis_channel, axis_x, axis_y)]
    arr = np.moveaxis(np.asarray(electrons.ndarray), src, range(3))
    return arr.reshape(arr.shape[0], -1).astype(np.float64)


def _probe_factors(instrument_mart, shape_input, shape_output):
    """
    Probe optika's per-element signal and variance coefficients.

    One ``integrate=False`` call of the CPU image chain on a uniform scene
    yields the per-wavelength nominal electrons and width at every pixel;
    dividing by the raw scatter of the same scene gives the per-element
    conversion factor, and width²/nominal gives the per-element
    variance-to-mean ratio.

    Parameters
    ----------
    instrument_mart
        The (shadow-filtered) CTIS instrument.
    shape_input
        The input shape of the forward weights.
    shape_output
        The output shape of the forward weights.
    """
    num_field = shape_input["field_x"] * shape_input["field_y"]
    num_detector = shape_output["detector_x"] * shape_output["detector_y"]
    num_wavelength = shape_input["wavelength"]
    num_channel = shape_input["channel"]

    # the CPU probe: per-wavelength nominal and width for a uniform scene
    backprojected = instrument_mart.backproject(
        na.FunctionArray(
            inputs=instrument_mart.coordinates_sensor,
            outputs=na.ScalarArray.ones(
                shape={
                    "channel": num_channel,
                    "detector_x": shape_output["detector_x"],
                    "detector_y": shape_output["detector_y"],
                }
            )
            * u.electron,
        ).outputs
    )
    unit_scene = na.unit(backprojected.outputs)
    del backprojected

    ones = (
        na.ScalarArray.ones(
            shape={
                "wavelength": num_wavelength,
                "field_x": shape_input["field_x"],
                "field_y": shape_input["field_y"],
            }
        )
        * unit_scene
    )
    probe = instrument_mart.image(ones, noise=False, uncertainty=True, integrate=False)
    nominal = probe.outputs.nominal
    width = probe.outputs.width

    table = instrument_mart.weights[0]
    axes = table.axes
    view = np.moveaxis(table.ndarray, axes.index("channel"), 0)

    factor_signal = np.zeros((num_channel, num_wavelength))
    vmr = np.zeros((num_channel, num_wavelength))
    spread_max = 0.0
    for c in range(num_channel):
        elements = view[c].reshape(-1)
        for j in range(num_wavelength):
            idx_in, idx_out, vals = elements[j]
            raw = np.zeros(num_detector)
            np.add.at(raw, idx_out.astype(np.int64), vals.astype(np.float64))
            if raw.max() == 0:
                continue

            n_j = nominal[{"channel": c, "wavelength": j}]
            w_j = width[{"channel": c, "wavelength": j}]
            n_flat = np.asarray(n_j.ndarray.value, dtype=np.float64).reshape(-1)
            w_flat = np.asarray(w_j.ndarray.value, dtype=np.float64).reshape(-1)

            bright = raw > 0.1 * raw.max()
            f = n_flat[bright] / raw[bright]
            v = np.square(w_flat[bright]) / n_flat[bright]
            factor_signal[c, j] = np.median(f)
            vmr[c, j] = np.median(v)
            spread = np.std(f) / np.median(f)
            spread_max = max(spread_max, spread)

    print(f"factor extraction max relative spread: {spread_max:.2e}", flush=True)
    print(
        f"vmr range: {vmr.min():.3f} - {vmr.max():.3f}"
        f" (band mean {vmr.mean():.3f})",
        flush=True,
    )
    return factor_signal, vmr


def main() -> None:
    """Run the GPU MART prototype."""
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()

    t_setup = time.perf_counter()
    instrument_mart, electrons, guess = _setup(args.frame)
    table, shape_input, shape_output = instrument_mart.weights
    num_field = shape_input["field_x"] * shape_input["field_y"]
    num_detector = shape_output["detector_x"] * shape_output["detector_y"]
    num_wavelength = shape_input["wavelength"]
    num_channel = shape_input["channel"]
    print(f"setup: {time.perf_counter() - t_setup:.0f} s", flush=True)

    t_probe = time.perf_counter()
    factor_signal, vmr = _probe_factors(instrument_mart, shape_input, shape_output)
    print(f"probe: {time.perf_counter() - t_probe:.0f} s", flush=True)

    read_noise = instrument_mart.system.sensor.read_noise
    read_noise = np.asarray(na.ScalarArray(read_noise).ndarray.value, dtype=float)
    read_noise = np.broadcast_to(read_noise, (num_channel,)).astype(np.float64)
    print(f"read noise per channel: {read_noise}", flush=True)

    obs = _flat_images(electrons, "channel", "detector_x", "detector_y")

    guess_unit = na.unit(guess)
    scene0 = np.moveaxis(
        np.asarray(guess.ndarray.value, dtype=np.float32),
        [guess.axes.index(ax) for ax in ("wavelength", "field_x", "field_y")],
        range(3),
    ).reshape(-1)

    # --- GPU upload ---------------------------------------------------------
    device_f = torch.device("cuda:0")
    device_t = torch.device("cuda:1")

    view = np.moveaxis(table.ndarray, table.axes.index("channel"), 0)
    fwd = []
    for c in range(num_channel):
        idx_in, idx_out, vals, vals_var = [], [], [], []
        for j, triple in enumerate(view[c].reshape(-1)):
            t_in, t_out, t_val = triple
            idx_in.append(t_in.astype(np.int64) + j * num_field)
            idx_out.append(t_out.astype(np.int64))
            vals.append(t_val.astype(np.float32) * np.float32(factor_signal[c, j]))
            vals_var.append(
                t_val.astype(np.float32)
                * np.float32(factor_signal[c, j] * vmr[c, j])
            )
        fwd.append(
            tuple(
                torch.from_numpy(np.concatenate(x)).to(device_f)
                for x in (idx_in, idx_out, vals, vals_var)
            )
        )

    table_t = instrument_mart.weights_transpose[0]
    view_t = np.moveaxis(table_t.ndarray, table_t.axes.index("channel"), 0)
    bwd = []
    for c in range(num_channel):
        idx_in, idx_out, vals = [], [], []
        for j, triple in enumerate(view_t[c].reshape(-1)):
            t_in, t_out, t_val = triple
            idx_in.append(t_in.astype(np.int64))
            idx_out.append(t_out.astype(np.int64) + j * num_field)
            vals.append(t_val.astype(np.float32))
        bwd.append(
            tuple(
                torch.from_numpy(np.concatenate(x)).to(device_t)
                for x in (idx_in, idx_out, vals)
            )
        )
    print("gpu upload complete", flush=True)

    obs_f = torch.from_numpy(obs.astype(np.float32)).to(device_f)
    obs_t = torch.from_numpy(obs.astype(np.float32)).to(device_t)
    read2 = torch.from_numpy(np.square(read_noise).astype(np.float32)).to(device_f)
    vmr_mean = float(np.average(vmr, weights=factor_signal))
    num_scene = num_wavelength * num_field

    def forward(scene_f: torch.Tensor, variance: bool):
        """
        Apply the forward weights, optionally also accumulating variance.

        Parameters
        ----------
        scene_f
            The scene, resident on the forward device.
        variance
            Whether to also compute the per-wavelength variance image.
        """
        images = torch.zeros((num_channel, num_detector), device=device_f)
        var = (
            torch.zeros((num_channel, num_detector), device=device_f)
            if variance
            else None
        )
        for c in range(num_channel):
            t_in, t_out, t_val, t_var = fwd[c]
            contribution = scene_f[t_in]
            images[c].index_add_(0, t_out, t_val * contribution)
            if variance:
                var[c].index_add_(0, t_out, t_var * contribution)
        return images, var

    def backproject(images_t: torch.Tensor) -> torch.Tensor:
        """
        Apply the transpose weights to detector images.

        Parameters
        ----------
        images_t
            The detector images, resident on the transpose device.
        """
        result = torch.zeros((num_channel, num_scene), device=device_t)
        for c in range(num_channel):
            t_in, t_out, t_val = bwd[c]
            result[c].index_add_(0, t_out, t_val * images_t[c][t_in])
        return result

    backproj_obs = torch.clamp(backproject(obs_t), min=0)

    def run(mode: str):
        """
        Run the MART loop with the given noise mode.

        Parameters
        ----------
        mode
            Either ``"exact"`` or ``"fast"``.
        """
        scene = torch.from_numpy(scene0.copy()).to(device_t)
        chi2_trace = []
        merit_old = float("inf")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        message = f"max iterations ({args.max_iter}) exceeded"
        for i in range(args.max_iter):
            scene_f = scene.to(device_f)
            images, var = forward(scene_f, variance=(mode == "exact"))

            if mode == "exact":
                width2 = var + read2[:, None]
            else:
                width2 = vmr_mean * images + read2[:, None]
            chi2 = ((obs_f - images) ** 2 / width2).double().mean(dim=1)
            chi2_trace.append(chi2.cpu().numpy())

            merit = float(chi2.mean())
            if (merit_old - merit) < args.threshold:
                message = f"converged (delta chi2 < {args.threshold})"
                break
            merit_old = merit

            backproj_new = torch.clamp(backproject(images.to(device_t)), min=0)
            correction = backproj_obs / backproj_new
            correction = torch.nan_to_num(correction, nan=1.0, posinf=1.0, neginf=1.0)
            correction = correction**args.gamma
            correction = correction.prod(dim=0) ** (1.0 / num_channel)
            scene = scene * correction
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        trace = np.stack(chi2_trace)
        print(
            f"[{mode}] {len(trace)} iterations, {elapsed:.1f} s"
            f" ({elapsed / len(trace):.2f} s/iter): {message}",
            flush=True,
        )
        print(f"[{mode}] final chi2: {trace[-1]}", flush=True)
        return scene.cpu().numpy(), trace

    scene_exact, trace_exact = run("exact")
    scene_fast, trace_fast = run("fast")

    diff = np.abs(scene_fast - scene_exact).mean() / np.abs(scene_exact).mean()
    print(f"fast vs exact solution mean |diff|/|exact|: {diff:.2e}", flush=True)

    # --- compare against the cached CPU product -----------------------------
    cpu = esis.flights.f1.data.level_4_frame(args.frame)
    chi2_cpu = np.asarray(
        cpu.mean_chi_squared[{cpu.axis_time: 0}].ndarray, dtype=float
    )
    n_iter_cpu = int(np.asarray(cpu.num_iteration.ndarray).reshape(-1)[0])
    print(f"[cpu ] {n_iter_cpu} iterations, final chi2: {chi2_cpu}", flush=True)

    sol = cpu.outputs[{cpu.axis_time: 0}]
    sol_flat = np.moveaxis(
        np.asarray(sol.ndarray.value, dtype=np.float64),
        [sol.axes.index(ax) for ax in ("wavelength", "field_x", "field_y")],
        range(3),
    ).reshape(-1)
    unit_cpu = na.unit(cpu.outputs)
    if guess_unit is not None and unit_cpu is not None:
        scale = (1 * guess_unit).to_value(unit_cpu)
    else:
        scale = 1.0
    diff_cpu = np.abs(scene_exact * scale - sol_flat).mean() / np.abs(sol_flat).mean()
    print(f"gpu exact vs cpu solution mean |diff|/|cpu|: {diff_cpu:.2e}", flush=True)

    out = "/home/t26q518/level4_runs/bench_gpu/gpu_mart_frame%d.npz" % args.frame
    np.savez_compressed(
        out,
        trace_exact=trace_exact,
        trace_fast=trace_fast,
        chi2_cpu=chi2_cpu,
        n_iter_cpu=n_iter_cpu,
        factor_signal=factor_signal,
        vmr=vmr,
        vmr_mean=vmr_mean,
        diff_fast_exact=diff,
        diff_gpu_cpu=diff_cpu,
    )
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
