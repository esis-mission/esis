#!/usr/bin/env python3
"""
Portable reference driver for the tied-window Level-4 MART inversion.

A single-machine adaptation of ``local_mart.py`` (itself the streamed
single-GPU port of the production driver ``gpu_mart_tied.py`` that ran
on tempest): no cluster paths, no reference files from earlier runs,
an ``--out`` option, and a ``--device cpu`` fallback for machines
without CUDA.  Unlike ``local_mart.py``, the saved ``npz`` carries the
pedestal and channel-normalization diagnostics, so ``export_fits.py``
can convert it straight to the self-describing Level-4 FITS product.

The production run behind the distributed results was

    python -u gpu_mart_tied.py --pitch 0.75 --num-velocity 24 \\
        --fs-scale 1.03 --index-int32 --single-device

on one H100.  This driver reproduces the same model — tied O IV and
Mg X windows, hard field-stop octagon support (vertex-on-axis, three
percent proud of the ideal aperture), frame-transfer shadow mask,
per-frame per-channel pedestal and channel normalization — at whatever
grid the hardware at hand affords.  The FITS export needs a run over
every frame (``--frames all``), since the product carries the full
flight's time axis; a ``--coadd`` run cannot be exported.

Adapted from ``local_mart.py``; validated end-to-end 2026-08-07 at
``--pitch 1.0 --num-velocity 12`` on one 25 GB laptop GPU (63 min:
6 min assembly from a warm weights cache, ~110 s/frame, chi-squared
near one on every channel), including the FITS export and a
``Level_4.from_fits`` round trip of the result.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse
import pathlib
import time

import numpy as np
import scipy.ndimage

import astropy.units as u
import named_arrays as na

import esis
from esis.data._level_4 import _caching

import stream_assembly
import tied_config
from gpu_mart import _probe_factors
from local_mart import _register, _support
from probe_invariance import build as build_probe


def main() -> None:
    """Run the inversion."""
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitch", type=float, required=True, help="arcsec")
    parser.add_argument("--num-velocity", type=int, required=True)
    parser.add_argument("--fs-scale", type=float, default=1.03)
    parser.add_argument("--frames", default="all")
    parser.add_argument("--coadd", action="store_true")
    parser.add_argument("--pitch-probe", type=float, default=6.0)
    parser.add_argument("--chunk", type=int, default=40_000_000)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument(
        "--out",
        default=str(pathlib.Path.home() / "esis_local_runs"),
        help="directory the npz is written to",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', or a torch device like 'cuda:0'",
    )
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"p{args.pitch:g}_nv{args.num_velocity}"
    if args.coadd:
        tag += "_coadd"

    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"device: {device}", flush=True)
    t_start = time.perf_counter()

    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system
    kwargs = dict(
        key=_caching.key_system(system),
        wavelength=tied_config.grids(args.num_velocity)[2],
        degree=2,
        code=_caching.code_state(),
    )
    windows = tied_config.windows()
    _, member_grids, _ = tied_config.grids(args.num_velocity)
    position, _, _, num_field = tied_config.position_grid(system, args.pitch * u.arcsec)
    num_field_cells = num_field * num_field
    num_wavelength = len(windows) * args.num_velocity
    print(f"num_field={num_field}, num_wavelength={num_wavelength}", flush=True)

    # the noise factors, measured on a cheap grid and rescaled: vmr does not
    # depend on the grid, and factor_signal scales as the area of a cell
    probe, shape_in_p, shape_out_p, num_field_probe = build_probe(
        args.pitch_probe, args.num_velocity
    )
    factor_signal, vmr = _probe_factors(probe, shape_in_p, shape_out_p)
    factor_signal = factor_signal * (num_field_probe / num_field) ** 2
    del probe
    print(f"probe rescaled from {num_field_probe} to {num_field}", flush=True)

    a = esis.flights.f1.data.level_1()
    where_shadow = a.where_shadow()
    shape_detector = {
        "channel": a.shape[a.axis_channel],
        "detector_x": a.shape[a.axis_x],
        "detector_y": a.shape[a.axis_y],
    }
    keep = (~where_shadow).broadcast_to(shape_detector)
    src = [keep.axes.index(ax) for ax in shape_detector]
    det_ok = np.moveaxis(keep.ndarray, src, range(3)).copy()
    num_channel = det_ok.shape[0]
    det_ok[:, :4, :] = det_ok[:, -4:, :] = False
    det_ok[:, :, :4] = det_ok[:, :, -4:] = False
    det_ok_flat = det_ok.reshape(num_channel, -1)
    num_detector = det_ok_flat.shape[1]

    inside = _support(system, position, num_field, args.fs_scale)
    print(f"support keeps {inside.mean():.3f}", flush=True)

    forward, transpose, ratios, shapes, count_f, _ = stream_assembly.assemble(
        system=system,
        kwargs=kwargs,
        windows=windows,
        member_grids=member_grids,
        position=position,
        num_velocity=args.num_velocity,
        inside_flat=inside.ravel(),
        det_ok_flat=det_ok_flat,
        factor_signal=factor_signal,
        vmr=vmr,
        device_forward=device,
        device_transpose=device,
    )
    resident = (
        torch.cuda.memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    )
    print(
        f"assembled: {count_f.sum() / 1e9:.3f}G triples,"
        f" {resident:.1f} GiB resident, {time.perf_counter() - t_start:.0f} s",
        flush=True,
    )

    # the seed's own image marks where the model puts light, which is what
    # separates the illuminated field from the background the pedestal is
    # measured in
    num_scene = num_wavelength * num_field_cells
    velocity = np.linspace(-1, 1, args.num_velocity + 1)
    velocity = (velocity[:-1] + velocity[1:]) / 2
    profile = np.concatenate(
        [np.maximum(np.exp(-((velocity * 3) ** 2) / 2) * w[3], 0.01) for w in windows]
    )
    seed = (
        profile.astype(np.float32)[:, None] * inside.astype(np.float32).ravel()[None, :]
    ).ravel()

    def apply_forward(scene, variance):
        """
        Push a scene through the forward weights.

        Parameters
        ----------
        scene
            The scene, on the device.
        variance
            Whether to accumulate the variance image too.
        """
        images = torch.zeros((num_channel, num_detector), device=device)
        var = torch.zeros_like(images) if variance else None
        for c in range(num_channel):
            i_in, i_out, values_c = forward[c]
            for lo in range(0, i_in.numel(), args.chunk):
                sl = slice(lo, lo + args.chunk)
                contribution = values_c[sl] * scene[i_in[sl]]
                images[c].index_add_(0, i_out[sl], contribution)
                if variance:
                    var[c].index_add_(0, i_out[sl], contribution * ratios[c][sl])
        return images, var

    def apply_transpose(images):
        """
        Push detector images back onto the scene.

        Parameters
        ----------
        images
            The detector images, on the device.
        """
        result = torch.zeros((num_channel, num_scene), device=device)
        for c in range(num_channel):
            i_in, i_out, values_c = transpose[c]
            for lo in range(0, i_in.numel(), args.chunk):
                sl = slice(lo, lo + args.chunk)
                result[c].index_add_(0, i_out[sl], values_c[sl] * images[c][i_in[sl]])
        return result

    # --- the data ---------------------------------------------------------
    electrons = np.maximum(a.outputs, 0)
    order = ("channel", a.axis_time, "detector_x", "detector_y")
    values = np.moveaxis(
        np.asarray(electrons.ndarray.value),
        [electrons.axes.index(ax) for ax in order],
        range(4),
    )
    num_time = values.shape[1]

    if args.coadd:
        print("registering and summing the flight", flush=True)
        summed = np.zeros((num_channel, values.shape[2], values.shape[3]))
        for c in range(num_channel):
            shifts = _register([values[c, t] for t in range(num_time)])
            print(
                f"  channel {c}: shift x {shifts[:, 0].min():+.1f} to"
                f" {shifts[:, 0].max():+.1f}, y {shifts[:, 1].min():+.1f} to"
                f" {shifts[:, 1].max():+.1f} px",
                flush=True,
            )
            for t in range(num_time):
                summed[c] += scipy.ndimage.shift(
                    values[c, t], -shifts[t], order=1, mode="constant", cval=0
                )
        values = summed[:, None]
        num_time = 1

    seed_gpu = torch.from_numpy(seed.copy()).to(device)
    image_seed = apply_forward(seed_gpu, False)[0]
    total_seed = float(image_seed.sum())
    footprint = (image_seed > 1e-4 * image_seed.mean(dim=1, keepdim=True)).cpu().numpy()
    del seed_gpu, image_seed
    background = (~footprint.reshape(num_channel, -1)) & det_ok_flat
    print(
        f"background pixels per channel: {background.mean(axis=1).round(3)}",
        flush=True,
    )

    pedestal = np.zeros((num_time, num_channel))
    factor_norm = np.ones((num_time, num_channel))
    observed = np.empty((num_time, num_channel, num_detector), dtype=np.float32)
    for t in range(num_time):
        frame = values[:, t].reshape(num_channel, -1)
        for c in range(num_channel):
            pedestal[t, c] = np.median(frame[c][background[c]])
        frame = np.maximum(frame - pedestal[t][:, None], 0) * det_ok_flat
        means = frame.mean(axis=1)
        factor_norm[t] = means[0] / means
        observed[t] = frame * factor_norm[t][:, None]
    print(
        f"pedestal: {np.array2string(pedestal[0], precision=1)} e-",
        flush=True,
    )

    read_noise = np.broadcast_to(
        np.asarray(na.ScalarArray(system.sensor.read_noise).ndarray.value, dtype=float),
        (num_channel,),
    )
    if args.coadd:
        read_noise = read_noise * np.sqrt(30)
    read2 = torch.from_numpy(np.square(read_noise).astype(np.float32)).to(device)

    frames = (
        list(range(num_time))
        if args.frames == "all"
        else [int(x) for x in args.frames.split(",")]
    )
    solutions = np.empty((len(frames), num_scene), dtype=np.float32)
    chi2 = np.empty((len(frames), num_channel))
    iterations = np.empty(len(frames), dtype=int)

    t_frames = time.perf_counter()
    for n, f in enumerate(frames):
        obs = torch.from_numpy(observed[f]).to(device)
        scene = torch.from_numpy(seed.copy()).to(device)
        scene *= float(obs.sum()) / total_seed
        back_obs = torch.clamp(apply_transpose(obs), min=0)

        previous = float("inf")
        t_frame = time.perf_counter()
        for i in range(args.max_iter):
            images, var = apply_forward(scene, True)
            merit = ((obs - images) ** 2 / (var + read2[:, None])).double().mean(dim=1)
            value = float(merit.mean())
            if previous - value < 1e-2:
                break
            previous = value
            correction = back_obs / torch.clamp(apply_transpose(images), min=0)
            correction = torch.nan_to_num(correction, nan=1.0, posinf=1.0, neginf=1.0)
            scene = scene * correction.prod(dim=0) ** (1.0 / num_channel)
        if device.type == "cuda":
            torch.cuda.synchronize()

        solutions[n] = scene.cpu().numpy()
        chi2[n] = merit.cpu().numpy()
        iterations[n] = i + 1
        print(
            f"frame {f:3d}: {i + 1:3d} iterations,"
            f" chi2 {np.array2string(chi2[n], precision=2)},"
            f" {time.perf_counter() - t_frame:.1f} s",
            flush=True,
        )

    path = out / f"reproduce_{tag}.npz"
    np.savez(
        path,
        solutions=solutions.reshape(len(frames), num_wavelength, num_field, num_field),
        frames=np.array(frames),
        chi2_final=chi2,
        iterations=iterations,
        pedestals=pedestal[frames],
        factor_norm=factor_norm[frames],
        inside=inside,
        factor_signal=factor_signal,
        vmr=vmr,
        labels=np.array([w[0] for w in windows]),
    )
    print(
        f"saved {path} ({path.stat().st_size / 2**30:.2f} GiB)"
        f" | frames {time.perf_counter() - t_frames:.0f} s"
        f" | TOTAL {time.perf_counter() - t_start:.0f} s",
        flush=True,
    )
    if device.type == "cuda":
        print(f"peak GPU: {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB")
    print(
        "next: python export_fits.py"
        f" --npz {path} --out <fits-dir>"
        f" --pitch {args.pitch:g} --num-velocity {args.num_velocity}",
        flush=True,
    )


if __name__ == "__main__":
    main()
