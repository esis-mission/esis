#!/usr/bin/env python3
"""
Tied-window Level-4 inversion on GPUs: ties + hard FS support + pedestal.

The five-window configuration of ``tied_config`` (O IV 608/610 and the Mg X
doublet tied at fixed photon ratios), with the hard field-stop octagon
support (scene-side triple filtering + zeroed seed), the frame-transfer
shadow mask (detector-side, from the package), and per-frame per-channel
pedestal subtraction measured from out-of-footprint pixels.

Ported from prototypes/mart_review2_frame15.py (validated 2026-07-30);
EA gains deliberately omitted. Exact per-wavelength noise model throughout.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse
import pathlib
import time
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation
import matplotlib.colors
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

import astropy.units as u
import named_arrays as na

import esis
from esis.data._level_4 import _caching

import tied_config
from gpu_mart import _flat_images, _probe_factors

OUT = pathlib.Path("/home/group/charleskankelborg/level4_gpu_tied")


def main() -> None:
    """Run the tied-window GPU inversion."""
    import ctis
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitch", type=float, required=True, help="arcsec")
    parser.add_argument("--num-velocity", type=int, required=True)
    parser.add_argument("--frames", default="all")
    parser.add_argument("--movies", action="store_true")
    parser.add_argument(
        "--fs-scale",
        type=float,
        default=1.0,
        help="scale factor on the field-stop octagon radius",
    )
    parser.add_argument(
        "--chain",
        action="store_true",
        help="warm-start chain 15..29 then 14..0 instead of parallel seed",
    )
    parser.add_argument(
        "--index-int32",
        action="store_true",
        help="pack the weight indices as int32, halving their memory",
    )
    parser.add_argument(
        "--single-device",
        action="store_true",
        help="hold both weight directions on one GPU",
    )
    parser.add_argument(
        "--mask-border",
        type=int,
        default=0,
        help=(
            "drop this many detector pixels from every edge; astroscrappy is"
            " blind there, so the cosmic rays that survive Level 1 pile up in"
            " that margin"
        ),
    )
    parser.add_argument(
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
    parser.add_argument(
        "--chunk-triples",
        type=int,
        default=250_000_000,
        help=(
            "triples per apply chunk; the gather and the scaled product are"
            " transient, so a smaller chunk trades a little launch overhead"
            " for a much smaller peak"
        ),
    )
    args = parser.parse_args()
    chunk = args.chunk_triples

    OUT.mkdir(exist_ok=True)
    tag = f"p{args.pitch:g}_nv{args.num_velocity}"
    if args.fs_scale != 1:
        tag += f"_fs{args.fs_scale:g}"
    if args.chain:
        tag += "_chain"
    if args.mask_border:
        tag += f"_mask{args.mask_border}"

    t_start = time.perf_counter()

    # --- CPU setup -----------------------------------------------------------
    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system
    key = _caching.key_system(system)
    code = _caching.code_state()

    windows = tied_config.windows()
    num_window = len(windows)
    num_cell = args.num_velocity
    num_wavelength = num_window * num_cell

    velocity, member_grids, wavelength_union = tied_config.grids(num_cell)
    position, center, extent, num_field = tied_config.position_grid(
        system, args.pitch * u.arcsec
    )
    num_field_cells = num_field * num_field
    print(f"num_field={num_field}", flush=True)

    kwargs = dict(key=key, wavelength=wavelength_union, degree=2, code=code)
    linear = _caching.linear_system(system, **kwargs)

    weights_member = {}
    for lam, grid in member_grids.items():
        coordinates = na.SpectralPositionalVectorArray(
            wavelength=grid, position=position
        )
        kwargs_member = dict(
            coordinates_scene=coordinates,
            axis_wavelength="wavelength",
            axis_field=("field_x", "field_y"),
            **kwargs,
        )
        weights_member[lam] = (
            _caching.weights(system, **kwargs_member),
            _caching.weights_transpose(system, **kwargs_member),
        )
    print(f"member weights loaded: {time.perf_counter() - t_start:.0f} s", flush=True)

    fw0, tr0 = next(iter(weights_member.values()))
    num_channel = fw0[1]["channel"]
    num_detector = fw0[2]["detector_x"] * fw0[2]["detector_y"]

    # --- FS octagon, orientation scored against the untied GPU product -------
    x_vertices = position.x.ndarray.to_value(u.arcsec)
    y_vertices = position.y.ndarray.to_value(u.arcsec)
    x_cell = (x_vertices[:-1] + x_vertices[1:]) / 2
    y_cell = (y_vertices[:-1] + y_vertices[1:]) / 2
    XX, YY = np.meshgrid(x_cell, y_cell, indexing="ij")
    cx = float(center.x.ndarray.to_value(u.arcsec))
    cy = float(center.y.ndarray.to_value(u.arcsec))
    half_x = float((extent.x / 2 / 1.25).ndarray.to_value(u.arcsec))

    untied = np.load(
        "/home/group/charleskankelborg/level4_gpu/gpu_full_run.npz",
        mmap_mode="r",
    )
    reference = np.asarray(untied["solutions"][15, 120:143]).sum(axis=0)
    if reference.shape != XX.shape:
        zoom = (XX.shape[0] / reference.shape[0], XX.shape[1] / reference.shape[1])
        reference = scipy.ndimage.zoom(reference, zoom, order=1)
    bright = reference > np.percentile(reference, 70)

    candidates = {
        "vertex-on-axis small": (half_x, 0.0),
        "edge-on-axis": (half_x / np.cos(np.pi / 8), np.pi / 8),
        "vertex-on-axis": (half_x / np.cos(np.pi / 8), 0.0),
    }

    def _octagon_inside(radius, phase):
        angles = phase + np.arange(8) * np.pi / 4
        polygon = matplotlib.path.Path(
            np.stack(
                [cx + radius * np.cos(angles), cy + radius * np.sin(angles)],
                axis=-1,
            )
        )
        return polygon.contains_points(
            np.stack([XX.ravel(), YY.ravel()], axis=-1)
        ).reshape(XX.shape)

    # score at NOMINAL size (the metric finds the true octagon; scaling
    # before scoring lets a padded smaller candidate win), then inflate
    # the chosen orientation by fs_scale
    best = None
    for label, (radius, phase) in candidates.items():
        inside = _octagon_inside(radius, phase)
        score = bright[inside].mean() - bright[~inside].mean()
        print(f"orientation {label}: score {score:.4f}", flush=True)
        if best is None or score > best[1]:
            best = (label, score, radius, phase)
    label_best, _, radius_best, phase_best = best
    inside = _octagon_inside(radius_best * args.fs_scale, phase_best)
    print(
        f"chosen orientation: {label_best} (scale {args.fs_scale:g});"
        f" inside fraction {inside.mean():.3f}",
        flush=True,
    )
    inside_flat = inside.ravel()

    # --- shadow mask (package method, full flight) ----------------------------
    a = esis.flights.f1.data.level_1()
    where_shadow = a.where_shadow()
    shape_detector = {
        "channel": a.shape[a.axis_channel],
        "detector_x": a.shape[a.axis_x],
        "detector_y": a.shape[a.axis_y],
    }
    keep = (~where_shadow).broadcast_to(shape_detector)
    src = [keep.axes.index(ax) for ax in shape_detector]
    det_ok = np.moveaxis(keep.ndarray, src, range(3))
    if args.mask_border:
        n = args.mask_border
        before = det_ok.mean()
        det_ok = det_ok.copy()
        det_ok[:, :n, :] = False
        det_ok[:, -n:, :] = False
        det_ok[:, :, :n] = False
        det_ok[:, :, -n:] = False
        print(
            f"border mask: {n} px, detector kept {before:.4f} -> {det_ok.mean():.4f}",
            flush=True,
        )
    det_ok_flat = det_ok.reshape(num_channel, -1)

    # --- tie the members: concatenate, ratio-scale, filter --------------------
    arr_fw = np.empty((num_channel, num_wavelength), dtype=object)
    arr_tr = np.empty((num_channel, num_wavelength), dtype=object)
    n_before = 0
    n_after = 0

    for w, (name, members, _, _) in enumerate(windows):
        fws, trs, scales = [], [], []
        for wavelength_0, scale in members:
            forward, transpose = weights_member[float(wavelength_0.to_value(u.AA))]
            fws.append(forward[0].ndarray)
            trs.append(transpose[0].ndarray)
            scales.append(scale)
        flux = np.array(
            [
                s
                * sum(
                    float(f[c, j][2].sum())
                    for c in range(num_channel)
                    for j in range(num_cell)
                )
                for f, s in zip(fws, scales)
            ]
        )
        alpha = flux / flux.sum()
        for c in range(num_channel):
            for j in range(num_cell):
                k = w * num_cell + j
                idx_in = np.concatenate([f[c, j][0] for f in fws])
                idx_out = np.concatenate([f[c, j][1] for f in fws])
                vals = np.concatenate(
                    [(s * f[c, j][2]).astype(np.float32) for f, s in zip(fws, scales)]
                )
                keep_f = inside_flat[idx_in] & det_ok_flat[c][idx_out]
                n_before += vals.size
                n_after += int(keep_f.sum())
                arr_fw[c, k] = (
                    np.ascontiguousarray(idx_in[keep_f]),
                    np.ascontiguousarray(idx_out[keep_f]),
                    np.ascontiguousarray(vals[keep_f]),
                )

                idx_in = np.concatenate([t[c, j][0] for t in trs])
                idx_out = np.concatenate([t[c, j][1] for t in trs])
                vals = np.concatenate(
                    [(am * t[c, j][2]).astype(np.float32) for t, am in zip(trs, alpha)]
                )
                keep_t = inside_flat[idx_out] & det_ok_flat[c][idx_in]
                arr_tr[c, k] = (
                    np.ascontiguousarray(idx_in[keep_t]),
                    np.ascontiguousarray(idx_out[keep_t]),
                    np.ascontiguousarray(vals[keep_t]),
                )
    print(
        f"tied + filtered: kept {n_after / n_before:.3f} of forward triples,"
        f" {time.perf_counter() - t_start:.0f} s",
        flush=True,
    )

    shape_in = dict(fw0[1])
    shape_in["wavelength"] = num_wavelength
    shape_out = dict(fw0[2])
    shape_out["wavelength"] = num_wavelength
    shape_in_t = dict(tr0[1])
    shape_in_t["wavelength"] = num_wavelength
    shape_out_t = dict(tr0[2])
    shape_out_t["wavelength"] = num_wavelength

    # synthetic monotonic scene-wavelength vertices (primary member per window)
    vertices = None
    for name, members, _, _ in windows:
        grid = member_grids[float(members[0][0].to_value(u.AA))]
        grid = grid.ndarray.to_value(u.AA)
        if vertices is None:
            vertices = list(grid)
        else:
            vertices.extend(vertices[-1] + np.cumsum(np.diff(grid)))
    wavelength_scene = na.ScalarArray(np.array(vertices) * u.AA, axes="wavelength")
    coordinates_scene = na.SpectralPositionalVectorArray(
        wavelength=wavelength_scene,
        position=position,
    )

    instrument_mart = ctis.instruments.OptikaInstrument(
        system=linear,
        coordinates_scene=coordinates_scene,
        channel=a[{a.axis_time: 15}].channel,
        axis_channel="channel",
        axis_wavelength="wavelength",
        axis_scene_xy=("field_x", "field_y"),
    )
    instrument_mart.weights = (
        na.ScalarArray(arr_fw, axes=("channel", "wavelength")),
        shape_in,
        shape_out,
    )
    instrument_mart.weights_transpose = (
        na.ScalarArray(arr_tr, axes=("channel", "wavelength")),
        shape_in_t,
        shape_out_t,
    )

    # --- seed: window profile on the FS support -------------------------------
    velocity_center = (
        velocity[dict(wavelength=slice(None, -1))]
        + velocity[dict(wavelength=slice(1, None))]
    ) / 2

    profile = np.empty(num_wavelength)
    for w, (name, _, width, amplitude) in enumerate(windows):
        gaussian = amplitude * np.exp(-np.square(velocity_center / width) / 2)
        profile[w * num_cell : (w + 1) * num_cell] = np.maximum(
            np.asarray(gaussian.ndarray), 0.01
        )
    seed_xy = inside.astype(np.float32)
    seed = (profile.astype(np.float32)[:, None, None] * seed_xy[None, :, :]).reshape(-1)

    # --- noise-model probe -----------------------------------------------------
    t0 = time.perf_counter()
    factor_signal, vmr = _probe_factors(instrument_mart, shape_in, shape_out)
    print(f"probe: {time.perf_counter() - t0:.0f} s", flush=True)

    read_noise = instrument_mart.system.sensor.read_noise
    read_noise = np.asarray(na.ScalarArray(read_noise).ndarray.value, dtype=float)
    read_noise = np.broadcast_to(read_noise, (num_channel,)).astype(np.float64)

    # --- footprint for the pedestal (CPU forward of the seed) ------------------
    images_probe = na.FunctionArray(
        inputs=instrument_mart.coordinates_sensor,
        outputs=na.ScalarArray.ones(shape=shape_detector) * u.electron,
    )
    unit_scene = na.unit(instrument_mart.backproject(images_probe.outputs).outputs)
    seed_na = na.ScalarArray(
        seed.reshape(num_wavelength, num_field, num_field).astype(np.float64),
        axes=("wavelength", "field_x", "field_y"),
    )
    image_seed = instrument_mart.image(seed_na * unit_scene, noise=False)
    fp = np.moveaxis(
        np.asarray(image_seed.outputs.ndarray.value),
        [image_seed.outputs.axes.index(ax) for ax in shape_detector],
        range(3),
    )
    footprint = fp > 1e-4 * fp.mean(axis=(-2, -1), keepdims=True)
    background_ok = ~footprint & det_ok
    print(
        f"background pixels per channel: {background_ok.mean(axis=(-2, -1))}",
        flush=True,
    )

    # --- preprocess all frames --------------------------------------------------
    num_time = a.shape[a.axis_time]
    electrons = np.maximum(a.outputs, 0)
    e_all = np.moveaxis(
        np.asarray(electrons.ndarray.value),
        [
            electrons.axes.index(ax)
            for ax in ("channel", a.axis_time, "detector_x", "detector_y")
        ],
        range(4),
    )
    pedestals = np.zeros((num_time, num_channel))
    obs_all = np.empty((num_time, num_channel, num_detector), dtype=np.float32)
    factor_norm = np.zeros((num_time, num_channel))
    for f in range(num_time):
        frame = e_all[:, f]
        for c in range(num_channel):
            pedestals[f, c] = np.median(frame[c][background_ok[c]])
        frame = np.maximum(frame - pedestals[f][:, None, None], 0)
        frame = frame * det_ok
        means = frame.mean(axis=(-2, -1))
        factor_norm[f] = means[0] / means
        frame = frame * factor_norm[f][:, None, None]
        obs_all[f] = frame.reshape(num_channel, -1)
    print(
        f"pedestals (frame 15): {np.array2string(pedestals[15], precision=2)} e-",
        flush=True,
    )

    # --- GPU upload --------------------------------------------------------------
    if args.device == "cpu":
        device_f = device_t = torch.device("cpu")
        print(f"cpu threads: {torch.get_num_threads()}", flush=True)
    else:
        device_f = torch.device("cuda:0")
        device_t = torch.device("cuda:0" if args.single_device else "cuda:1")

    # the indices are flattened, so they run to num_wavelength * num_field**2
    # (about 1.6e8 here) and to the detector pixel count; int32 holds both
    # with room to spare, and halves the resident weights
    dtype_index = np.int32 if args.index_int32 else np.int64
    limit = num_wavelength * num_field_cells
    assert limit < np.iinfo(dtype_index).max, "indices overflow the chosen dtype"

    fwd = []
    for c in range(num_channel):
        idx_in, idx_out, vals, vals_var = [], [], [], []
        for k in range(num_wavelength):
            t_in, t_out, t_val = arr_fw[c, k]
            idx_in.append(t_in.astype(dtype_index) + dtype_index(k * num_field_cells))
            idx_out.append(t_out.astype(dtype_index))
            vals.append(t_val * np.float32(factor_signal[c, k]))
            vals_var.append(t_val * np.float32(factor_signal[c, k] * vmr[c, k]))
        fwd.append(
            tuple(
                torch.from_numpy(np.concatenate(x)).to(device_f)
                for x in (idx_in, idx_out, vals, vals_var)
            )
        )
    bwd = []
    for c in range(num_channel):
        idx_in, idx_out, vals = [], [], []
        for k in range(num_wavelength):
            t_in, t_out, t_val = arr_tr[c, k]
            idx_in.append(t_in.astype(dtype_index))
            idx_out.append(t_out.astype(dtype_index) + dtype_index(k * num_field_cells))
            vals.append(t_val)
        bwd.append(
            tuple(
                torch.from_numpy(np.concatenate(x)).to(device_t)
                for x in (idx_in, idx_out, vals)
            )
        )
    read2 = torch.from_numpy(np.square(read_noise).astype(np.float32)).to(device_f)
    num_scene = num_wavelength * num_field_cells
    print(f"gpu upload complete: {time.perf_counter() - t_start:.0f} s", flush=True)

    num_triple_f = sum(int(t[0].numel()) for t in fwd)
    num_triple_b = sum(int(t[0].numel()) for t in bwd)
    bytes_per = 2 * np.dtype(dtype_index).itemsize + 4
    print(
        f"index dtype {np.dtype(dtype_index).name},"
        f" {bytes_per} B/triple (forward carries a variance copy, so +4)",
        flush=True,
    )
    for name, device, num in (
        ("forward", device_f, num_triple_f),
        ("transpose", device_t, num_triple_b),
    ):
        if device.type == "cuda":
            where = (
                f" {torch.cuda.memory_allocated(device) / 2**30:.1f} GiB resident on"
                f" {torch.cuda.get_device_name(device)}"
                f" ({torch.cuda.get_device_properties(device).total_memory / 2**30:.0f}"
                " GiB)"
            )
        else:
            where = f" {num * bytes_per / 2**30:.1f} GiB resident in host memory"
        print(f"{name}: {num / 1e9:.3f}G triples,{where}", flush=True)
    if device_f.type == "cuda":
        print(
            f"peak reserved: {torch.cuda.max_memory_reserved(device_f) / 2**30:.1f} GiB"
            f" (cuda:0), {torch.cuda.max_memory_reserved(device_t) / 2**30:.1f} GiB"
            f" ({device_t})",
            flush=True,
        )

    # effective detected photons per unit scene value: per triple the mean
    # signal is w*s and its variance vmr-scaled, so photons = signal^2/var;
    # column-summing w^2/w_var over channels gives N[s] = colsum[s] * scene[s]
    colsum_photon = np.zeros(num_scene, dtype=np.float64)
    for c in range(num_channel):
        t_in, t_out, t_val, t_var = fwd[c]
        per = torch.zeros(num_scene, device=device_f)
        per.index_add_(0, t_in, t_val * t_val / torch.clamp(t_var, min=1e-30))
        colsum_photon += per.double().cpu().numpy()
        del per

    def forward(scene_f, variance: bool):
        """
        Apply the forward weights, optionally also accumulating variance.

        Parameters
        ----------
        scene_f
            The scene, resident on the forward device.
        variance
            Whether to also compute the variance image.
        """
        images = torch.zeros((num_channel, num_detector), device=device_f)
        var = (
            torch.zeros((num_channel, num_detector), device=device_f)
            if variance
            else None
        )
        for c in range(num_channel):
            t_in, t_out, t_val, t_var = fwd[c]
            for lo in range(0, t_in.numel(), chunk):
                sl = slice(lo, lo + chunk)
                contribution = scene_f[t_in[sl]]
                images[c].index_add_(0, t_out[sl], t_val[sl] * contribution)
                if variance:
                    var[c].index_add_(0, t_out[sl], t_var[sl] * contribution)
                del contribution
        return images, var

    def backproject(images_t):
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
            for lo in range(0, t_in.numel(), chunk):
                sl = slice(lo, lo + chunk)
                result[c].index_add_(0, t_out[sl], t_val[sl] * images_t[c][t_in[sl]])
        return result

    seed_f = torch.from_numpy(seed.copy()).to(device_f)
    image_seed_gpu, _ = forward(seed_f, variance=False)
    total_seed = float(image_seed_gpu.sum())
    del seed_f, image_seed_gpu

    if args.frames == "all":
        frames = list(range(num_time))
    else:
        frames = [int(x) for x in args.frames.split(",")]
    if args.chain:
        frames = [f for f in range(15, num_time) if f in frames] + [
            f for f in range(14, -1, -1) if f in frames
        ]

    solutions = np.empty((len(frames), num_scene), dtype=np.float32)
    photons = np.empty((len(frames), num_scene), dtype=np.float32)
    chi2_final = np.empty((len(frames), num_channel))
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

    t_frames = time.perf_counter()
    done = {}
    for n, f in enumerate(frames):
        obs = obs_all[f].astype(np.float32)
        obs_f = torch.from_numpy(obs).to(device_f)
        obs_t = torch.from_numpy(obs).to(device_t)

        warm = done.get(f + 1, done.get(f - 1)) if args.chain else None
        if warm is not None:
            # inherit the neighbor's solution, floored at 1% of the seed so
            # underflowed cells inside the support can still recover
            scene = torch.from_numpy(np.maximum(warm, seed * 0.01)).to(device_t)
        else:
            scene = torch.from_numpy(seed.copy()).to(device_t)
            scene = scene * (float(obs_f.sum()) / total_seed)

        backproj_obs = torch.clamp(backproject(obs_t), min=0)

        merit_old = float("inf")
        t_frame = time.perf_counter()
        for i in range(100):
            scene_f = scene.to(device_f)
            images, var = forward(scene_f, variance=True)
            width2 = var + read2[:, None]
            chi2 = ((obs_f - images) ** 2 / width2).double().mean(dim=1)

            merit = float(chi2.mean())
            if (merit_old - merit) < 1e-2:
                break
            merit_old = merit

            backproj_new = torch.clamp(backproject(images.to(device_t)), min=0)
            correction = backproj_obs / backproj_new
            correction = torch.nan_to_num(correction, nan=1.0, posinf=1.0, neginf=1.0)
            correction = correction.prod(dim=0) ** (1.0 / num_channel)
            scene = scene * correction
        if device_f.type == "cuda":
            torch.cuda.synchronize()

        solutions[n] = scene.cpu().numpy()
        if args.chain:
            done[f] = solutions[n]
        photons[n] = solutions[n] * colsum_photon
        chi2_final[n] = chi2.cpu().numpy()
        iterations[n] = i + 1
        if residuals is not None:
            # the loop broke on `images`, so this is the residual of exactly
            # the scene that was stored, in units of its own uncertainty
            residual = ((obs_f - images) / torch.sqrt(width2)).cpu().numpy()
            residual[~det_ok_flat] = np.nan
            residuals[n] = bin_detector(residual)
        print(
            f"frame {f:3d}: {i + 1:3d} iterations,"
            f" chi2 {np.array2string(chi2_final[n], precision=2)},"
            f" {time.perf_counter() - t_frame:.1f} s",
            flush=True,
        )
    t_invert = time.perf_counter() - t_frames
    print(f"frames inverted: {t_invert:.0f} s", flush=True)

    np.savez(
        OUT / f"gpu_tied_{tag}.npz",
        solutions=solutions.reshape(len(frames), num_wavelength, num_field, num_field),
        # the photon-equivalent conversion is time-independent, so store the
        # per-cell factor (photons = solutions * factor_photon) rather than a
        # second copy of the whole cube
        factor_photon=colsum_photon.astype(np.float32).reshape(
            num_wavelength, num_field, num_field
        ),
        frames=np.array(frames),
        chi2_final=chi2_final,
        iterations=iterations,
        pedestals=pedestals,
        factor_norm=factor_norm,
        inside=inside,
        factor_signal=factor_signal,
        vmr=vmr,
        labels=np.array([w[0] for w in windows]),
    )
    print(f"saved {OUT / f'gpu_tied_{tag}.npz'}", flush=True)

    if residuals is not None:
        np.savez(
            OUT / f"gpu_tied_{tag}_residual.npz",
            residual=residuals,
            frames=np.array(frames),
            bin=bin_det,
            shape_detector=np.array([nx_det, ny_det]),
        )
        print(f"saved {OUT / f'gpu_tied_{tag}_residual.npz'}", flush=True)

    if args.movies:
        time_values = a.inputs.time
        if a.axis_channel in time_values.shape:
            time_values = time_values.mean(a.axis_channel)

        outputs = na.ScalarArray(
            ndarray=solutions.reshape(
                len(frames), num_wavelength, num_field, num_field
            ).astype(np.float64)
            * (unit_scene if unit_scene is not None else 1),
            axes=("time", "wavelength", "field_x", "field_y"),
        )
        wavelength_center = na.ScalarArray(
            u.Quantity([w[1][0][0].to_value(u.AA) for w in windows] * u.AA),
            axes="line",
        )
        l4 = esis.data.Level_4(
            inputs=na.TemporalSpectralPositionalVectorArray(
                time=(
                    time_values[{a.axis_time: frames}]
                    if len(frames) < num_time
                    else time_values
                ),
                wavelength=wavelength_scene,
                position=position,
            ),
            outputs=outputs,
            instrument=instrument,
            wavelength_center=wavelength_center,
            label_line=[w[0] for w in windows],
            num_velocity=num_cell,
            mean_chi_squared=na.ScalarArray(chi2_final, axes=("time", "channel")),
            num_iteration=na.ScalarArray(iterations, axes=("time",)),
            where_shadow=where_shadow,
        )
        # note: Level_4.window() assumes gap cells; patch the slices directly
        l4.window = lambda i: {"wavelength": slice(i * num_cell, (i + 1) * num_cell)}

        index_OV = num_window - 1
        ani = l4.animate_intensity(index_line=index_OV)
        ani.save(OUT / f"full_fov_intensity_{tag}.gif", writer="pillow", fps=5)
        ani = l4.animate_doppler(index_line=index_OV)
        ani.save(OUT / f"full_fov_doppler_{tag}.gif", writer="pillow", fps=5)

        # photons/pixel movie: SNR readable as sqrt(N)
        photons_map = photons.reshape(
            len(frames), num_wavelength, num_field, num_field
        )[:, index_OV * num_cell : (index_OV + 1) * num_cell].sum(axis=1)
        vmax = np.percentile(photons_map, 99.5)
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        img = ax.imshow(
            photons_map[0].T,
            origin="lower",
            extent=(
                x_vertices[0],
                x_vertices[-1],
                y_vertices[0],
                y_vertices[-1],
            ),
            norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=vmax),
            cmap="viridis",
        )
        ax.set_xlabel("helioprojective x (arcsec)")
        ax.set_ylabel("helioprojective y (arcsec)")
        fig.colorbar(img, ax=ax, label="detected photons / pixel")

        def _update(n):
            img.set_data(photons_map[n].T)
            ax.set_title(f"{windows[index_OV][0]} — frame {frames[n]}")
            return (img,)

        ani = matplotlib.animation.FuncAnimation(
            fig, _update, frames=len(frames), blit=False
        )
        ani.save(OUT / f"full_fov_photons_{tag}.gif", writer="pillow", fps=5)
        plt.close(fig)

        position_event = l4.locate_event()
        print(f"event located at {position_event}", flush=True)
        pos = (
            position_event.to_value(u.arcsec)
            if hasattr(position_event, "to_value")
            else np.asarray(position_event, dtype=float)
        )
        ex = float(pos[0])
        ey = float(pos[1])
        near = (np.abs(XX - ex) < 5) & (np.abs(YY - ey) < 5)
        print(
            f"photons/pixel at event (frame 15, O V, 10 arcsec box):"
            f" median {np.median(photons_map[15][near]):.0f},"
            f" max {photons_map[15][near].max():.0f}",
            flush=True,
        )
        try:
            context = esis.flights.f1.data.aia_context()
            import sunpy.visualization.colormaps  # noqa: F401

            cmaps = {
                "AIA 304": "sdoaia304",
                "AIA 171": "sdoaia171",
                "AIA 193": "sdoaia193",
            }
        except Exception as e:  # noqa: BLE001
            print(f"aia context unavailable ({e})", flush=True)
            context, cmaps = None, None
        ani = l4.animate_event(
            position=position_event, context=context, cmaps_context=cmaps
        )
        ani.save(OUT / f"event_e_{tag}.gif", writer="pillow", fps=5)
        print("movies saved", flush=True)

    print(f"TOTAL: {time.perf_counter() - t_start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
