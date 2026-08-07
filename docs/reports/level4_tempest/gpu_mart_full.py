#!/usr/bin/env python3
"""
The full 30-frame Level-4 production run on GPUs, plus the movies.

Inverts every frame of the flight with the exact per-wavelength noise model
(coefficients probed from optika, as in ``gpu_mart.py``), the weights
resident on two GPUs for the whole run — the long-mission configuration:
upload once, invert arbitrarily many timestamps.

Afterwards assembles an in-memory :class:`esis.data.Level_4` and renders the
full-field and Event-E movies (with AIA context if the cache has it).

Experimental tooling: untracked, not part of the esis package.
"""

import time

import numpy as np

import astropy.units as u
import named_arrays as na

import esis
from esis.data._level_4 import _caching
from esis.data._level_4._level_4 import _filter_weights_shadow, _guess

from gpu_mart import _flat_images, _probe_factors


def _setup_shared():
    """Mirror the CPU pipeline for the full flight, up to the inversion."""
    import astropy.constants
    import ctis

    from esis.flights.f1.data._level_4._level_4 import _grid_velocity, _lines

    a = esis.flights.f1.data.level_1()
    axis_channel = a.axis_channel

    where_shadow = a.where_shadow()

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
        channel=a[{a.axis_time: 15}].channel,
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

    electrons = np.maximum(a.outputs, 0)
    electrons = electrons * ~where_shadow
    mean_channel = electrons.mean((a.axis_x, a.axis_y))
    mean_reference = mean_channel[{axis_channel: 0}]
    factor_norm = mean_reference / mean_channel
    electrons = electrons * factor_norm

    guess_15 = _guess(
        instrument_mart=instrument_mart,
        electrons_reference=electrons[{a.axis_time: 15}],
        width_doppler=width_doppler,
        velocity=velocity,
        num_velocity=num_velocity,
        floor_guess=0.01,
        axis_wavelength="wavelength",
        axis_line="line",
    )

    time_values = a.inputs.time
    if axis_channel in time_values.shape:
        time_values = time_values.mean(axis_channel)

    meta = dict(
        a=a,
        instrument=instrument,
        wavelength_center=wavelength_center,
        num_velocity=num_velocity,
        wavelength_vertices=wavelength_vertices,
        position=position,
        factor_norm=factor_norm,
        where_shadow=where_shadow,
        time=time_values,
    )
    return instrument_mart, electrons, guess_15, meta


def main() -> None:
    """Run the full-flight GPU inversion and render the movies."""
    import torch

    t_start = time.perf_counter()
    instrument_mart, electrons, guess_15, meta = _setup_shared()
    a = meta["a"]
    axis_time = a.axis_time
    num_time = a.shape[axis_time]

    table, shape_input, shape_output = instrument_mart.weights
    num_field = shape_input["field_x"] * shape_input["field_y"]
    num_detector = shape_output["detector_x"] * shape_output["detector_y"]
    num_wavelength = shape_input["wavelength"]
    num_channel = shape_input["channel"]
    num_scene = num_wavelength * num_field
    t_setup = time.perf_counter() - t_start
    print(f"setup: {t_setup:.0f} s", flush=True)

    t0 = time.perf_counter()
    factor_signal, vmr = _probe_factors(instrument_mart, shape_input, shape_output)
    t_probe = time.perf_counter() - t0
    print(f"probe: {t_probe:.0f} s", flush=True)

    read_noise = instrument_mart.system.sensor.read_noise
    read_noise = np.asarray(na.ScalarArray(read_noise).ndarray.value, dtype=float)
    read_noise = np.broadcast_to(read_noise, (num_channel,)).astype(np.float64)

    guess_unit = na.unit(guess_15)
    scene_15 = np.moveaxis(
        np.asarray(guess_15.ndarray.value, dtype=np.float32),
        [guess_15.axes.index(ax) for ax in ("wavelength", "field_x", "field_y")],
        range(3),
    ).reshape(-1)

    # --- GPU upload ---------------------------------------------------------
    t0 = time.perf_counter()
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
                t_val.astype(np.float32) * np.float32(factor_signal[c, j] * vmr[c, j])
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
    read2 = torch.from_numpy(np.square(read_noise).astype(np.float32)).to(device_f)
    t_upload = time.perf_counter() - t0
    print(f"upload: {t_upload:.0f} s", flush=True)

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
            contribution = scene_f[t_in]
            images[c].index_add_(0, t_out, t_val * contribution)
            if variance:
                var[c].index_add_(0, t_out, t_var * contribution)
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
            result[c].index_add_(0, t_out, t_val * images_t[c][t_in])
        return result

    # the seed image total, for per-frame guess scaling (forward is linear)
    seed_f = torch.from_numpy(scene_15.copy()).to(device_f)
    image_seed, _ = forward(seed_f, variance=False)
    total_seed = float(image_seed.sum())
    del seed_f, image_seed

    # --- invert every frame ---------------------------------------------------
    solutions = np.empty((num_time, num_scene), dtype=np.float32)
    chi2_final = np.empty((num_time, num_channel))
    iterations = np.empty(num_time, dtype=int)

    t_frames = time.perf_counter()
    for f in range(num_time):
        obs = _flat_images(
            electrons[{axis_time: f}], "channel", "detector_x", "detector_y"
        )
        obs_f = torch.from_numpy(obs.astype(np.float32)).to(device_f)
        obs_t = torch.from_numpy(obs.astype(np.float32)).to(device_t)

        total_obs = float(obs_f.sum())
        scene = torch.from_numpy(scene_15.copy()).to(device_t)
        scene = scene * (total_obs / total_seed)

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
        torch.cuda.synchronize()

        solutions[f] = scene.cpu().numpy()
        chi2_final[f] = chi2.cpu().numpy()
        iterations[f] = i + 1
        print(
            f"frame {f:3d}: {i + 1:3d} iterations,"
            f" chi2 {np.array2string(chi2_final[f], precision=2)},"
            f" {time.perf_counter() - t_frame:.1f} s",
            flush=True,
        )
    t_invert = time.perf_counter() - t_frames
    print(
        f"all {num_time} frames inverted: {t_invert:.0f} s"
        f" ({t_invert / num_time:.1f} s/frame)",
        flush=True,
    )

    out_dir = "/home/group/charleskankelborg/level4_gpu"
    import pathlib

    pathlib.Path(out_dir).mkdir(exist_ok=True)
    np.savez(
        out_dir + "/gpu_full_run.npz",
        solutions=solutions.reshape(
            num_time, num_wavelength, shape_input["field_x"], shape_input["field_y"]
        ),
        chi2_final=chi2_final,
        iterations=iterations,
        factor_signal=factor_signal,
        vmr=vmr,
        t_setup=t_setup,
        t_probe=t_probe,
        t_upload=t_upload,
        t_invert=t_invert,
    )
    print("products saved", flush=True)

    # --- assemble a Level_4 and render the movies -----------------------------
    import matplotlib

    matplotlib.use("Agg")

    outputs = na.ScalarArray(
        ndarray=solutions.reshape(
            num_time,
            num_wavelength,
            shape_input["field_x"],
            shape_input["field_y"],
        ).astype(np.float64)
        * (guess_unit if guess_unit is not None else 1),
        axes=("time", "wavelength", "field_x", "field_y"),
    )
    l4 = esis.data.Level_4(
        inputs=na.TemporalSpectralPositionalVectorArray(
            time=meta["time"],
            wavelength=meta["wavelength_vertices"],
            position=meta["position"],
        ),
        outputs=outputs,
        instrument=meta["instrument"],
        wavelength_center=meta["wavelength_center"],
        num_velocity=meta["num_velocity"],
        mean_chi_squared=na.ScalarArray(chi2_final, axes=("time", "channel")),
        num_iteration=na.ScalarArray(iterations, axes=("time",)),
        factor_norm=meta["factor_norm"],
        where_shadow=meta["where_shadow"],
    )

    t0 = time.perf_counter()
    index_OV = l4.num_line - 1

    ani = l4.animate_intensity(index_line=index_OV)
    ani.save(out_dir + "/full_fov_intensity.gif", writer="pillow", fps=5)
    print("full-fov intensity movie saved", flush=True)

    ani = l4.animate_doppler(index_line=index_OV)
    ani.save(out_dir + "/full_fov_doppler.gif", writer="pillow", fps=5)
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
    ani.save(out_dir + "/event_e.gif", writer="pillow", fps=5)
    print(f"event movie saved; movies: {time.perf_counter() - t0:.0f} s", flush=True)

    total = time.perf_counter() - t_start
    print(f"TOTAL: {total:.0f} s ({total / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
