#!/usr/bin/env python3
"""
Does the streamed assembly produce exactly what the direct one does?

The streamed path writes blocks into preallocated device tensors at
computed offsets, which is the kind of code that yields a plausible wrong
answer rather than an error.  This builds a small configuration both ways
and compares the device tensors element by element.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse

import numpy as np

import astropy.units as u
import named_arrays as na

import esis
from esis.data._level_4 import _caching

import stream_assembly
import tied_config


def direct(
    *,
    system,
    kwargs,
    windows,
    member_grids,
    position,
    num_velocity,
    inside_flat,
    det_ok_flat,
    factor_signal,
    num_channel,
    num_field_cells,
    dtype_index,
):
    """
    Assemble the tied weights the way the production driver does.

    Parameters
    ----------
    system
        The optical system.
    kwargs
        Shared caching keyword arguments.
    windows
        The window table.
    member_grids
        Per-member wavelength grids.
    position
        The scene grid.
    num_velocity
        Velocity bins per window.
    inside_flat
        Field-stop support mask.
    det_ok_flat
        Usable detector pixels per channel.
    factor_signal
        Per-element conversion factor.
    num_channel
        The number of channels.
    num_field_cells
        The number of scene cells per wavelength.
    dtype_index
        The integer type of the indices.
    """
    member = {}
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
        member[lam] = (
            _caching.weights(system, **kwargs_member),
            _caching.weights_transpose(system, **kwargs_member),
        )

    num_wavelength = len(windows) * num_velocity
    out_f = [[[], [], []] for _ in range(num_channel)]
    out_t = [[[], [], []] for _ in range(num_channel)]

    for w, (_, members, _, _) in enumerate(windows):
        tables_f, tables_t, scales = [], [], []
        for wavelength_0, scale in members:
            forward, transpose = member[float(wavelength_0.to_value(u.AA))]
            tables_f.append(forward[0].ndarray)
            tables_t.append(transpose[0].ndarray)
            scales.append(float(scale))
        flux = np.array(
            [
                s
                * sum(
                    float(t[c, j][2].sum())
                    for c in range(num_channel)
                    for j in range(num_velocity)
                )
                for t, s in zip(tables_f, scales)
            ]
        )
        alpha = flux / flux.sum()

        for c in range(num_channel):
            for j in range(num_velocity):
                k = w * num_velocity + j
                i_in = np.concatenate([t[c, j][0] for t in tables_f])
                i_out = np.concatenate([t[c, j][1] for t in tables_f])
                values = np.concatenate(
                    [
                        (s * t[c, j][2]).astype(np.float32)
                        for t, s in zip(tables_f, scales)
                    ]
                )
                keep = inside_flat[i_in] & det_ok_flat[c][i_out]
                out_f[c][0].append(
                    (i_in[keep] + dtype_index(k * num_field_cells)).astype(dtype_index)
                )
                out_f[c][1].append(i_out[keep].astype(dtype_index))
                out_f[c][2].append(
                    (values[keep] * np.float32(factor_signal[c, k])).astype(np.float32)
                )

                i_in = np.concatenate([t[c, j][0] for t in tables_t])
                i_out = np.concatenate([t[c, j][1] for t in tables_t])
                values = np.concatenate(
                    [
                        (a * t[c, j][2]).astype(np.float32)
                        for t, a in zip(tables_t, alpha)
                    ]
                )
                keep = inside_flat[i_out] & det_ok_flat[c][i_in]
                out_t[c][0].append(i_in[keep].astype(dtype_index))
                out_t[c][1].append(
                    (i_out[keep] + dtype_index(k * num_field_cells)).astype(dtype_index)
                )
                out_t[c][2].append((values[keep]).astype(np.float32))

    forward = [tuple(np.concatenate(x) for x in c) for c in out_f]
    transpose = [tuple(np.concatenate(x) for x in c) for c in out_t]
    return forward, transpose


def main() -> None:
    """Compare the two assemblies."""
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitch", type=float, default=6.0)
    parser.add_argument("--num-velocity", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

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
    num_channel = det_ok.shape[0]
    det_ok_flat = det_ok.reshape(num_channel, -1)

    # a support that keeps a nontrivial, off-centre part of the field, so a
    # mistaken offset cannot hide behind symmetry
    xx, yy = np.meshgrid(np.arange(num_field), np.arange(num_field), indexing="ij")
    inside = ((xx - 0.45 * num_field) ** 2 + (yy - 0.55 * num_field) ** 2) < (
        0.4 * num_field
    ) ** 2
    inside_flat = inside.ravel()
    print(f"num_field={num_field}, support keeps {inside.mean():.3f}", flush=True)

    rng = np.random.default_rng(4)
    factor_signal = 1 + rng.random((num_channel, num_wavelength))
    vmr = 3 + rng.random((num_channel, num_wavelength))

    common = dict(
        system=system,
        kwargs=kwargs,
        windows=windows,
        member_grids=member_grids,
        position=position,
        num_velocity=args.num_velocity,
        inside_flat=inside_flat,
        det_ok_flat=det_ok_flat,
        factor_signal=factor_signal,
    )

    expected_f, expected_t = direct(
        num_channel=num_channel,
        num_field_cells=num_field_cells,
        dtype_index=np.int32,
        **common,
    )
    print("direct assembly done", flush=True)

    got_f, got_t, ratios, shapes, count_f, count_t = stream_assembly.assemble(
        vmr=vmr,
        device_forward=device,
        device_transpose=device,
        dtype_index=np.int32,
        **common,
    )
    print("streamed assembly done", flush=True)

    worst = 0.0
    for name, expected, got in (
        ("forward", expected_f, got_f),
        ("transpose", expected_t, got_t),
    ):
        for c in range(num_channel):
            for index, label in enumerate(("index in", "index out", "value")):
                e = expected[c][index]
                g = got[c][index].cpu().numpy()
                if e.shape != g.shape:
                    print(f"  {name} channel {c} {label}: SHAPE {e.shape} vs {g.shape}")
                    worst = np.inf
                    continue
                if index < 2:
                    bad = int((e != g).sum())
                    if bad:
                        print(f"  {name} channel {c} {label}: {bad} mismatched")
                        worst = np.inf
                else:
                    denominator = np.maximum(np.abs(e).max(), 1e-30)
                    diff = np.abs(e - g).max() / denominator
                    worst = max(worst, float(diff))
    print(f"\nworst relative difference in the values: {worst:.3e}")
    print("indices identical" if worst != np.inf else "MISMATCH")

    # the variance ratio must reproduce the direct per-triple variance array
    for c in range(num_channel):
        n = expected_f[c][2].size
        assert ratios[c].numel() == n, (ratios[c].numel(), n)
    print("variance ratio arrays sized correctly")


if __name__ == "__main__":
    main()
