#!/usr/bin/env python3
"""
Assemble the tied weights onto the GPU without ever holding them on the host.

The direct approach materializes the problem three times in host memory:
every member's weights at once (~168 GB at the production grid), then the
tied and filtered copy, then a contiguous copy per channel to hand to the
device.  That is hundreds of gigabytes of single-threaded copying, and it
is what forces production onto a fat node.

This walks the members one at a time instead.  A first pass records how
many weights survive the field stop and the shadow mask, which fixes the
size and the offset of every block; a second pass fills a preallocated
device tensor block by block, freeing each member as it goes.  Host memory
never holds more than one member.

The noise-probe factors cannot be measured this way, since the probe needs
an assembled table.  They are measured once on a coarse grid instead:
``vmr`` is independent of the scene grid, and ``factor_signal`` scales as
the area of a scene cell, both verified by ``probe_invariance``.

Experimental tooling: untracked, not part of the esis package.
"""

import time

import numpy as np

import astropy.units as u
import named_arrays as na

from esis.data._level_4 import _caching


def _members_of(windows):
    """
    List every distinct member wavelength, with where it contributes.

    Parameters
    ----------
    windows
        The window table from ``tied_config``.
    """
    result = {}
    for w, (_, members, _, _) in enumerate(windows):
        for index, (wavelength_0, scale) in enumerate(members):
            key = float(wavelength_0.to_value(u.AA))
            result.setdefault(key, []).append((w, index, float(scale)))
    return result


def assemble(
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
    vmr,
    device_forward,
    device_transpose,
    dtype_index=np.int32,
    verbose=True,
):
    """
    Build the device-resident tied weights, one member at a time.

    Parameters
    ----------
    system
        The sequential optical system the weights were cached against.
    kwargs
        The caching keyword arguments shared by every member.
    windows
        The window table from ``tied_config``.
    member_grids
        The wavelength grid of each member.
    position
        The scene grid vertices.
    num_velocity
        The velocity bins per window.
    inside_flat
        The flattened field-stop support mask.
    det_ok_flat
        The per-channel flattened mask of usable detector pixels.
    factor_signal
        The per-element conversion to calibrated electrons.
    vmr
        The per-element variance-to-mean ratio.
    device_forward
        The device to hold the forward weights.
    device_transpose
        The device to hold the transpose weights.
    dtype_index
        The integer type of the stored indices.
    verbose
        Whether to report progress.
    """
    import torch

    num_window = len(windows)
    num_wavelength = num_window * num_velocity
    num_channel = det_ok_flat.shape[0]
    num_field_cells = inside_flat.size

    contributions = _members_of(windows)
    order = sorted(contributions)

    def _load(wavelength_0):
        """
        Load one member's weights in both directions.

        Parameters
        ----------
        wavelength_0
            The member's rest wavelength in Angstrom.
        """
        coordinates = na.SpectralPositionalVectorArray(
            wavelength=member_grids[wavelength_0],
            position=position,
        )
        kwargs_member = dict(
            coordinates_scene=coordinates,
            axis_wavelength="wavelength",
            axis_field=("field_x", "field_y"),
            **kwargs,
        )
        forward = _caching.weights(system, **kwargs_member)
        transpose = _caching.weights_transpose(system, **kwargs_member)
        return forward, transpose

    # --- pass one: how many weights survive, and where each block starts ---
    t0 = time.perf_counter()
    count_forward = np.zeros((num_channel, num_wavelength, 2), dtype=np.int64)
    count_transpose = np.zeros((num_channel, num_wavelength, 2), dtype=np.int64)
    flux = {}
    shape_in = shape_out = shape_in_t = shape_out_t = None

    for wavelength_0 in order:
        forward, transpose = _load(wavelength_0)
        if shape_in is None:
            shape_in, shape_out = dict(forward[1]), dict(forward[2])
            shape_in_t, shape_out_t = dict(transpose[1]), dict(transpose[2])
        table_f = forward[0].ndarray
        table_t = transpose[0].ndarray

        total = 0.0
        for w, index, scale in contributions[wavelength_0]:
            for c in range(num_channel):
                for j in range(num_velocity):
                    k = w * num_velocity + j
                    i_in, i_out, values = table_f[c, j]
                    keep = inside_flat[i_in] & det_ok_flat[c][i_out]
                    count_forward[c, k, index] = int(keep.sum())
                    total += float(values.sum())

                    i_in, i_out, _ = table_t[c, j]
                    keep = inside_flat[i_out] & det_ok_flat[c][i_in]
                    count_transpose[c, k, index] = int(keep.sum())
        flux[wavelength_0] = total * contributions[wavelength_0][0][2]
        del forward, transpose, table_f, table_t

    if verbose:
        print(
            f"  pass 1 (counts): {time.perf_counter() - t0:.0f} s,"
            f" {count_forward.sum() / 1e9:.3f}G forward triples",
            flush=True,
        )

    # the transpose shares each window's flux between its members
    alpha = {}
    for w, (_, members, _, _) in enumerate(windows):
        total = sum(flux[float(m[0].to_value(u.AA))] for m in members)
        for index, (wavelength_0, _) in enumerate(members):
            alpha[(w, index)] = flux[float(wavelength_0.to_value(u.AA))] / total

    # --- preallocate, one contiguous block per channel ---
    offset_forward = np.zeros((num_channel, num_wavelength, 2), dtype=np.int64)
    offset_transpose = np.zeros((num_channel, num_wavelength, 2), dtype=np.int64)
    for c in range(num_channel):
        offset_forward[c] = np.concatenate(
            [[0], np.cumsum(count_forward[c].reshape(-1))[:-1]]
        ).reshape(num_wavelength, 2)
        offset_transpose[c] = np.concatenate(
            [[0], np.cumsum(count_transpose[c].reshape(-1))[:-1]]
        ).reshape(num_wavelength, 2)

    torch_index = torch.int32 if dtype_index == np.int32 else torch.int64
    forward_gpu, transpose_gpu = [], []
    for c in range(num_channel):
        n = int(count_forward[c].sum())
        forward_gpu.append(
            (
                torch.empty(n, dtype=torch_index, device=device_forward),
                torch.empty(n, dtype=torch_index, device=device_forward),
                torch.empty(n, dtype=torch.float32, device=device_forward),
            )
        )
        n = int(count_transpose[c].sum())
        transpose_gpu.append(
            (
                torch.empty(n, dtype=torch_index, device=device_transpose),
                torch.empty(n, dtype=torch_index, device=device_transpose),
                torch.empty(n, dtype=torch.float32, device=device_transpose),
            )
        )

    # --- pass two: fill the blocks, one member at a time ---
    t0 = time.perf_counter()
    for wavelength_0 in order:
        forward, transpose = _load(wavelength_0)
        table_f = forward[0].ndarray
        table_t = transpose[0].ndarray

        for w, index, scale in contributions[wavelength_0]:
            share = alpha[(w, index)]
            for c in range(num_channel):
                for j in range(num_velocity):
                    k = w * num_velocity + j

                    i_in, i_out, values = table_f[c, j]
                    keep = inside_flat[i_in] & det_ok_flat[c][i_out]
                    lo = int(offset_forward[c, k, index])
                    hi = lo + int(count_forward[c, k, index])
                    if hi > lo:
                        block = forward_gpu[c]
                        block[0][lo:hi] = torch.from_numpy(
                            (i_in[keep] + dtype_index(k * num_field_cells)).astype(
                                dtype_index
                            )
                        ).to(device_forward)
                        block[1][lo:hi] = torch.from_numpy(
                            i_out[keep].astype(dtype_index)
                        ).to(device_forward)
                        block[2][lo:hi] = torch.from_numpy(
                            (
                                values[keep] * np.float32(scale * factor_signal[c, k])
                            ).astype(np.float32)
                        ).to(device_forward)

                    i_in, i_out, values = table_t[c, j]
                    keep = inside_flat[i_out] & det_ok_flat[c][i_in]
                    lo = int(offset_transpose[c, k, index])
                    hi = lo + int(count_transpose[c, k, index])
                    if hi > lo:
                        block = transpose_gpu[c]
                        block[0][lo:hi] = torch.from_numpy(
                            i_in[keep].astype(dtype_index)
                        ).to(device_transpose)
                        block[1][lo:hi] = torch.from_numpy(
                            (i_out[keep] + dtype_index(k * num_field_cells)).astype(
                                dtype_index
                            )
                        ).to(device_transpose)
                        block[2][lo:hi] = torch.from_numpy(
                            (values[keep] * np.float32(share)).astype(np.float32)
                        ).to(device_transpose)

        del forward, transpose, table_f, table_t

    if verbose:
        print(f"  pass 2 (fill): {time.perf_counter() - t0:.0f} s", flush=True)

    # the variance is the signal scaled by a per-element constant, so the
    # forward carries a per-triple ratio rather than a second value array
    ratio = torch.empty(0)
    ratios = []
    for c in range(num_channel):
        per_triple = np.empty(int(count_forward[c].sum()), dtype=np.float32)
        for k in range(num_wavelength):
            for index in range(2):
                lo = int(offset_forward[c, k, index])
                hi = lo + int(count_forward[c, k, index])
                per_triple[lo:hi] = np.float32(vmr[c, k])
        ratios.append(torch.from_numpy(per_triple).to(device_forward))
    del ratio

    shapes = (shape_in, shape_out, shape_in_t, shape_out_t)
    return forward_gpu, transpose_gpu, ratios, shapes, count_forward, count_transpose
