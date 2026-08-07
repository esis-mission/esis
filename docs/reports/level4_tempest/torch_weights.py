#!/usr/bin/env python3
"""
Prototype: GPU (torch) build of the conservative regridding weights.

The expensive step of the weights build is conservative polygon clipping:
every distorted scene cell (a quadrilateral in pixel coordinates) is
clipped against the unit pixel lattice and the overlap areas become the
sparse weights. This port keeps optika's distortion and radiometry on the
CPU (exact by construction, seconds of work) and moves only the clipping
to the GPU: a vectorized Sutherland–Hodgman clip of each cell against its
candidate pixels, in float64.

``--validate`` builds one member both ways and compares against the CPU
triples: pair-set intersection, per-cell row sums (conservation), and a
functional apply on random scenes. ``--bench`` times the production-scale
member.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse
import time

import numpy as np

import astropy.units as u
import named_arrays as na

import esis
from esis.data._level_4 import _caching

import tied_config


def cpu_preamble(pitch: float, num_velocity: int, member: float):
    """
    Reproduce optika's weights() inputs on the CPU.

    Returns the distorted cell-corner positions in pixel coordinates, the
    per-cell radiometric factor, the pixel-lattice origin, and the CPU
    weights entry for comparison.

    Parameters
    ----------
    pitch
        The scene pitch in arcsec.
    num_velocity
        The number of velocity bins per window.
    member
        The member rest wavelength in Angstrom.
    """
    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system
    key = _caching.key_system(system)
    code = _caching.code_state()

    _, member_grids, wavelength_union = tied_config.grids(num_velocity)
    position, _, _, _ = tied_config.position_grid(system, pitch * u.arcsec)
    grid = member_grids[member]

    kwargs = dict(key=key, wavelength=wavelength_union, degree=2, code=code)
    linear = _caching.linear_system(system, **kwargs)

    coordinates = na.SpectralPositionalVectorArray(
        wavelength=grid,
        position=position,
    )
    coordinates = coordinates.cell_centers("wavelength")
    position_sensor = linear.distortion.distort(coordinates).position

    coordinates_cell = coordinates.cell_centers(("field_x", "field_y"))
    if linear.vignetting is not None:
        weights_vignetting = linear.vignetting(coordinates_cell)
    else:
        weights_vignetting = 1
    # mirror what optika's LinearSystem.weights does. Note that optika
    # before PR #152 landed stripped the effective area with `.value`, in
    # its natural mm^2, while declaring cm^2 -- weights built against that
    # version are a factor of 100 larger than these
    field_stop = linear.field_stop
    if field_stop is not None:
        weights_stop = field_stop(
            position=na.Cartesian3dVectorArray(
                x=coordinates_cell.position.x,
                y=coordinates_cell.position.y,
            ),
        )
    else:
        weights_stop = 1
    weights_area = linear.area_effective(coordinates_cell.wavelength)
    weights_input = (
        weights_vignetting * weights_stop * weights_area.to_value(linear.weights_unit)
    )

    order = ("channel", "wavelength", "field_x", "field_y")

    def _nd(a):
        src = [a.axes.index(ax) for ax in order if ax in a.axes]
        dst = list(range(len(src)))
        out = np.moveaxis(np.asarray(a.ndarray), src, dst)
        return out

    corners_x = _nd(position_sensor.x.to(u.pix)).astype(np.float64)
    corners_y = _nd(position_sensor.y.to(u.pix)).astype(np.float64)
    shape_cells = dict(
        zip(
            order,
            (
                corners_x.shape[0],
                corners_x.shape[1],
                corners_x.shape[2] - 1,
                corners_x.shape[3] - 1,
            ),
        )
    )
    factor = _nd(weights_input.broadcast_to(shape_cells)).astype(np.float64)

    sensor = linear.coordinates_sensor
    grid_x = np.asarray(sensor.x.ndarray.to_value(u.pix), dtype=np.float64)
    grid_y = np.asarray(sensor.y.ndarray.to_value(u.pix), dtype=np.float64)
    assert np.allclose(np.diff(grid_x), 1) and np.allclose(np.diff(grid_y), 1)
    origin = (grid_x[0], grid_y[0])
    num_pixels = (grid_x.size - 1, grid_y.size - 1)

    kwargs_member = dict(
        coordinates_scene=na.SpectralPositionalVectorArray(
            wavelength=grid, position=position
        ),
        axis_wavelength="wavelength",
        axis_field=("field_x", "field_y"),
        **kwargs,
    )
    cpu = _caching._weights(system, **kwargs_member)

    return corners_x, corners_y, factor, origin, num_pixels, cpu


def _clip_halfplane(poly, count, axis, sign, bound):
    """
    Clip padded polygons against a half-plane, Sutherland–Hodgman style.

    Parameters
    ----------
    poly
        The padded polygon vertices, shape ``(M, V, 2)``.
    count
        The number of valid vertices per polygon, shape ``(M,)``.
    axis
        The coordinate axis of the half-plane boundary (0 or 1).
    sign
        Keep vertices where ``sign * (coordinate - bound) >= 0``.
    bound
        The boundary coordinate.
    """
    import torch

    M, V, _ = poly.shape
    device = poly.device
    slots = torch.arange(V, device=device)

    valid = slots[None, :] < count[:, None]
    index_next = (slots[None, :] + 1) % torch.clamp(count[:, None], min=1)
    vertex = poly
    vertex_next = torch.gather(poly, 1, index_next.unsqueeze(-1).expand(M, V, 2))

    bound = bound[:, None] if bound.dim() == 1 else bound
    distance = sign * (vertex[..., axis] - bound)
    distance_next = sign * (vertex_next[..., axis] - bound)
    inside = distance >= 0
    inside_next = distance_next >= 0

    emit_vertex = inside & valid
    emit_cross = (inside != inside_next) & valid

    t = distance / (distance - distance_next)
    t = torch.nan_to_num(t, nan=0.0)
    crossing = vertex + t.unsqueeze(-1) * (vertex_next - vertex)

    # interleave [vertex, crossing] per slot, order-preserving compaction
    mask = torch.stack([emit_vertex, emit_cross], dim=2).reshape(M, 2 * V)
    points = torch.stack([vertex, crossing], dim=2).reshape(M, 2 * V, 2)
    key = torch.where(
        mask,
        torch.arange(2 * V, device=device)[None, :].expand(M, 2 * V),
        torch.full((M, 2 * V), 2 * V, device=device),
    )
    order = torch.argsort(key, dim=1, stable=True)
    points = torch.gather(points, 1, order.unsqueeze(-1).expand(M, 2 * V, 2))
    count_out = mask.sum(dim=1)
    return points[:, : V + 2], count_out


def _area(poly, count):
    """
    Compute the absolute shoelace area of padded polygons.

    Parameters
    ----------
    poly
        The padded polygon vertices, shape ``(M, V, 2)``.
    count
        The number of valid vertices per polygon.
    """
    import torch

    M, V, _ = poly.shape
    slots = torch.arange(V, device=poly.device)
    valid = slots[None, :] < count[:, None]
    # the padding beyond `count` holds garbage (including NaN from masked
    # crossing computations), and NaN * False poisons the masked sum
    poly = torch.where(valid.unsqueeze(-1), poly, torch.zeros_like(poly))
    index_next = (slots[None, :] + 1) % torch.clamp(count[:, None], min=1)
    vertex_next = torch.gather(poly, 1, index_next.unsqueeze(-1).expand(M, V, 2))
    cross = poly[..., 0] * vertex_next[..., 1] - vertex_next[..., 0] * poly[..., 1]
    return 0.5 * torch.abs((cross * valid).sum(dim=1))


def gpu_build_element(
    cx, cy, factor, origin, num_pixels, device, chunk=100_000, dtype=None
):
    """
    Build one (channel, wavelength) element's triples on the GPU.

    Parameters
    ----------
    cx
        The cell-corner x coordinates, shape ``(nx + 1, ny + 1)``, pixels.
    cy
        The cell-corner y coordinates, same shape.
    factor
        The per-cell radiometric factor, shape ``(nx, ny)``.
    origin
        The pixel-lattice origin ``(x0, y0)``.
    num_pixels
        The lattice size ``(npx, npy)``.
    device
        The torch device.
    chunk
        The number of cells per processing chunk.
    dtype
        The floating-point type of the clipping, :obj:`None` for float64.
        Each quadrilateral is shifted to its own candidate-pixel block
        before clipping, so the coordinates are of order one rather than
        of order the detector width, which is what lets float32 hold
        enough significant figures to matter here.
    """
    import torch

    if dtype is None:
        dtype = torch.float64

    nx, ny = factor.shape
    npx, npy = num_pixels

    quads = np.stack(
        [
            np.stack([cx[:-1, :-1], cy[:-1, :-1]], axis=-1),
            np.stack([cx[1:, :-1], cy[1:, :-1]], axis=-1),
            np.stack([cx[1:, 1:], cy[1:, 1:]], axis=-1),
            np.stack([cx[:-1, 1:], cy[:-1, 1:]], axis=-1),
        ],
        axis=2,
    ).reshape(-1, 4, 2)
    quads[..., 0] -= origin[0]
    quads[..., 1] -= origin[1]
    # conservative convention: each fragment is the area *fraction* of its
    # input cell, times the radiometric factor
    x = quads[..., 0]
    y = quads[..., 1]
    area_quad = 0.5 * np.abs(
        (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)
    )
    factor_flat = factor.reshape(-1) / area_quad

    out_in, out_px, out_val = [], [], []
    for lo in range(0, quads.shape[0], chunk):
        q = torch.from_numpy(quads[lo : lo + chunk]).to(device)
        f = torch.from_numpy(factor_flat[lo : lo + chunk]).to(device)
        M = q.shape[0]

        i0 = torch.floor(q[..., 0].min(dim=1).values).long()
        j0 = torch.floor(q[..., 1].min(dim=1).values).long()
        i1 = torch.floor(q[..., 0].max(dim=1).values - 1e-12).long()
        j1 = torch.floor(q[..., 1].max(dim=1).values - 1e-12).long()
        Kx = int((i1 - i0).max().item()) + 1
        Ky = int((j1 - j0).max().item()) + 1

        dx = torch.arange(Kx, device=device)
        dy = torch.arange(Ky, device=device)
        px = (i0[:, None, None] + dx[None, :, None]).expand(M, Kx, Ky)
        py = (j0[:, None, None] + dy[None, None, :]).expand(M, Kx, Ky)
        px = px.reshape(M * Kx * Ky)
        py = py.reshape(M * Kx * Ky)

        # shift each quadrilateral onto its own candidate block: the clip
        # then runs on coordinates of order Kx rather than of order the
        # detector width, so the shoelace differences keep their
        # significant figures even in single precision
        offset = torch.stack([i0, j0], dim=-1).to(q.dtype)
        q = (q - offset[:, None, :]).to(dtype)

        poly = torch.zeros((M, 10, 2), dtype=dtype, device=device)
        poly[:, :4] = q
        poly = poly.unsqueeze(1).expand(M, Kx * Ky, 10, 2).reshape(-1, 10, 2)
        num = torch.full((M * Kx * Ky,), 4, dtype=torch.long, device=device)

        x_lo = dx[None, :, None].expand(M, Kx, Ky).reshape(M * Kx * Ky).to(dtype)
        y_lo = dy[None, None, :].expand(M, Kx, Ky).reshape(M * Kx * Ky).to(dtype)
        poly, num = _clip_halfplane(poly, num, 0, +1.0, x_lo)
        poly, num = _clip_halfplane(poly, num, 0, -1.0, x_lo + 1)
        poly, num = _clip_halfplane(poly, num, 1, +1.0, y_lo)
        poly, num = _clip_halfplane(poly, num, 1, -1.0, y_lo + 1)
        area = _area(poly, num)

        cell = (
            torch.arange(lo, lo + M, device=device)[:, None]
            .expand(M, Kx * Ky)
            .reshape(-1)
        )
        keep = (area > 0) & (px >= 0) & (px < npx) & (py >= 0) & (py < npy)
        out_in.append(cell[keep])
        out_px.append((px[keep] * npy + py[keep]))
        out_val.append(area[keep] * f.repeat_interleave(Kx * Ky)[keep])

    import torch as _t

    return (
        _t.cat(out_in).cpu().numpy(),
        _t.cat(out_px).cpu().numpy(),
        _t.cat(out_val).cpu().numpy(),
    )


def compare(cpu_element, gpu_triples, num_input: int, num_output: int, rng):
    """
    Compare a CPU weights element against the GPU triples.

    Parameters
    ----------
    cpu_element
        The CPU ``(indices_input, indices_output, values)`` triple.
    gpu_triples
        The GPU triple in the same convention.
    num_input
        The flattened input size.
    num_output
        The flattened output size.
    rng
        A numpy random generator for the functional test.
    """
    ci, co, cv = (np.asarray(x) for x in cpu_element)
    gi, go, gv = gpu_triples

    def rowsum(idx, val):
        out = np.zeros(num_input)
        np.add.at(out, idx, val)
        return out

    rs_cpu = rowsum(ci.astype(np.int64), cv.astype(np.float64))
    rs_gpu = rowsum(gi, gv)
    nz = rs_cpu > 0
    ratio = rs_gpu[nz] / rs_cpu[nz]
    print(
        f"  row-sum ratio gpu/cpu: median {np.median(ratio):.6f}"
        f" spread {np.std(ratio):.2e}"
    )
    print(
        f"  diag: median row-sum cpu {np.median(rs_cpu[nz]):.6e}"
        f" gpu {np.median(rs_gpu[nz]):.6e}"
    )
    scale = np.median(ratio)
    gv = gv / scale

    key_cpu = ci.astype(np.int64) * num_output + co.astype(np.int64)
    key_gpu = gi * num_output + go
    order_c = np.argsort(key_cpu, kind="stable")
    order_g = np.argsort(key_gpu, kind="stable")
    key_cpu, cv2 = key_cpu[order_c], cv[order_c].astype(np.float64)
    key_gpu, gv2 = key_gpu[order_g], gv[order_g]
    common, idx_c, idx_g = np.intersect1d(key_cpu, key_gpu, return_indices=True)
    frac_common = common.size / max(key_cpu.size, 1)
    diff = np.abs(gv2[idx_g] - cv2[idx_c])
    rel = diff / np.maximum(np.abs(cv2[idx_c]), 1e-300)
    weight_missing = (
        cv2.sum() - cv2[idx_c].sum() + np.abs(gv2.sum() - gv2[idx_g].sum())
    ) / cv2.sum()
    print(
        f"  pairs: cpu {key_cpu.size}, gpu {key_gpu.size},"
        f" common {frac_common:.6f};"
        f" |dvalue| max {diff.max():.2e} (rel {rel.max():.2e});"
        f" weight in unmatched pairs {weight_missing:.2e}"
    )

    scene = rng.random(num_input)
    img_cpu = np.zeros(num_output)
    np.add.at(img_cpu, co.astype(np.int64), cv.astype(np.float64) * scene[ci])
    img_gpu = np.zeros(num_output)
    np.add.at(img_gpu, go, gv * scene[gi])
    denom = np.abs(img_cpu).max()
    print(
        f"  functional apply max |dimage|/max: {np.abs(img_gpu - img_cpu).max() / denom:.2e}"
    )
    return scale


def main() -> None:
    """Run the validation or benchmark."""
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitch", type=float, default=1.5)
    parser.add_argument("--num-velocity", type=int, default=14)
    parser.add_argument("--member", type=float, default=629.732)
    parser.add_argument("--elements", type=int, default=4)
    parser.add_argument(
        "--float32",
        action="store_true",
        help="clip in single precision instead of double",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.float32 else torch.float64
    print(f"device: {device} | clip dtype: {dtype}", flush=True)
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}", flush=True)

    t0 = time.perf_counter()
    corners_x, corners_y, factor, origin, num_pixels, cpu = cpu_preamble(
        args.pitch, args.num_velocity, args.member
    )
    print(f"cpu preamble: {time.perf_counter() - t0:.0f} s", flush=True)

    table, shape_in, shape_out = cpu
    num_channel = shape_in["channel"]
    num_cells = shape_in["field_x"] * shape_in["field_y"]
    num_detector = shape_out["detector_x"] * shape_out["detector_y"]
    view = np.moveaxis(table.ndarray, table.axes.index("channel"), 0)

    rng = np.random.default_rng(42)
    num_wavelength = corners_x.shape[1]
    todo = [(c, j) for c in range(num_channel) for j in range(num_wavelength)][
        : args.elements
    ]

    t_gpu = 0.0
    for c, j in todo:
        print(f"element channel {c} wavelength {j}:", flush=True)
        cx = corners_x[c, j]
        cy = corners_y[c, j]
        x4 = np.stack([cx[:-1, :-1], cx[1:, :-1], cx[1:, 1:], cx[:-1, 1:]], axis=-1)
        y4 = np.stack([cy[:-1, :-1], cy[1:, :-1], cy[1:, 1:], cy[:-1, 1:]], axis=-1)
        area_quad = 0.5 * np.abs(
            (x4 * np.roll(y4, -1, axis=-1) - np.roll(x4, -1, axis=-1) * y4).sum(axis=-1)
        )
        print(
            f"  diag: median factor {np.median(factor[c, j]):.6e},"
            f" median quad area {np.median(area_quad):.4f} px^2,"
            f" factor/area {np.median(factor[c, j] / area_quad):.6e}",
            flush=True,
        )
        t0 = time.perf_counter()
        triples = gpu_build_element(
            corners_x[c, j],
            corners_y[c, j],
            factor[c, j],
            origin,
            num_pixels,
            device,
            dtype=dtype,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        t_gpu += dt
        print(f"  gpu build: {dt:.2f} s ({triples[0].size / 1e6:.1f}M triples)")
        compare(view[c].reshape(-1)[j], triples, num_cells, num_detector, rng)

    total = num_channel * num_wavelength
    print(
        f"gpu per element: {t_gpu / len(todo):.2f} s"
        f" (× {total} elements ≈ {t_gpu / len(todo) * total:.0f} s per member)",
        flush=True,
    )


if __name__ == "__main__":
    main()
