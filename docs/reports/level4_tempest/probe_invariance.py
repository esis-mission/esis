#!/usr/bin/env python3
"""
Are the noise-probe factors independent of the scene grid?

``_probe_factors`` extracts, for each channel and velocity cell, the
conversion from raw weight sums to calibrated electrons and the
variance-to-mean ratio.  Both come from ratios of quantities that are
linear in the weights, so they ought not to care how finely the scene is
sampled -- which would let the probe run once on a cheap coarse grid
instead of on the production grid it is being applied to.

That is the assumption the streaming assembly needs, because streaming
never holds an assembled weights table for the probe to use.  This checks
it by probing two coarse grids and comparing.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse

import numpy as np

import astropy.units as u
import named_arrays as na

import esis
from esis.data._level_4 import _caching

import tied_config
from gpu_mart import _probe_factors


def build(pitch: float, num_velocity: int):
    """
    Assemble the tied instrument on a given grid, without any masking.

    Parameters
    ----------
    pitch
        The scene pitch in arcsec.
    num_velocity
        The velocity bins per window.
    """
    import ctis

    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system
    key = _caching.key_system(system)
    code = _caching.code_state()

    windows = tied_config.windows()
    num_window = len(windows)
    num_wavelength = num_window * num_velocity

    _, member_grids, union = tied_config.grids(num_velocity)
    position, _, _, num_field = tied_config.position_grid(system, pitch * u.arcsec)

    kwargs = dict(key=key, wavelength=union, degree=2, code=code)
    linear = _caching.linear_system(system, **kwargs)

    member = {}
    for lam, grid in member_grids.items():
        coordinates = na.SpectralPositionalVectorArray(
            wavelength=grid, position=position
        )
        member[lam] = _caching.weights(
            system,
            coordinates_scene=coordinates,
            axis_wavelength="wavelength",
            axis_field=("field_x", "field_y"),
            **kwargs,
        )

    fw0 = next(iter(member.values()))
    num_channel = fw0[1]["channel"]

    arr = np.empty((num_channel, num_wavelength), dtype=object)
    for w, (_, members, _, _) in enumerate(windows):
        tables, scales = [], []
        for wavelength_0, scale in members:
            tables.append(member[float(wavelength_0.to_value(u.AA))][0].ndarray)
            scales.append(scale)
        for c in range(num_channel):
            for j in range(num_velocity):
                k = w * num_velocity + j
                arr[c, k] = (
                    np.concatenate([t[c, j][0] for t in tables]),
                    np.concatenate([t[c, j][1] for t in tables]),
                    np.concatenate(
                        [
                            (s * t[c, j][2]).astype(np.float32)
                            for t, s in zip(tables, scales)
                        ]
                    ),
                )

    shape_in = dict(fw0[1])
    shape_in["wavelength"] = num_wavelength
    shape_out = dict(fw0[2])
    shape_out["wavelength"] = num_wavelength

    a = esis.flights.f1.data.level_1()
    vertices = []
    for _, members, _, _ in windows:
        grid = member_grids[float(members[0][0].to_value(u.AA))]
        grid = grid.ndarray.to_value(u.AA)
        if not vertices:
            vertices = list(grid)
        else:
            vertices.extend(vertices[-1] + np.cumsum(np.diff(grid)))

    instrument_mart = ctis.instruments.OptikaInstrument(
        system=linear,
        coordinates_scene=na.SpectralPositionalVectorArray(
            wavelength=na.ScalarArray(np.array(vertices) * u.AA, axes="wavelength"),
            position=position,
        ),
        channel=a[{a.axis_time: 15}].channel,
        axis_channel="channel",
        axis_wavelength="wavelength",
        axis_scene_xy=("field_x", "field_y"),
    )
    instrument_mart.weights = (
        na.ScalarArray(arr, axes=("channel", "wavelength")),
        shape_in,
        shape_out,
    )
    return instrument_mart, shape_in, shape_out, num_field


def main() -> None:
    """Compare the probe on two grids."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-velocity", type=int, default=4)
    parser.add_argument("--pitch", type=float, nargs=2, default=[6.0, 4.0])
    args = parser.parse_args()

    results = []
    for pitch in args.pitch:
        instrument, shape_in, shape_out, num_field = build(pitch, args.num_velocity)
        print(f"probing {pitch:g} arcsec (num_field={num_field})", flush=True)
        results.append(_probe_factors(instrument, shape_in, shape_out))
        del instrument

    (fs_a, vmr_a), (fs_b, vmr_b) = results
    ok = (fs_a != 0) & (fs_b != 0)
    for name, a, b in (("factor_signal", fs_a, fs_b), ("vmr", vmr_a, vmr_b)):
        rel = np.abs(a[ok] - b[ok]) / np.abs(b[ok])
        print(
            f"{name}: max relative difference between grids {rel.max():.3e},"
            f" median {np.median(rel):.3e}"
        )


if __name__ == "__main__":
    main()
