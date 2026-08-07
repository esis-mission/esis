#!/usr/bin/env python3
"""
Build the per-member weights of the tied-window configuration (CPU job).

Builds (or loads from cache) the union linearization and the forward and
transpose weights of every spectral member, for the given scene pitch and
velocity binning. Run on a fat CPU node before the GPU inversion.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse
import time

import astropy.units as u
import named_arrays as na

import esis
from esis.data._level_4 import _caching

import tied_config


def main() -> None:
    """Build the member weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitch", type=float, required=True, help="arcsec")
    parser.add_argument("--num-velocity", type=int, required=True)
    args = parser.parse_args()

    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system
    key = _caching.key_system(system)
    code = _caching.code_state()

    velocity, member_grids, wavelength_union = tied_config.grids(args.num_velocity)
    position, center, extent, num_field = tied_config.position_grid(
        system, args.pitch * u.arcsec
    )
    print(f"pitch {args.pitch}, num_velocity {args.num_velocity}: "
          f"num_field={num_field}, members={len(member_grids)}", flush=True)

    kwargs = dict(key=key, wavelength=wavelength_union, degree=2, code=code)

    t0 = time.perf_counter()
    _caching.linear_system(system, **kwargs)
    print(f"linearize: {time.perf_counter() - t0:.0f} s", flush=True)

    for lam, grid in member_grids.items():
        coordinates = na.SpectralPositionalVectorArray(
            wavelength=grid,
            position=position,
        )
        kwargs_member = dict(
            coordinates_scene=coordinates,
            axis_wavelength="wavelength",
            axis_field=("field_x", "field_y"),
            **kwargs,
        )
        t0 = time.perf_counter()
        forward = _caching.weights(system, **kwargs_member)
        transpose = _caching.weights_transpose(system, **kwargs_member)
        num = sum(t[2].size for t in forward[0].ndarray.reshape(-1))
        print(
            f"member {lam:.2f}: {time.perf_counter() - t0:.0f} s"
            f" ({num / 1e6:.0f}M forward triples)",
            flush=True,
        )
        del forward, transpose

    print("BUILD DONE", flush=True)


if __name__ == "__main__":
    main()
