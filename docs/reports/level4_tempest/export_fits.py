#!/usr/bin/env python3
"""
Export a tied-window GPU run to self-describing FITS.

The GPU driver saves a bare ``npz`` whose coordinates only make sense
alongside this repository.  This rebuilds the coordinates from the
instrument model and the tied-window configuration, wraps the run in an
:class:`esis.data.Level_4`, and writes it with
:meth:`esis.data.Level_4.to_fits`, so the product can be read by anyone
with astropy or sunpy.

The tied windows are stored back-to-back with no gap cells, while the
Level-4 convention separates adjacent windows by one cell so a single
monotonic wavelength axis can describe them all.  The cube is therefore
re-laid-out onto the standard convention, with the inter-window cells
left empty; ``to_fits`` omits them from the file.

The photon-equivalent cube, if the run recorded one, is appended as one
extra extension per window, so signal-to-noise can be read off the file
as ``sqrt(photons)``.

Experimental tooling: untracked, not part of the esis package.
"""

import argparse
import pathlib
import time as _time

import numpy as np
import scipy.ndimage

import astropy.constants
import astropy.io.fits
import astropy.units as u
import named_arrays as na

import esis

import tied_config

DIR = pathlib.Path("/home/group/charleskankelborg/level4_gpu_tied")


def main() -> None:
    """Convert the run to FITS."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pitch", type=float, default=0.75)
    parser.add_argument("--num-velocity", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-coregister",
        action="store_true",
        help="ship the frames on the instrument grid, as reconstructed",
    )
    args = parser.parse_args()

    t0 = _time.perf_counter()
    z = np.load(args.npz, mmap_mode="r")
    solutions = z["solutions"]
    num_time, num_cell_total, num_x, num_y = solutions.shape

    windows = tied_config.windows()
    num_line = len(windows)
    num_velocity = args.num_velocity
    assert num_cell_total == num_line * num_velocity

    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    system = instrument.system
    position, _, _, num_field = tied_config.position_grid(system, args.pitch * u.arcsec)
    assert num_field == num_x

    a = esis.flights.f1.data.level_1()
    time = a.inputs.time
    if a.axis_channel in time.shape:
        time = time.mean(a.axis_channel)

    # the standard convention: one spurious cell between adjacent windows
    num_cell_standard = num_line * (num_velocity + 1) - 1
    limit = tied_config.LIMIT_VELOCITY

    velocity = np.linspace(
        -limit.to_value(u.km / u.s),
        limit.to_value(u.km / u.s),
        num_velocity + 1,
    )
    rest = u.Quantity([w[1][0][0] for w in windows])
    vertices = []
    for i in range(num_line):
        lam = rest[i].to_value(u.AA) * (
            1 + velocity / astropy.constants.c.to_value(u.km / u.s)
        )
        vertices.extend(lam)
    vertices = np.array(vertices)
    assert vertices.size == num_cell_standard + 1
    # NOT monotonic in general: the O IV and Mg X windows are centred 0.04 A
    # apart and so overlap almost entirely, which is the whole point of tying
    # them. Nothing downstream needs a single ordered spectrum -- every
    # velocity axis is computed within its own window, against its own rest
    # wavelength -- so the overlap is harmless here.

    print(f"loading and re-laying out the cube ({num_time} frames)", flush=True)
    outputs = np.zeros(
        (num_time, num_cell_standard, num_x, num_y),
        dtype=np.float32,
    )
    for i in range(num_line):
        start = i * (num_velocity + 1)
        outputs[:, start : start + num_velocity] = solutions[
            :, i * num_velocity : (i + 1) * num_velocity
        ]

    # the GPU driver works in the backprojection unit of the linearized
    # system, the same unit the package product carries
    unit = u.ph / (u.AA * u.s * u.cm**2 * u.deg**2)

    level_4 = esis.data.Level_4(
        inputs=na.TemporalSpectralPositionalVectorArray(
            time=time,
            wavelength=na.ScalarArray(vertices * u.AA, axes="wavelength"),
            position=position,
        ),
        outputs=na.ScalarArray(
            outputs * unit, axes=("time", "wavelength", "field_x", "field_y")
        ),
        instrument=instrument,
        wavelength_center=na.ScalarArray(rest, axes="line"),
        label_line=[w[0] for w in windows],
        members_line=[
            [(u.Quantity(wavelength), float(ratio)) for wavelength, ratio in w[1]]
            for w in windows
        ],
        num_velocity=num_velocity,
        mean_chi_squared=na.ScalarArray(
            np.asarray(z["chi2_final"]), axes=("time", "channel")
        ),
        num_iteration=na.ScalarArray(np.asarray(z["iterations"]), axes=("time",)),
        factor_norm=na.ScalarArray(
            np.asarray(z["factor_norm"]), axes=("time", "channel")
        ),
        where_shadow=a.where_shadow(),
    )

    from esis.data._level_4 import _fits

    if not args.no_coregister:
        print("coregistering onto a common sky frame", flush=True)
        level_4 = level_4.coregister()
        offset = level_4.drift_applied.to_value(u.arcsec)
        print(
            f"  removed drift: x {offset[:, 0].min():+.2f} to"
            f" {offset[:, 0].max():+.2f}, y {offset[:, 1].min():+.2f} to"
            f" {offset[:, 1].max():+.2f} arcsec"
            f" ({_time.perf_counter() - t0:.0f} s)",
            flush=True,
        )

    print(f"writing {args.out} (one file per line)", flush=True)
    level_4.to_fits(args.out, overwrite=args.overwrite, split=True)
    print(f"  wrote in {_time.perf_counter() - t0:.0f} s", flush=True)

    keys = list(z.files)
    if "photons" in keys or "factor_photon" in keys:
        print("appending the photon extension to each line", flush=True)
        for i in range(num_line):
            sl = slice(i * num_velocity, (i + 1) * num_velocity)
            if "photons" in keys:
                data = np.asarray(z["photons"][:, sl], dtype=np.float32)
            else:
                data = np.asarray(
                    solutions[:, sl] * z["factor_photon"][sl],
                    dtype=np.float32,
                )
            if level_4.drift_applied is not None:
                pitch = float(
                    position.x.ndarray.to_value(u.arcsec)[1]
                    - position.x.ndarray.to_value(u.arcsec)[0]
                )
                shift = level_4.drift_applied.to_value(u.arcsec) / pitch
                for t in range(data.shape[0]):
                    for k in range(data.shape[1]):
                        data[t, k] = scipy.ndimage.shift(
                            data[t, k], -shift[t], order=1, mode="constant", cval=0
                        )
            hdu = astropy.io.fits.ImageHDU(np.moveaxis(data, 2, 3))
            hdu.header["EXTNAME"] = "PHOTONS"
            hdu.header["BUNIT"] = ("photon", "detected photons per scene pixel")
            hdu.header.add_comment("Detected-photon equivalent of this window,")
            hdu.header.add_comment("summed over all four camera channels, so the")
            hdu.header.add_comment("signal-to-noise ratio is sqrt of this value.")
            hdu.header.add_comment("Axes match the radiance extension.")
            file = pathlib.Path(args.out) / _fits.line_filename(level_4, i)
            astropy.io.fits.append(file, hdu.data, hdu.header)

    total = 0
    for file in sorted(pathlib.Path(args.out).glob("*.fits")):
        size = file.stat().st_size / 2**30
        total += size
        print(f"  {file.name}: {size:.1f} GiB", flush=True)
    print(f"DONE: {total:.1f} GiB in {_time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
