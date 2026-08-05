#!/usr/bin/env python3
"""
Install the inter-channel coalignment into the linearized system.

The correction was fitted in the sky plane against channel 1 and verified
by re-measuring the shift field. It is applied here by adding the fitted
shift to the *scene* side of the distortion model's calibration pairs.
Because `PolynomialDistortionModel` holds those pairs and fits on demand,
that gives exactly

    corrected_distort(X) = distort(X - d(X))

Import and call :func:`install` before anything builds weights. The
injection happens downstream of the joblib key, so a run using it must be
given its own ESIS_CACHE_DIR: sharing a cache with an uncorrected run
would silently reuse that run's weights.
"""

import pathlib
import dataclasses

import numpy as np
import astropy.units as u
import astropy.table
import named_arrays as na

ROOT = pathlib.Path(__file__).resolve().parent.parent
PATH_CORRECTION = ROOT / "coalignment_20260804" / "coalignment.ecsv"
NUM_SKY = 512


def install() -> None:
    """Patch `_caching.linear_system` so its distortion carries the coalignment."""
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import esis
    from esis.data._level_4 import _caching
    import coalign

    table = astropy.table.QTable.read(PATH_CORRECTION, format="ascii.ecsv")
    correction = {int(row["channel"]): row for row in table}

    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    sky = coalign.sky_grid(instrument.system, NUM_SKY)
    span_x = sky.stop.x - sky.start.x
    span_y = sky.stop.y - sky.start.y
    center_x = (sky.stop.x + sky.start.x) / 2
    center_y = (sky.stop.y + sky.start.y) / 2
    pixel_x, pixel_y = span_x / NUM_SKY, span_y / NUM_SKY

    names = list(coalign.LINES)
    wavelength_mean = np.mean([w.to_value(u.AA) for w in coalign.LINES.values()])
    wavelength_half = abs(coalign.LINES[names[1]].to_value(u.AA) - wavelength_mean)

    original = _caching.linear_system.__wrapped__

    def corrected(system, key, wavelength, degree, code):
        linear = original(system, key, wavelength, degree, code)
        distortion = linear.distortion

        scene = distortion.coordinates_scene
        position = scene.position
        cx = 2 * (position.x - center_x) / span_x
        cy = 2 * (position.y - center_y) / span_y
        dw = (scene.wavelength - wavelength_mean * u.AA) / (wavelength_half * u.AA)

        shift_x = 0 * position.x
        shift_y = 0 * position.y
        num_channel = na.shape(position.x)["channel"]
        for c, row in correction.items():
            select = na.ScalarArray(np.arange(num_channel) == c, axes="channel")
            dxc = (
                row["dx_0"] + row["dx_x"] * cx + row["dx_y"] * cy + row["dx_w"] * dw
            ) * pixel_x
            dyc = (
                row["dy_0"] + row["dy_x"] * cx + row["dy_y"] * cy + row["dy_w"] * dw
            ) * pixel_y
            shift_x = shift_x + np.where(select, dxc, 0 * dxc)
            shift_y = shift_y + np.where(select, dyc, 0 * dyc)

        scene_corrected = na.SpectralPositionalVectorArray(
            wavelength=scene.wavelength,
            position=na.Cartesian2dVectorArray(
                x=position.x + shift_x,
                y=position.y + shift_y,
            ),
        )
        return dataclasses.replace(
            linear,
            distortion=dataclasses.replace(
                distortion,
                coordinates_scene=scene_corrected,
            ),
        )

    _caching.linear_system = _caching.esis.memory.cache(ignore=["system"])(corrected)
    print("coalignment correction installed into the linear system", flush=True)
