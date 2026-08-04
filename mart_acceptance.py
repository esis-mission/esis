#!/usr/bin/env python3
"""
The acceptance test: does the coalignment lower MART residuals?

Everything measured so far has been about the forward model agreeing with
itself. This asks the only question that decides whether the coalignment
was worth doing: inverting the same frame with and without the correction,
does the reconstruction fit the data better, and does the spurious
velocity trend across the field of view go away?

The correction is injected by replacing the distortion model of the
linearized system. Because `PolynomialDistortionModel` holds calibration
pairs and fits on demand, adding the fitted shift to the scene side of
those pairs gives exactly

    corrected_distort(X) = distort(X - d(X))

which is the mapping verified by re-measuring the shift field.

The two configurations must not share a cache. Injecting the correction
downstream of the joblib key would otherwise let the second run silently
reuse the first run's weights, which would produce two identical answers
and look like the correction had no effect. Each run therefore sets its
own ESIS_CACHE_DIR.
"""

import os
import pathlib
import dataclasses

import numpy as np
import astropy.units as u
import astropy.table
import named_arrays as na

CORRECTED = bool(int(os.environ.get("ESIS_CORRECTED", "0")))
INDEX_TIME = int(os.environ.get("ESIS_INDEX_TIME", "15"))
PITCH_SCENE = float(os.environ.get("ESIS_PITCH_SCENE", "4")) * u.arcsec
PITCH_VELOCITY = float(os.environ.get("ESIS_PITCH_VELOCITY", "25")) * u.km / u.s
NUM_ITERATION = int(os.environ.get("ESIS_NUM_ITERATION", "50"))

PATH_CORRECTION = (
    pathlib.Path(__file__).parent / "coalignment_20260804" / "coalignment.ecsv"
)


def install_correction() -> None:
    """Patch the linearized system so its distortion carries the coalignment."""
    import optika
    import esis
    from esis.data._level_4 import _caching
    import coalign

    table = astropy.table.QTable.read(PATH_CORRECTION, format="ascii.ecsv")
    correction = {int(row["channel"]): row for row in table}

    # the sky grid the correction was fitted on, needed to undo its
    # normalization
    instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    sky = coalign.sky_grid(instrument.system, 512)
    span_x = sky.stop.x - sky.start.x
    span_y = sky.stop.y - sky.start.y
    center_x = (sky.stop.x + sky.start.x) / 2
    center_y = (sky.stop.y + sky.start.y) / 2
    pixel_x, pixel_y = span_x / 512, span_y / 512

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
        for c, row in correction.items():
            select = na.ScalarArray(
                np.arange(na.shape(position.x)["channel"]) == c,
                axes="channel",
            )
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

        distortion_corrected = dataclasses.replace(
            distortion,
            coordinates_scene=scene_corrected,
        )
        return dataclasses.replace(linear, distortion=distortion_corrected)

    _caching.linear_system = _caching.esis.memory.cache(ignore=["system"])(corrected)
    print("installed the coalignment correction into the linear system", flush=True)


def main() -> None:
    """Invert one frame and report the residual and the velocity trend."""
    label = "corrected" if CORRECTED else "baseline"
    print(f"=== MART acceptance test: {label} ===", flush=True)
    print(f"frame {INDEX_TIME}, pitch_scene {PITCH_SCENE}, "
          f"pitch_velocity {PITCH_VELOCITY}, {NUM_ITERATION} iterations",
          flush=True)

    if CORRECTED:
        install_correction()

    import esis

    result = esis.flights.f1.data.level_4_frame(
        index_time=INDEX_TIME,
        pitch_scene=PITCH_SCENE,
        pitch_velocity=PITCH_VELOCITY,
        num_iteration=NUM_ITERATION,
        verbose=True,
    )

    chi2 = na.value(result.mean_chi_squared).ndarray
    print(f"\nmean chi squared: {np.array(chi2).ravel()}", flush=True)
    print(f"  overall {np.mean(chi2):.6f}", flush=True)

    directory = pathlib.Path(__file__).parent / "mart_acceptance_20260804"
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"chi2_{label}.npy", np.array(chi2))

    # `velocity` is the Doppler axis of the cube, not a map; the map is the
    # intensity-weighted first moment, which is what the movies plot and
    # what carries the red-to-blue trend across the field
    velocity = result.velocity_mean.to(u.km / u.s)
    intensity = result.intensity

    for index_line, name in enumerate(result.label_line or []):
        v = na.value(velocity[{result.axis_line: index_line}]).ndarray
        w = na.value(intensity[{result.axis_line: index_line}]).ndarray
        np.save(directory / f"velocity_{label}_line{index_line}.npy", v)
        np.save(directory / f"intensity_{label}_line{index_line}.npy", w)

        finite = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if finite.sum() < 10:
            continue
        print(f"  {name:20s} velocity mean {np.nanmean(v[finite]):+7.2f} "
              f"spread {np.nanstd(v[finite]):6.2f} km/s", flush=True)

    print(f"saved velocity and intensity maps, shape "
          f"{na.shape(velocity)}", flush=True)

    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
