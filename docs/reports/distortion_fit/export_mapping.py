#!/usr/bin/env python3
"""
Export a distortion mapping as calibration pairs for the MART acceptance test.

The inversion pipeline lives on a branch with an older instrument model, so
the mapping is handed over as the sensor position of a dense
(channel, wavelength, x, y) grid on the sky, from which a degree-2 polynomial
refit reproduces it exactly.

    python export_mapping.py out.npz [distortion_reference.ecsv]

Without a file, the reference committed to the package is exported.  With
one, its parameters are applied to the model the fit runs on.
"""

import pathlib
import sys

import numpy as np
import astropy.units as u
import named_arrays as na
import esis
from esis.flights.f1.optics._fits import _fits


def main() -> None:
    """Export the mapping named on the command line."""
    out = pathlib.Path(sys.argv[1])
    if len(sys.argv) > 2:
        parameters = esis.optics.DistortionParameters.from_file(sys.argv[2])
        instrument = parameters.to_instrument(_fits._base(None))
    else:
        instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)

    scene, _ = _fits._frame(15, 401)
    extent = scene.inputs.position
    wavelength = na.linspace(575 * u.AA, 640 * u.AA, axis="wavelength", num=27)
    x = na.linspace(extent.x.min(), extent.x.max(), axis="field_x", num=17)
    y = na.linspace(extent.y.min(), extent.y.max(), axis="field_y", num=17)
    coords = na.SpectralPositionalVectorArray(
        wavelength=wavelength,
        position=na.Cartesian2dVectorArray(x=x, y=y),
    )
    axes = ("wavelength", "field_x", "field_y")
    sx, sy = [], []
    for c in range(instrument.camera.channel.shape["channel"]):
        channel = instrument[dict(channel=c)]
        linear = channel.system.linearize(wavelength=channel.wavelength, degree=2)
        s = linear.distortion.distort(coords).position
        sx.append(np.asarray(na.value(s.x).ndarray_aligned(axes)))
        sy.append(np.asarray(na.value(s.y).ndarray_aligned(axes)))
        print(
            f"channel {c}: x {np.nanmin(sx[-1]):.0f}..{np.nanmax(sx[-1]):.0f} px",
            flush=True,
        )
    np.savez(
        out,
        wavelength_AA=np.asarray(na.value(wavelength).ndarray),
        x_arcsec=np.asarray(na.value(x).ndarray),
        y_arcsec=np.asarray(na.value(y).ndarray),
        sensor_x_px=np.stack(sx),
        sensor_y_px=np.stack(sy),
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
