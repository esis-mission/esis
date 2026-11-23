"""Helpers shared by the tests of the distortion fit."""

import astropy.units as u
import named_arrays as na
import optika
import esis

__all__ = [
    "idealized",
    "wavelength_lines",
    "wavelengths_alignment",
    "wavelengths_lines_named",
]


def idealized(instrument: esis.optics.Instrument) -> esis.optics.Instrument:
    """
    Replace the material models of an instrument with ideal ones.

    The distortion fit measures geometry, not throughput, so the coatings,
    filter and sensor are made ideal to save raytracing time, and the
    uncertainty wrappers are stripped so that the distortion parameters are
    plain scalars.

    Parameters
    ----------
    instrument
        The instrument model to idealize (modified in place and returned).
    """
    instrument.grating.material = optika.materials.Mirror()
    instrument.primary_mirror.material = optika.materials.Mirror()
    instrument.filter.material = None
    instrument.camera.sensor.material = optika.sensors.materials.IdealSensorMaterial()
    coefficients = instrument.grating.rulings.spacing.coefficients
    for k in list(coefficients):
        coefficients[k] = na.nominal(coefficients[k])
    grating = instrument.grating
    grating.translation = na.nominal(grating.translation)
    grating.yaw = na.nominal(grating.yaw)
    grating.pitch = na.nominal(grating.pitch)
    grating.roll = na.nominal(grating.roll)
    return instrument


def wavelength_lines() -> na.ScalarArray:
    """Return three bright lines in the passband of the test instrument."""
    return na.ScalarArray(
        u.Quantity(list(wavelengths_lines_named().values())),
        axes="wavelength",
    )


def wavelengths_alignment() -> dict[str, u.Quantity]:
    """Return two of :func:`wavelength_lines`, by name, to align channels on."""
    lines = wavelengths_lines_named()
    return {name: lines[name] for name in ("He I", "O V")}


def wavelengths_lines_named() -> dict[str, u.Quantity]:
    """Return the lines of :func:`wavelength_lines`, by name."""
    spectrum = esis.flights.f1.spectrum
    return {
        "He I": spectrum.He_I.wavelength,
        "Mg X": spectrum.Mg_X.wavelength,
        "O V": spectrum.O_V.wavelength,
    }
