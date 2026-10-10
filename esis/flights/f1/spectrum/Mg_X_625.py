r"""
Properties of the :math:`\text{Mg\,X}\;625\,\AA` spectral line.

The weaker line of the :math:`\text{Mg\,X}` doublet. The stronger is
:mod:`esis.flights.f1.spectrum.Mg_X`. This one lies :math:`4.8\,\AA` short of
:math:`\text{O\,V}\;630\,\AA`.
"""

import astropy.units as u

__all__ = [
    "wavelength",
    "radiance",
    "radiance_coronal_hole",
    "radiance_active_region",
    "radiance_active_region_cds",
]

wavelength = 624.941 * u.AA
"""Rest wavelength calculated by the Chianti Atomic Database :cite:p:`Dere1997`."""

radiance = 51.43 * u.erg / u.cm**2 / u.sr / u.s
"""Average quiet-sun radiance measured by :cite:t:`Vernazza1978`."""

radiance_coronal_hole = 2.62 * u.erg / u.cm**2 / u.sr / u.s
"""Average coronal-hole radiance measured by :cite:t:`Vernazza1978`."""

radiance_active_region = 397.64 * u.erg / u.cm**2 / u.sr / u.s
"""Average active-region radiance measured by :cite:t:`Vernazza1978`."""

radiance_active_region_cds = 239.249 * u.erg / u.cm**2 / u.sr / u.s
"""
Average radiance of a weak active region, from a raster of the Coronal
Diagnostic Spectrometer :cite:p:`Harrison1995`.

It is the same raster as
:attr:`esis.flights.f1.spectrum.O_V.radiance_active_region_cds`.
"""
