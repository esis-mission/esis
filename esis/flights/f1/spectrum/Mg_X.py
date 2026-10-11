r"""
Properties of the :math:`\text{Mg\,X}\;609\,\AA` spectral line.

The stronger line of the :math:`\text{Mg\,X}` doublet. The weaker is
:mod:`esis.flights.f1.spectrum.Mg_X_625`.
"""

import numpy as np
import astropy.units as u

__all__ = [
    "wavelength",
    "radiance",
    "radiance_coronal_hole",
    "radiance_active_region",
    "fwhm",
    "width_doppler",
]

wavelength = 609.793 * u.AA
"""Rest wavelength calculated by the Chianti Atomic Database :cite:p:`Dere1997`."""

radiance = 125.05 * u.erg / u.cm**2 / u.sr / u.s
"""Average quiet-sun radiance measured by :cite:t:`Vernazza1978`."""

radiance_coronal_hole = 13.33 * u.erg / u.cm**2 / u.sr / u.s
"""Average coronal-hole radiance measured by :cite:t:`Vernazza1978`."""

radiance_active_region = 805.49 * u.erg / u.cm**2 / u.sr / u.s
"""Average active-region radiance measured by :cite:t:`Vernazza1978`."""

fwhm = 0.138 * u.AA
"""Average quiet-sun full width at half maximum measured by :cite:t:`Doschek2004`."""

_width = fwhm / (2 * np.sqrt(2 * np.log(2)))

_eq = u.doppler_optical(wavelength)

width_doppler = (wavelength + _width).to(u.km / u.s, equivalencies=_eq)
"""Average quiet-sun Doppler width computed from :attr:`fwhm`."""
