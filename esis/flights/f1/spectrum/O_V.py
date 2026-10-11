r"""Properties of the :math:`\text{O\,V}\;630\,\AA` spectral line."""

import numpy as np
import astropy.units as u

__all__ = [
    "wavelength",
    "radiance",
    "radiance_coronal_hole",
    "radiance_active_region",
    "radiance_active_region_cds",
    "fwhm",
    "width_doppler",
]

wavelength = 629.732 * u.AA
"""Rest wavelength calculated by the Chianti Atomic Database :cite:p:`Dere1997`."""

radiance = 334.97 * u.erg / u.cm**2 / u.sr / u.s
"""Average quiet-sun radiance measured by :cite:t:`Vernazza1978`."""

radiance_coronal_hole = 285.77 * u.erg / u.cm**2 / u.sr / u.s
"""Average coronal-hole radiance measured by :cite:t:`Vernazza1978`."""

radiance_active_region = 1018.65 * u.erg / u.cm**2 / u.sr / u.s
"""Average active-region radiance measured by :cite:t:`Vernazza1978`."""

radiance_active_region_cds = 519.534 * u.erg / u.cm**2 / u.sr / u.s
"""
Average radiance of a weak active region, from a raster of the Coronal
Diagnostic Spectrometer :cite:p:`Harrison1995`.

It is the raster the ESIS proposal simulated its images from. Which raster
that was is not recorded.
"""

fwhm = 0.129 * u.AA
"""Average quiet-sun full width at half maximum measured by :cite:t:`Doschek2004`."""

_width = fwhm / (2 * np.sqrt(2 * np.log(2)))

_eq = u.doppler_optical(wavelength)

width_doppler = (wavelength + _width).to(u.km / u.s, equivalencies=_eq)
"""Average quiet-sun Doppler width computed from :attr:`fwhm`."""
