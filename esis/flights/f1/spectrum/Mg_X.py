r"""Properties of the :math:`\text{Mg,\textsc{x}} 609\,\AA` spectral line."""

import numpy as np
import astropy.units as u

__all__ = [
    "wavelength",
    "radiance",
    "fwhm",
]

#: Rest wavelength calculated by the Chianti Atomic Database :cite:p:`Dere1997`.
wavelength = 609.793 * u.AA

#: Average quiet-sun radiance measured by :cite:t:`Vernazza1978`.
radiance = 125.05 * u.erg / u.cm**2 / u.sr / u.s

_fwhm = 0.138 * u.mAA

_width = _fwhm / (2 * np.sqrt(2 * np.log(2)))

#: Average quiet-sun Doppler width measured by :cite:t:`Doschek2004`.
width_doppler = _width.to(u.km / u.s, equivalencies=u.doppler_optical(wavelength))