r"""Properties of the :math:`\text{O\,V}\;630\,\AA` spectral line."""

import numpy as np
import astropy.units as u

__all__ = [
    "wavelength",
    "radiance",
    "width_doppler",
]

#: Rest wavelength calculated by the Chianti Atomic Database :cite:p:`Dere1997`.
wavelength = 629.732 * u.AA

#: Average quiet-sun radiance measured by :cite:t:`Vernazza1978`.
radiance = 334.97 * u.erg / u.cm**2 / u.sr / u.s

_fwhm = 0.129 * u.AA

_width = _fwhm / (2 * np.sqrt(2 * np.log(2)))

_eq = u.doppler_optical(wavelength)

#: Average quiet-sun Doppler width measured by :cite:t:`Doschek2004`.
width_doppler = (wavelength + _width).to(u.km / u.s, equivalencies=_eq)
