r"""Properties of the :math:`\text{He\,I}\;584\,\AA` spectral line."""

import astropy.units as u

__all__ = [
    "wavelength",
    "radiance",
    "width_doppler",
]

#: Rest wavelength calculated by the Chianti Atomic Database :cite:p:`Dere1997`.
wavelength = 584.334 * u.AA

#: Average quiet-sun radiance measured by :cite:t:`Vernazza1978`.
radiance = 544.98 * u.erg / u.cm**2 / u.sr / u.s

#: Average quiet-sun Doppler width measured by :cite:t:`Peter1999`.
width_doppler = 20 * u.km / u.s
