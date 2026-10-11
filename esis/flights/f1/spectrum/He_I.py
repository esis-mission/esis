r"""Properties of the :math:`\text{He\,I}\;584\,\AA` spectral line."""

import astropy.units as u

__all__ = [
    "wavelength",
    "radiance",
    "radiance_coronal_hole",
    "radiance_active_region",
    "width_doppler",
]

wavelength = 584.334 * u.AA
"""Rest wavelength calculated by the Chianti Atomic Database :cite:p:`Dere1997`."""

radiance = 544.98 * u.erg / u.cm**2 / u.sr / u.s
"""Average quiet-sun radiance measured by :cite:t:`Vernazza1978`."""

radiance_coronal_hole = 250.12 * u.erg / u.cm**2 / u.sr / u.s
"""Average coronal-hole radiance measured by :cite:t:`Vernazza1978`."""

radiance_active_region = 5470.68 * u.erg / u.cm**2 / u.sr / u.s
"""Average active-region radiance measured by :cite:t:`Vernazza1978`."""

width_doppler = 20 * u.km / u.s
"""Average quiet-sun Doppler width measured by :cite:t:`Peter1999`."""
