r"""Properties of the :math:`\text{O\,IV}\;608\,\AA` spectral line."""

import astropy.units as u

from . import O_V

__all__ = [
    "wavelength",
    "radiance",
    "width_doppler",
]

#: Rest wavelength calculated by the Chianti Atomic Database :cite:p:`Dere1997`.
wavelength = 608.398 * u.AA

#: Average quiet-sun radiance, from the intensity of this line relative to
#: O V 630 measured by :cite:t:`Parker2022`.
radiance = 0.06 * O_V.radiance

#: Average quiet-sun Doppler width: the thermal width of oxygen at the peak
#: formation temperature of this ion (:math:`10^{5.18}` K), added in
#: quadrature to the nonthermal velocity implied by
#: :attr:`esis.flights.f1.spectrum.O_V.width_doppler`.
width_doppler = 24.4 * u.km / u.s
