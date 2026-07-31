r"""Properties of the :math:`\text{O\,III}\;600\,\AA` spectral line."""

import astropy.units as u

from . import O_V

__all__ = [
    "wavelength",
    "radiance",
    "width_doppler",
]

#: Rest wavelength calculated by the Chianti Atomic Database :cite:p:`Dere1997`.
wavelength = 599.590 * u.AA

#: Average quiet-sun radiance, from the intensity of this line relative to
#: O V 630 measured by :cite:t:`Parker2022`.
radiance = 0.13 * O_V.radiance

#: Average quiet-sun Doppler width: the thermal width of oxygen at the peak
#: formation temperature of this ion (:math:`10^{4.95}` K), added in
#: quadrature to the nonthermal velocity implied by
#: :attr:`esis.flights.f1.spectrum.O_V.width_doppler`.
width_doppler = 23.0 * u.km / u.s
