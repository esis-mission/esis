r"""Properties of the :math:`\text{Mg\,X}\;625\,\AA` spectral line."""

import astropy.units as u

from . import Mg_X
from . import O_V

__all__ = [
    "wavelength",
    "radiance",
    "width_doppler",
]

#: Rest wavelength calculated by the Chianti Atomic Database :cite:p:`Dere1997`.
wavelength = 624.941 * u.AA

#: Average quiet-sun radiance, from the intensity of this line relative to
#: O V 630 measured by :cite:t:`Parker2022`.
radiance = 0.13 * O_V.radiance

#: Average quiet-sun Doppler width, equal to
#: :attr:`esis.flights.f1.spectrum.Mg_X.width_doppler` since both lines of
#: the resonance doublet are emitted by the same ion at the same temperature.
width_doppler = Mg_X.width_doppler
