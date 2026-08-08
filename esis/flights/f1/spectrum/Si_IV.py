r"""
Properties of the :math:`\text{Si\,IV}\;1394\,\AA` spectral line.

This line is not observed by ESIS. It is the line observed by IRIS which is
shifted and scaled onto :math:`\text{O\,V}\;630\,\AA` by
:func:`esis.data.synth.scene_iris` to synthesize a scene.
"""

import numpy as np
import astropy.units as u

__all__ = [
    "wavelength",
    "radiance",
    "fwhm",
    "width_doppler",
]

#: Rest wavelength calculated by the Chianti Atomic Database :cite:p:`Dere1997`.
wavelength = 1393.755 * u.AA

#: Average quiet-sun radiance measured by :cite:t:`Brekke1993` with the High
#: Resolution Telescope and Spectrograph.
#:
#: Taken from the literature rather than measured from the raster being
#: scaled, so that the ratio of this to the radiance of the simulated line is
#: a property of the two lines rather than of the observation. Measuring it
#: from the raster would divide the observation by itself, which would give
#: every scene the same brightness however bright the Sun was that day.
radiance = 280.0 * u.erg / u.cm**2 / u.sr / u.s

#: Average quiet-sun full width at half maximum, measured from the median
#: spectral profile of an IRIS raster. The median is used rather than the mean
#: so that the value is not biased by the bright, broad profiles of transient
#: events.
fwhm = 0.151 * u.AA

_width = fwhm / (2 * np.sqrt(2 * np.log(2)))

_eq = u.doppler_optical(wavelength)

#: Average quiet-sun Doppler width computed from :attr:`fwhm`.
width_doppler = (wavelength + _width).to(u.km / u.s, equivalencies=_eq)
