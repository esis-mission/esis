r"""
Properties of the :math:`\text{He\,II}\;304\,\AA` spectral line.

This line is not observed by ESIS. It is one of the brightest lines in the
extreme ultraviolet, and the gratings diffract it in second order onto the
same part of the detector as :math:`607.6\,\AA` in first order, inside the
passband. The multilayer coating of the gratings is designed to reflect as
little of it as possible.
"""

import astropy.units as u

__all__ = [
    "wavelength",
]

wavelength = 303.785 * u.AA
r"""
Rest wavelength calculated by the Chianti Atomic Database :cite:p:`Dere1997`.

The line is the :math:`2p \to 1s` doublet of hydrogen-like helium. This is
the stronger of its two components, from :math:`2p\,^2P_{3/2}`; the other
lies :math:`0.001\,\AA` away.
"""
