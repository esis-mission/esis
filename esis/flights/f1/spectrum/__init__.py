"""
Properties of the spectral lines observed during this flight.

The radiances measured by :cite:t:`Vernazza1978` are averages over each kind
of region they observed: the quiet Sun, coronal holes, and active regions.
Their Table 1 gives three measures of each, and these are the third, the
absolute intensity with the nearby continuum removed.
"""

from . import O_V
from . import Mg_X
from . import Mg_X_625
from . import He_I
from . import He_II
from . import Si_IV

__all__ = [
    "O_V",
    "Mg_X",
    "Mg_X_625",
    "He_I",
    "He_II",
    "Si_IV",
]
