"""Models and data associated with the proposed second flight in summer 2027."""

from ._wavelength import wavelength_Ne_VII, wavelength_Si_XII
from . import optics

__all__ = [
    "wavelength_Ne_VII",
    "wavelength_Si_XII",
    "optics",
]
