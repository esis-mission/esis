"""
Represent and process ESIS observations into spatial-spectral cubes.

Description of the ESIS Data Levels
===================================

Level 0
-------
The raw data gathered by the ESIS instrument, saved as FITS files.

 * Loaded into memory as a subclass of :class:`named_arrays.FunctionArray`.
 * Temperatures and voltages are converted to physical units.

Level 1
-------
A representation of the photons gathered by the sensors in physical units.

This is intended to invert the camera/sensor model and convert from
DN to photons incident on the front surface of the sensor.

The following steps are applied to the Level-0 dataset to create the Level-1
dataset.
 * Bias subtraction
 * Non-active pixel removal
 * Dark frame subtraction
 * Cosmic ray removal
"""

from . import abc
from ._level_0 import Level_0
from ._level_1 import Level_1

__all__ = [
    "abc",
    "Level_0",
    "Level_1",
]
