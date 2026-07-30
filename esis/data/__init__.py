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
dataset:

* Bias subtraction
* Non-active pixel removal
* Dark frame subtraction
* Cosmic ray removal

Level 4
-------
Time-dependent spatial-spectral cubes reconstructed from the Level-1 images.

Where the lower levels live on the sensor grid, this level lives on the
scene grid: the spectral radiance of a set of spectral lines, each with a
Doppler window around its rest wavelength, on a spatial grid matching the
plate scale of the instrument.

The following steps are applied to the Level-1 dataset to create the Level-4
dataset:

* Negative pixel clipping
* Channel normalization
* Inversion of every frame with the multiplicative algebraic reconstruction
  technique (MART), using the fitted, linearized optical system as the
  forward model
"""

from . import abc
from . import synth
from ._level_0 import Level_0
from ._level_1 import Level_1
from ._level_4 import Level_4

__all__ = [
    "abc",
    "synth",
    "Level_0",
    "Level_1",
    "Level_4",
]
