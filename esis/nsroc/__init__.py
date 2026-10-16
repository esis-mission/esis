"""
A representation of the vehicle and trajectory information provided by NSROC.

NSROC is the NASA Sounding Rocket Operations Contract and is the organization
responsible for providing the launch services for ESIS.
"""

from ._timeline import Timeline
from ._sparcs import Sparcs

__all__ = [
    "Timeline",
    "Sparcs",
]
