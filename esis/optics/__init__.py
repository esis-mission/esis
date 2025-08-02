"""
A generalized model of the ESIS optical system.
"""

from . import abc
from ._central_obscurations import CentralObscuration

__all__ = [
    "abc",
    "CentralObscuration",
]
