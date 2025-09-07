"""Model the ESIS optical system and invert images captured during flight."""

foo = 1
"a test variable"

bar = "a"

from ._caching import memory
from . import optics
from . import nsroc
from . import data
from . import flights

__all__ = [
    "foo",
    "bar",
    "memory",
    "optics",
    "nsroc",
    "data",
    "flights",
]
