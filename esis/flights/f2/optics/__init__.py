"""Models of the optical system specific to this flight."""

from . import materials
from ._instruments import design_proposed

__all__ = [
    "materials",
    "design_proposed",
]
