"""Models of the optical system specific to this flight."""

from ._requirements import requirements
from . import primaries
from . import gratings
from . import filters
from ._instruments import (
    design_full,
    design,
    design_single,
    as_built_unfocused,
    as_built,
    _as_built_focused as _as_built_focused,
    distortion_fit,
    distortion_fit_bounds,
)
from ._fits import (
    measure_window_edges,
    fit_distortion_reference,
    fit_distortion_pointing,
    acceptance,
)

__all__ = [
    "requirements",
    "primaries",
    "gratings",
    "filters",
    "design",
    "design_single",
    "design_full",
    "as_built_unfocused",
    "as_built",
    "distortion_fit",
    "distortion_fit_bounds",
    "measure_window_edges",
    "fit_distortion_reference",
    "fit_distortion_pointing",
    "acceptance",
]
