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
    field_stop_vertices,
    _as_built_focused as _as_built_focused,
    distortion_fit,
    distortion_fit_bounds,
)
from ._fits import (
    measure_window_edges,
    measure_field_stop,
    fit_distortion_reference,
    realign_distortion_reference,
    fit_distortion_pointing,
    fit_defocus_history,
    frame_parameters,
    fit_coregistration,
    apply_coregistration,
    acceptance,
    field_stop_converged,
    measure_clocking,
    record_distortion_fit,
    distortion_fit_table,
    plot_distortion_flight,
    plot_coalignment_tiles,
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
    "field_stop_vertices",
    "distortion_fit",
    "distortion_fit_bounds",
    "measure_window_edges",
    "measure_field_stop",
    "fit_distortion_reference",
    "realign_distortion_reference",
    "fit_distortion_pointing",
    "fit_defocus_history",
    "frame_parameters",
    "fit_coregistration",
    "apply_coregistration",
    "acceptance",
    "field_stop_converged",
    "measure_clocking",
    "record_distortion_fit",
    "distortion_fit_table",
    "plot_distortion_flight",
    "plot_coalignment_tiles",
]
