"""
Shared configuration of the tied-window (multiplet) Level-4 inversion.

Five windows over seven spectral members: the O IV 608/610 pair tied at the
CHIANTI pure branching ratio and the Mg X doublet tied at its resonance
ratio. Window spans are ±210 km/s so the production velocity bins are
exactly 17.5 km/s (24 bins).

Experimental tooling: untracked, not part of the esis package.
"""

import numpy as np

import astropy.constants
import astropy.units as u
import named_arrays as na

RATIO_MGX = 0.52
RATIO_OIV = 0.538

LIMIT_VELOCITY = 210 * u.km / u.s


def windows():
    """Build the window table: (label, members, seed width, seed amplitude)."""
    from esis.flights.f1 import spectrum

    return [
        (
            "He I 584",
            [(spectrum.He_I.wavelength, 1.0)],
            spectrum.He_I.width_doppler,
            0.70,
        ),
        (
            "O III 600",
            [(spectrum.O_III.wavelength, 1.0)],
            spectrum.O_III.width_doppler,
            0.13,
        ),
        (
            "O IV 608+610",
            [(609.83 * u.AA, 1.0), (spectrum.O_IV.wavelength, RATIO_OIV)],
            spectrum.O_IV.width_doppler,
            0.17,
        ),
        (
            "Mg X 610+625",
            [(spectrum.Mg_X.wavelength, 1.0), (spectrum.Mg_X_625.wavelength, RATIO_MGX)],
            spectrum.Mg_X.width_doppler,
            0.25,
        ),
        (
            "O V 630",
            [(spectrum.O_V.wavelength, 1.0)],
            spectrum.O_V.width_doppler,
            1.00,
        ),
    ]


def grids(num_velocity: int):
    """
    Build the velocity grid and the per-member wavelength vertex grids.

    Parameters
    ----------
    num_velocity
        The number of velocity bins per window.
    """
    velocity = na.linspace(
        start=-LIMIT_VELOCITY,
        stop=LIMIT_VELOCITY,
        axis="wavelength",
        num=num_velocity + 1,
    )
    member_grids = {}
    for _, members, _, _ in windows():
        for wavelength_0, _ in members:
            grid = (wavelength_0 * (1 + velocity / astropy.constants.c)).to(u.AA)
            member_grids[float(wavelength_0.to_value(u.AA))] = grid
    union = np.sort(
        np.concatenate([g.ndarray.to_value(u.AA) for g in member_grids.values()])
    )
    wavelength_union = na.ScalarArray(union * u.AA, axes="wavelength")
    return velocity, member_grids, wavelength_union


def position_grid(system, pitch_scene: u.Quantity, factor_fov: float = 1.25):
    """
    Build the scene position grid for the given pitch.

    Parameters
    ----------
    system
        The (already keyed) sequential system.
    pitch_scene
        The spatial pitch of the scene grid.
    factor_fov
        The padding factor beyond the raytrace field extent.
    """
    field = system.rayfunction_default.inputs.field
    center = (field.max() + field.min()) / 2
    halfwidth = factor_fov * (field.max() - field.min()) / 2
    start, stop = center - halfwidth, center + halfwidth
    extent = stop - start
    ratio = (np.maximum(extent.x, extent.y) / pitch_scene).ndarray
    num_field = int(np.ceil(ratio.to_value(u.dimensionless_unscaled)))
    position = na.Cartesian2dVectorLinearSpace(
        start=start,
        stop=stop,
        axis=na.Cartesian2dVectorArray("field_x", "field_y"),
        num=num_field + 1,
    )
    return position, center, extent, num_field
