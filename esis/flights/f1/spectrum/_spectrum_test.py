import types
import pytest
import astropy.units as u
from esis.flights.f1 import spectrum

_unit_radiance = u.erg / u.cm**2 / u.sr / u.s


def _name(line: types.ModuleType) -> str:
    """Name a line by its module."""
    return line.__name__.rsplit(".", 1)[-1]


@pytest.mark.parametrize(
    argnames="line",
    argvalues=[
        spectrum.O_V,
        spectrum.Mg_X,
        spectrum.Mg_X_625,
        spectrum.He_I,
    ],
    ids=_name,
)
def test_radiance_by_context(line: types.ModuleType) -> None:
    # Vernazza & Reeves's coronal holes are fainter than their quiet Sun, and
    # their active regions brighter, in every one of these lines, so a value
    # copied into the wrong context breaks the order.
    assert line.radiance.unit.is_equivalent(_unit_radiance)
    assert line.radiance_coronal_hole.unit.is_equivalent(_unit_radiance)
    assert line.radiance_active_region.unit.is_equivalent(_unit_radiance)
    assert line.radiance_coronal_hole < line.radiance < line.radiance_active_region


@pytest.mark.parametrize(
    argnames="line",
    argvalues=[
        spectrum.O_V,
        spectrum.Mg_X_625,
    ],
    ids=_name,
)
def test_radiance_active_region_cds(line: types.ModuleType) -> None:
    # a weak active region, so brighter than the average quiet Sun and fainter
    # than the average active region
    radiance = line.radiance_active_region_cds
    assert radiance.unit.is_equivalent(_unit_radiance)
    assert line.radiance < radiance < line.radiance_active_region


@pytest.mark.parametrize(
    argnames="name",
    argvalues=[
        "radiance",
        "radiance_coronal_hole",
        "radiance_active_region",
    ],
)
def test_mg_x_doublet(name: str) -> None:
    # the 609 A line is the stronger of the doublet in every context
    assert getattr(spectrum.Mg_X, name) > getattr(spectrum.Mg_X_625, name)
    assert spectrum.Mg_X_625.wavelength > spectrum.Mg_X.wavelength
