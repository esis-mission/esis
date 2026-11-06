import collections.abc
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import esis


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_ruling_design(num_distribution: int) -> None:
    r = esis.flights.f1.optics.gratings.rulings.ruling_design(
        num_distribution=num_distribution,
    )
    assert isinstance(r, optika.rulings.AbstractRulings)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_ruling_measurement(num_distribution: int) -> None:
    r = esis.flights.f1.optics.gratings.rulings.ruling_measurement(
        num_distribution=num_distribution,
    )
    assert isinstance(r, optika.rulings.AbstractRulings)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_ruling_measurement_efficiency(num_distribution: int) -> None:
    """
    A grating with these rulings and its coating reflects what was measured.

    Its efficiency is raytraced at the wavelengths and the angle of incidence
    of the measurement.
    """
    gratings = esis.flights.f1.optics.gratings
    measurement = gratings.efficiencies.efficiency_vs_wavelength()
    angle = measurement.inputs.direction
    surface = optika.surfaces.Surface(
        material=gratings.materials.multilayer_fit(),
        rulings=gratings.rulings.ruling_measurement(
            num_distribution=num_distribution,
        ),
    )
    rays = optika.rays.RayVectorArray(
        wavelength=measurement.inputs.wavelength,
        position=na.Cartesian3dVectorArray() * u.mm,
        direction=na.Cartesian3dVectorArray(
            x=np.sin(angle),
            y=0,
            z=np.cos(angle),
        ),
    )
    intensity = surface.propagate_rays(rays).intensity
    assert np.allclose(intensity, measurement.outputs, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    argnames="ruling",
    argvalues=[
        esis.flights.f1.optics.gratings.rulings.ruling_design,
        esis.flights.f1.optics.gratings.rulings.ruling_measurement,
    ],
)
def test_width_density(
    ruling: collections.abc.Callable[..., optika.rulings.AbstractRulings],
) -> None:
    """The tolerance on the spacing is that of 1 per millimeter on the density."""
    spacing = ruling(num_distribution=11).spacing.coefficients[0]
    density = 1 / spacing.nominal
    width_density = spacing.width * np.square(density)
    assert np.isclose(width_density, 1 / u.mm)
