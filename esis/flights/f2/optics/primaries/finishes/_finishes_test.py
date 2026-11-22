import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import esis


@pytest.mark.parametrize("serial_number", ["SN2", "SN3"])
@pytest.mark.parametrize(
    argnames="objective,spacing,rtol_radius",
    argvalues=[
        ("1.4X", 5.9245 * u.um, 0.02),
        ("20X", 0.40741 * u.um, 0.1),
    ],
)
def test_finish_measurement(
    serial_number: str,
    objective: str,
    spacing: u.Quantity,
    rtol_radius: float,
) -> None:
    """
    The field is the height of the surface of a primary mirror.

    It is a square field of 1024 by 1024 pixels, measured everywhere but at
    one pixel at most. The heights still hold the curvature of the mirror, so
    the radius of curvature which a polynomial of the second degree fits to
    them is that of the design, to within the noise of the field. That checks
    the scale of the heights against that of the positions.
    """
    finishes = esis.flights.f2.optics.primaries.finishes
    finish = finishes.finish_measurement(
        serial_number,
        objective,
        axis_x="a",
        axis_y="b",
    )
    assert isinstance(finish, na.FunctionArray)

    position = finish.inputs
    height = finish.outputs
    assert isinstance(position, na.Cartesian2dVectorArray)
    assert height.shape == dict(b=1024, a=1024)
    assert height.unit == u.nm
    step = position.x[dict(a=1)] - position.x[dict(a=0)]
    assert np.isclose(step, spacing, rtol=1e-4)

    measured = np.isfinite(height)
    assert np.sum(~measured) <= 1

    fit = na.PolynomialFitFunctionArray.from_degree(
        inputs=position,
        outputs=np.where(measured, height, 0 * u.nm),
        degree=2,
        axis_polynomial=("a", "b"),
        where_polynomial=measured,
    )
    primary = esis.flights.f2.optics.design(num_distribution=0).primary_mirror
    radius = -2 * primary.sag.focal_length
    for term in ["x*x", "y*y"]:
        curvature = 2 * fit.coefficients.components[term]
        assert np.isclose(1 / curvature, radius, rtol=rtol_radius), term


def test_finish_measurement_arguments() -> None:
    """Only the mirrors and the objectives which were measured are accepted."""
    finish_measurement = esis.flights.f2.optics.primaries.finishes.finish_measurement
    with pytest.raises(ValueError, match="SN1"):
        finish_measurement("SN1", "20X")
    with pytest.raises(ValueError, match="10X"):
        finish_measurement("SN2", "10X")
