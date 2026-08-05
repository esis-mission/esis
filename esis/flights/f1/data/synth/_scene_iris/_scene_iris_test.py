import pytest
import numpy as np
import astropy.units as u
import esis
from esis.flights.f1.spectrum import O_V


@pytest.mark.parametrize("background_removal", [None, "trim_mean"])
@pytest.mark.parametrize(
    argnames="velocity_max",
    argvalues=[
        100 * u.km / u.s,
        10000 * u.km / u.s,
    ],
)
def test_scene_iris(
    background_removal: str,
    velocity_max: None | u.Quantity,
):
    axis_time = "time"
    axis_x = "detector_x"
    axis_y = "detector_y"
    axis_velocity = "velocity"

    try:
        result = esis.flights.f1.data.synth.scene_iris(
            time_start="2014-10-13 04:11",
            axis_time=axis_time,
            axis_detector_x=axis_x,
            axis_detector_y=axis_y,
            axis_velocity=axis_velocity,
            limit=1,
            velocity_max=velocity_max,
            background_removal=background_removal,
        )
    except OSError as e:  # pragma: nocover
        pytest.skip(f"IRIS archive is unreachable, skipping live-network test: {e}")

    assert result.outputs.unit.is_equivalent(u.erg / u.cm**2 / u.sr / u.AA / u.s)

    assert np.all(result.inputs.wavelength_rest == O_V.wavelength)

    radiance = result.integrate(component="wavelength", axis=axis_velocity)
    assert np.all(np.isfinite(radiance.outputs))
    assert np.all(radiance.outputs > 0)

    # The limit is a velocity in the simulated scene, not in the observations,
    # so it has to hold after the velocity has been scaled. It applies to the
    # center of each cell, the edges of the outermost cells lie beyond it.
    velocity = result.inputs.velocity.cell_centers(axis_velocity)
    assert np.all(np.abs(velocity) <= velocity_max)
