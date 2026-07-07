import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
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
    axis_txy = (axis_time, axis_x, axis_y)
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
    assert np.allclose(radiance.outputs.mean(), O_V.radiance, rtol=1e-1)

    spectrum = result.mean(axis_txy)
    fwhm = na.pdf.fwhm(
        x=spectrum.inputs.wavelength,
        f=spectrum.outputs,
        axis=axis_velocity,
    )
    assert np.allclose(fwhm, O_V.fwhm, rtol=1e-1)
