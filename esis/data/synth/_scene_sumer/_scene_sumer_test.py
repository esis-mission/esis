import functools
import dataclasses
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import soho
import esis
from esis.flights.f1.spectrum import O_V

_unit = u.erg / (u.cm**2 * u.sr * u.nm * u.s)
"""The unit of the radiance of the scenes."""


@functools.cache
def _observation() -> soho.sumer.SpectrographObservation:
    """Load the 4 MB file of the release which the tests use."""
    return soho.sumer.SpectrographObservation.from_level1(
        path=esis.flights.f1.data.path_sumer("2005-04-15T15:43"),
        wavelength=O_V.wavelength,
        order=2,
    )


def _radiance(scene: soho.sumer.SpectrographObservation) -> np.ndarray:
    """Return the radiance of a scene in :data:`_unit`, as a plain array."""
    axes = (scene.axis_detector_x, scene.axis_detector_y, scene.axis_wavelength)
    return scene.outputs.ndarray_aligned(axes).to_value(_unit)


def _speed(scene: soho.sumer.SpectrographObservation) -> u.Quantity:
    """Return the speed at the middle of each column of a scene."""
    velocity = scene.inputs.velocity.cell_centers(scene.axis_wavelength)
    return np.abs(velocity.ndarray_aligned((scene.axis_wavelength,)))


def test_scene_sumer_observation() -> None:
    observation = _observation()
    result = esis.data.synth.scene_sumer_observation(observation)
    assert isinstance(result, soho.sumer.SpectrographObservation)
    assert result.outputs.unit.is_equivalent(_unit)
    assert result.axis_wavelength == observation.axis_wavelength
    assert np.all(_speed(result) <= 125 * u.km / u.s)
    assert np.any(_speed(observation) > 125 * u.km / u.s)
    num_x = observation.outputs.shape[observation.axis_detector_x]
    assert result.outputs.shape[result.axis_detector_x] == num_x


def test_unchanged() -> None:
    """Without a cut or a continuum, the scene is the observation in other units."""
    observation = _observation()
    result = esis.data.synth.scene_sumer_observation(
        observation=observation,
        velocity_max=None,
        velocity_continuum=None,
    )
    expected = _radiance(observation)
    assert np.allclose(_radiance(result), expected, equal_nan=True)


def test_continuum() -> None:
    """The continuum is one level for each pixel, the same in every column."""
    observation = _observation()
    with_continuum = esis.data.synth.scene_sumer_observation(observation)
    without = esis.data.synth.scene_sumer_observation(
        observation=observation,
        velocity_continuum=None,
    )
    level = _radiance(without) - _radiance(with_continuum)
    finite = np.all(np.isfinite(level), axis=~0)
    level = level[finite]
    assert level.size
    assert np.allclose(
        level, level[..., :1], rtol=1e-6, atol=1e-6 * np.abs(level).max()
    )
    assert np.median(level[..., 0]) > 0


def test_fill() -> None:
    """The voxels SUMER did not observe are given the radiance asked for."""
    observation = _observation()
    outputs = observation.outputs.copy()
    index = {
        observation.axis_detector_x: 3,
        observation.axis_detector_y: 50,
    }
    outputs[index] = np.nan * outputs.unit
    observation = dataclasses.replace(observation, outputs=outputs)

    left = esis.data.synth.scene_sumer_observation(observation)
    fill = 2 * _unit
    filled = esis.data.synth.scene_sumer_observation(observation, fill=fill)

    assert np.all(np.isnan(na.as_named_array(left.outputs)[index].ndarray))
    assert np.all(filled.outputs[index] == fill)
    assert np.all(np.isfinite(filled.outputs))


def test_velocity_max_empty() -> None:
    with pytest.raises(ValueError, match="No column"):
        esis.data.synth.scene_sumer_observation(
            observation=_observation(),
            velocity_max=-1 * u.km / u.s,
        )
