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

_sigma_line = 10 * u.km / u.s
"""The standard deviation of the line of the synthetic spectra."""

_sigma_neighbour = 10 * u.km / u.s
"""The standard deviation of the neighbouring lines of the synthetic spectra."""


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


def _velocity(scene: soho.sumer.SpectrographObservation) -> na.AbstractScalar:
    """Return the velocity of the middle of each column of a scene."""
    return scene.inputs.velocity.cell_centers(scene.axis_wavelength)


def _speed(scene: soho.sumer.SpectrographObservation) -> u.Quantity:
    """Return the speed at the middle of each column of a scene."""
    velocity = _velocity(scene)
    return np.abs(velocity.ndarray_aligned((scene.axis_wavelength,)))


def _synthetic(
    fraction: float,
    velocities: u.Quantity,
    slope: float,
) -> tuple[soho.sumer.SpectrographObservation, na.AbstractScalar]:
    """
    Make spectra of a line, its neighbouring lines, and a sloping continuum.

    The spectra have the pixels and velocities of the test file.
    Return the observation, and the height of the line in each pixel.

    Parameters
    ----------
    fraction
        The height of the neighbouring lines, as a fraction of that of the
        line, which they have on average.
    velocities
        The centers of the neighbouring lines.
    slope
        The change of the continuum over 100 km/s, as a fraction of its level
        at the line.
    """
    observation = _observation()
    index = na.indices(observation.outputs.shape)
    x = index[observation.axis_detector_x]
    y = index[observation.axis_detector_y]
    velocity = _velocity(observation)

    height = 1000 * (1 + np.sin(x / 7) * np.cos(y / 5) / 2) * _unit
    radiance = height * np.exp(-np.square(velocity / _sigma_line) / 2)
    for i, center in enumerate(velocities):
        amplitude = fraction * height * (1 + np.cos(x / 3 + y / 4 + i) / 3)
        shape = np.exp(-np.square((velocity - center) / _sigma_neighbour) / 2)
        radiance = radiance + amplitude * shape
    continuum = 50 * _unit * (1 + slope * velocity / (100 * u.km / u.s))
    radiance = radiance + continuum

    return dataclasses.replace(observation, outputs=radiance), height


def test_scene_sumer_observation() -> None:
    observation = _observation()
    result = esis.data.synth.scene_sumer_observation(observation)
    assert isinstance(result, soho.sumer.SpectrographObservation)
    assert result.outputs.unit.is_equivalent(_unit)
    assert result.axis_wavelength == observation.axis_wavelength
    num_x = observation.outputs.shape[observation.axis_detector_x]
    assert result.outputs.shape[result.axis_detector_x] == num_x

    # The window of the test file reaches past the neighbouring line on the
    # blue side, which is weak, so the line is kept beyond it.
    # It does not reach Si II on the red side, so the line is cut short.
    velocity = _velocity(result)
    assert velocity.min() < -250 * u.km / u.s
    assert 80 * u.km / u.s < velocity.max() < 160 * u.km / u.s


def test_velocity_max() -> None:
    velocity_max = 125 * u.km / u.s
    observation = _observation()
    result = esis.data.synth.scene_sumer_observation(
        observation=observation,
        velocity_max=velocity_max,
    )
    assert np.all(_speed(result) <= velocity_max)
    assert np.any(_speed(observation) > velocity_max)


def test_unchanged() -> None:
    """Without a cut, continuum, or neighbouring lines, the scene is the observation."""
    observation = _observation()
    result = esis.data.synth.scene_sumer_observation(
        observation=observation,
        velocity_max=None,
        velocity_continuum=None,
        velocities_neighbour=None,
    )
    expected = _radiance(observation)
    assert np.allclose(_radiance(result), expected, equal_nan=True)


def test_continuum() -> None:
    """Without neighbouring lines, the continuum is the same in every column."""
    observation = _observation()
    with_continuum = esis.data.synth.scene_sumer_observation(
        observation=observation,
        velocities_neighbour=None,
    )
    without = esis.data.synth.scene_sumer_observation(
        observation=observation,
        velocity_continuum=None,
        velocities_neighbour=None,
    )
    level = _radiance(without) - _radiance(with_continuum)
    finite = np.all(np.isfinite(level), axis=~0)
    level = level[finite]
    assert level.size
    assert np.allclose(
        level, level[..., :1], rtol=1e-6, atol=1e-6 * np.abs(level).max()
    )
    assert np.median(level[..., 0]) > 0


def test_neighbours_subtracted() -> None:
    """
    Weak neighbouring lines and a sloping continuum are subtracted.

    The line is left alone, out to the edges of the window on both sides.
    """
    velocities = [-175, +130] * u.km / u.s
    observation, height = _synthetic(
        fraction=0.05,
        velocities=velocities,
        slope=0.2,
    )
    result = esis.data.synth.scene_sumer_observation(
        observation=observation,
        velocities_neighbour=velocities,
    )
    velocity = _velocity(result)
    assert velocity.min() < -250 * u.km / u.s
    assert velocity.max() > 160 * u.km / u.s

    expected = height * np.exp(-np.square(velocity / _sigma_line) / 2)
    atol = 1e-4 * height.max()
    assert np.all(np.abs(result.outputs - expected) < atol)


@pytest.mark.parametrize(
    argnames="fraction,height_neighbour_max",
    argvalues=[
        (0.5, 0.15),
        (0.05, 0.01),
    ],
)
def test_neighbours_strong(
    fraction: float,
    height_neighbour_max: float,
) -> None:
    """Neighbouring lines above the greatest height are cut off, not subtracted."""
    velocities = [-175, +130] * u.km / u.s
    observation, _ = _synthetic(
        fraction=fraction,
        velocities=velocities,
        slope=0.2,
    )
    result = esis.data.synth.scene_sumer_observation(
        observation=observation,
        velocities_neighbour=velocities,
        height_neighbour_max=height_neighbour_max,
    )
    # cut at the troughs, past 80 km/s and two standard deviations short of
    # the neighbouring lines
    velocity = _velocity(result)
    margin = 2 * _sigma_neighbour
    assert velocities[0] + margin <= velocity.min() <= -80 * u.km / u.s
    assert 80 * u.km / u.s <= velocity.max() <= velocities[1] - margin


def test_neighbours_same_side() -> None:
    with pytest.raises(ValueError, match="At most one"):
        esis.data.synth.scene_sumer_observation(
            observation=_observation(),
            velocities_neighbour=[150, 200] * u.km / u.s,
        )


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
