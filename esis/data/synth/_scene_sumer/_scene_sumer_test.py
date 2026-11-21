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


def _reach(scene: soho.sumer.SpectrographObservation) -> tuple[u.Quantity, u.Quantity]:
    """Return the speed of the outermost columns of a scene, blue and red."""
    velocity = _velocity(scene)
    return -velocity.min().ndarray, velocity.max().ndarray


def _assert_symmetric(scene: soho.sumer.SpectrographObservation) -> None:
    """Check that a scene reaches as far on both sides, to within a column."""
    blue, red = _reach(scene)
    width = np.abs(np.diff(_velocity(scene).ndarray)).max()
    assert abs(blue - red) <= width


def test_scene_sumer_observation() -> None:
    observation = _observation()
    result = esis.data.synth.scene_sumer_observation(observation)
    assert isinstance(result, soho.sumer.SpectrographObservation)
    assert result.outputs.unit.is_equivalent(_unit)
    assert result.axis_wavelength == observation.axis_wavelength
    num_x = observation.outputs.shape[observation.axis_detector_x]
    assert result.outputs.shape[result.axis_detector_x] == num_x

    # The window of the test file does not reach Si II on the red side,
    # so the line is cut short there, and as far on the blue side.
    _assert_symmetric(result)
    for reach in _reach(result):
        assert 80 * u.km / u.s < reach < 160 * u.km / u.s


def test_velocity_max() -> None:
    velocity_max = 100 * u.km / u.s
    observation = _observation()
    result = esis.data.synth.scene_sumer_observation(
        observation=observation,
        velocity_max=velocity_max,
    )
    assert np.all(_speed(result) <= velocity_max)
    assert np.any(_speed(observation) > velocity_max)


def test_unprocessed() -> None:
    """
    Without a continuum or neighbouring lines, the scene is the observation.

    It is only cut, as far on both sides,
    to leave out the partly lit columns at the ends of the window.
    """
    observation = _observation()
    result = esis.data.synth.scene_sumer_observation(
        observation=observation,
        num_continuum=None,
        velocities_neighbour=None,
    )
    _assert_symmetric(result)
    v_observation = _velocity(observation).ndarray.value
    v_result = _velocity(result).ndarray.value
    kept = np.isin(np.round(v_observation, 6), np.round(v_result, 6))
    assert 0 < kept.sum() < kept.size - 4
    expected = _radiance(observation)[..., kept]
    assert np.allclose(_radiance(result), expected, equal_nan=True)


def test_continuum() -> None:
    """The continuum is one level for each pixel, the same in every column."""
    observation = _observation()
    with_continuum = esis.data.synth.scene_sumer_observation(observation)
    without = esis.data.synth.scene_sumer_observation(
        observation=observation,
        num_continuum=None,
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
    Weak neighbouring lines and the continuum are subtracted.

    The line is left alone, as far on both sides as the shorter one reaches,
    past the neighbouring line on that side.
    """
    velocities = [-175, +130] * u.km / u.s
    observation, height = _synthetic(
        fraction=0.05,
        velocities=velocities,
        slope=0,
    )
    result = esis.data.synth.scene_sumer_observation(
        observation=observation,
        velocities_neighbour=velocities,
    )
    _assert_symmetric(result)
    for reach in _reach(result):
        assert reach > 160 * u.km / u.s

    velocity = _velocity(result)
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
    # cut at the trough past 80 km/s and two standard deviations short of
    # the nearer neighbouring line, on both sides
    _assert_symmetric(result)
    for reach in _reach(result):
        assert 80 * u.km / u.s <= reach <= velocities[1] - 2 * _sigma_neighbour


def test_neighbour_uncovered() -> None:
    """
    A neighbouring line which the window does not reach well past is cut off.

    The window of the test file ends at about 187 km/s,
    one standard deviation past this neighbouring line, but not two.
    """
    velocities = [-175, +170] * u.km / u.s
    observation, _ = _synthetic(
        fraction=0.05,
        velocities=velocities,
        slope=0.2,
    )
    result = esis.data.synth.scene_sumer_observation(
        observation=observation,
        velocities_neighbour=velocities,
    )
    _assert_symmetric(result)
    for reach in _reach(result):
        assert 80 * u.km / u.s <= reach <= velocities[1] - 2 * _sigma_neighbour


def test_neighbour_residual() -> None:
    """
    A line left past a subtracted neighbouring line is cut off.

    The scene ends at the lowest point of the mean profile before it,
    on both sides.
    """
    velocity_extra = -165 * u.km / u.s
    velocities = [-90, +130] * u.km / u.s
    observation, _ = _synthetic(
        fraction=0.05,
        velocities=np.concatenate([[velocity_extra], velocities]),
        slope=0.2,
    )
    result = esis.data.synth.scene_sumer_observation(
        observation=observation,
        velocities_neighbour=velocities,
    )
    _assert_symmetric(result)
    for reach in _reach(result):
        assert -velocities[0] < reach < -velocity_extra


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
