import dataclasses
import pathlib
import pickle
import types
import numpy as np
import pytest
import astropy.units as u
import named_arrays as na
import optika
import optika._tests.test_mixins
import esis

_instrument = esis.flights.f1.optics.design(num_distribution=0)

_grid_coarse = optika.vectors.ObjectVectorArray(
    wavelength=629.77 * u.AA,
    field=na.Cartesian2dVectorLinearSpace(
        start=-1,
        stop=1,
        axis=na.Cartesian2dVectorArray("field_x", "field_y"),
        num=5,
        centers=True,
    ),
    pupil=na.Cartesian2dVectorLinearSpace(
        start=-1,
        stop=1,
        axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
        num=5,
        centers=True,
    ),
)


def _allclose(x, y) -> bool:
    difference = x - y
    if na.unit(difference) is not None:
        difference = difference.to(na.unit(y))
    return bool(np.all(np.abs(na.value(difference)) < 1e-6))


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.DistortionParameters.from_instrument(_instrument),
        esis.optics.DistortionParameters.from_instrument(
            esis.flights.f1.optics.distortion_fit(num_distribution=0),
        ),
    ],
)
class TestDistortionParameters(
    optika._tests.test_mixins.AbstractTestPrintable,
):

    def test_pack_roundtrip(self, a: esis.optics.DistortionParameters):
        x = na.pack(a)
        b = na.unpack(x, a)
        for field in dataclasses.fields(a):
            assert _allclose(getattr(b, field.name), getattr(a, field.name))

    def test_to_instrument(self, a: esis.optics.DistortionParameters):
        result = a.to_instrument(_instrument)

        assert result is not _instrument
        assert isinstance(result, esis.optics.abc.AbstractInstrument)
        assert _allclose(result.grating.yaw, a.yaw_grating)
        assert _allclose(result.field_stop.roll, a.roll_field_stop)
        assert _allclose(
            result.primary_mirror.translation.z,
            -a.displacement_primary,
        )

        invariant = (
            result.primary_mirror.sag.focal_length + result.primary_mirror.translation.z
        )
        invariant_expected = (
            _instrument.primary_mirror.sag.focal_length
            + _instrument.primary_mirror.translation.z
        )
        assert _allclose(invariant, invariant_expected)

        assert _allclose(_instrument.grating.yaw, _instrument.grating.yaw.copy())

    def test_from_instrument_roundtrip(self, a: esis.optics.DistortionParameters):
        instrument = a.to_instrument(_instrument)
        b = esis.optics.DistortionParameters.from_instrument(instrument)
        for field in dataclasses.fields(a):
            assert _allclose(getattr(b, field.name), getattr(a, field.name))


def test_distortion_objective():
    instrument = esis.flights.f1.optics.design(
        grid=_grid_coarse,
        num_distribution=0,
    )[dict(channel=0)]
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)

    scene = na.FunctionArray(
        inputs=na.SpectralPositionalVectorArray(
            wavelength=na.linspace(
                start=629 * u.AA,
                stop=631 * u.AA,
                axis="wavelength",
                num=3,
            ),
            position=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("scene_x", "scene_y"),
                num=6,
            ),
        ),
        outputs=na.random.uniform(
            low=0 * u.photon / u.cm**2 / u.arcsec**2 / u.s / u.nm,
            high=100 * u.photon / u.cm**2 / u.arcsec**2 / u.s / u.nm,
            shape_random=dict(scene_x=5, scene_y=5),
        ),
    )

    observation = instrument.system.image(
        scene=scene,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
        noise=False,
    ).outputs

    objective = esis.optics.DistortionObjective(
        instrument=instrument,
        parameters=parameters,
        scene=scene,
        observation=observation,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
    )

    x = na.pack(parameters).ndarray
    result = objective(x)

    assert isinstance(result, float)
    assert np.isfinite(result)

    # the objective must survive pickling to support parallel optimization
    objective_pickled = pickle.loads(pickle.dumps(objective))
    assert isinstance(objective_pickled, esis.optics.DistortionObjective)


def test_correlation():
    a = na.random.uniform(
        low=0,
        high=100,
        shape_random=dict(x=11, y=11),
    )
    correlation = esis.optics._distortions._distortions._correlation(a, a)
    assert np.isclose(na.value(correlation).ndarray, 1)

    constant = na.ScalarArray(np.ones((11, 11)), axes=("x", "y"))
    correlation = esis.optics._distortions._distortions._correlation(a, constant)
    assert np.isfinite(na.value(correlation).ndarray)


def test_convergence_logger(tmp_path: pathlib.Path):
    logger = esis.optics.ConvergenceLogger(directory=tmp_path / "run")

    intermediate_result = types.SimpleNamespace(
        x=np.array([1.0, 2.0]),
        fun=-900.0,
        population_energies=np.array([-900.0, -800.0]),
    )
    logger(intermediate_result)
    logger(intermediate_result)

    assert logger.iteration == 2
    assert logger.path_data.exists()
    assert len(logger.path_data.read_text().splitlines()) == 3
    assert logger.path_log.exists()
    assert logger.path_plot.exists()
