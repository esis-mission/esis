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

    def test_file_roundtrip(
        self,
        a: esis.optics.DistortionParameters,
        tmp_path: pathlib.Path,
    ):
        path = tmp_path / "parameters.ecsv"
        a.to_file(path, metadata=dict(description="round-trip test"))

        b = esis.optics.DistortionParameters.from_file(path)
        for field in dataclasses.fields(a):
            assert _allclose(getattr(b, field.name), getattr(a, field.name))

        # the axis of the parameters can be renamed on load
        c = esis.optics.DistortionParameters.from_file(path, axis="chan")
        assert set(na.shape(c)) <= {"chan"}


def _scene() -> na.FunctionArray:
    return na.FunctionArray(
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


def test_distortion_objective():
    instrument = esis.flights.f1.optics.design(
        grid=_grid_coarse,
        num_distribution=0,
    )[dict(channel=0)]
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)

    scene = _scene()

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


def test_correlation_axis():
    a = na.random.uniform(
        low=0,
        high=100,
        shape_random=dict(x=11, y=11),
    )
    scale = na.ScalarArray(np.array([1.0, 5.0]), axes="channel")

    # a linear per-channel rescaling should not affect
    # the per-channel correlation
    correlation = esis.optics._distortions._distortions._correlation(
        a=a * scale,
        b=a,
        axis=("x", "y"),
    )
    assert na.shape(correlation) == {"channel": 2}
    assert np.allclose(na.value(correlation).ndarray, 1)


def test_fit_distortion(tmp_path: pathlib.Path):
    instrument = esis.flights.f1.optics.design(
        grid=_grid_coarse,
        num_distribution=0,
    )[dict(channel=0)]
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)
    bounds = esis.flights.f1.optics.distortion_fit_bounds(parameters)

    scene = _scene()

    observation = instrument.system.image(
        scene=scene,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
        noise=False,
    ).outputs

    lower = na.pack(bounds[0]).ndarray
    upper = na.pack(bounds[1]).ndarray

    result = esis.optics.fit_distortion(
        instrument=instrument,
        scene=scene,
        observation=observation,
        bounds=bounds,
        parameters=parameters,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
        directory=tmp_path / "fit",
        kwargs_optimizer=dict(
            init=np.linspace(lower, upper, num=5),
            maxiter=1,
            tol=1e6,
            polish=False,
        ),
    )

    assert isinstance(result, esis.optics.DistortionParameters)

    x = na.pack(result).ndarray
    assert np.all(lower <= x)
    assert np.all(x <= upper)

    assert (tmp_path / "fit" / "full_output.log").exists()
    assert (tmp_path / "fit" / "convergence_data.csv").exists()


def test_distortion_residual():
    instrument = esis.flights.f1.optics.design(
        grid=_grid_coarse,
        num_distribution=0,
    )[dict(channel=0)]
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)

    scene = _scene()

    observation = instrument.system.image(
        scene=scene,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
        noise=False,
    ).outputs

    residual = esis.optics.DistortionResidual(
        instrument=instrument,
        parameters=parameters,
        scene=scene,
        observation=observation,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
        smoothing=3,
    )

    x = na.pack(parameters).ndarray
    result = residual(x)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert np.all(np.isfinite(result))

    # unlike DistortionObjective, the residual is deterministic for a fixed
    # parameter vector, which is what makes a derivative-based fit possible
    assert np.array_equal(result, residual(x))

    # the residual must survive pickling to support parallel optimization
    residual_pickled = pickle.loads(pickle.dumps(residual))
    assert isinstance(residual_pickled, esis.optics.DistortionResidual)


def test_kernel_gaussian():
    kernel = esis.optics._distortions._distortions._kernel_gaussian(
        sigma=1.0,
        axis=("x", "y"),
    )
    assert set(na.shape(kernel)) == {"x", "y"}
    assert np.isclose(kernel.ndarray.sum(), 1)
    assert np.all(kernel.ndarray >= 0)

    # the kernel support must widen with sigma
    kernel_wide = esis.optics._distortions._distortions._kernel_gaussian(
        sigma=3.0,
        axis=("x", "y"),
    )
    assert na.shape(kernel_wide)["x"] > na.shape(kernel)["x"]


def test_distortion_residual_sigma_psf():
    instrument = esis.flights.f1.optics.design(
        grid=_grid_coarse,
        num_distribution=0,
    )[dict(channel=0)]
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)

    scene = _scene()

    observation = instrument.system.image(
        scene=scene,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
        noise=False,
    ).outputs

    residual = esis.optics.DistortionResidual(
        instrument=instrument,
        parameters=parameters,
        scene=scene,
        observation=observation,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
        smoothing=None,
        sigma_psf=1.0,
    )

    x = na.pack(parameters).ndarray
    result = residual(x)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert np.all(np.isfinite(result))
    assert np.array_equal(result, residual(x))

    # the point-spread function must actually change the modeled image
    residual_sharp = dataclasses.replace(residual, sigma_psf=None)
    assert not np.array_equal(result, residual_sharp(x))

    # the residual must survive pickling to support parallel optimization
    residual_pickled = pickle.loads(pickle.dumps(residual))
    assert isinstance(residual_pickled, esis.optics.DistortionResidual)


def test_peak_parabola():
    peak_parabola = esis.optics._distortions._distortions._peak_parabola

    # the vertex of an exact parabola is recovered even between samples
    x = np.linspace(-2, 2, 5)
    y = -np.square(x - 0.3)
    assert np.isclose(peak_parabola(x, y), 0.3)

    # a peak at the edge of the scan does not extrapolate past the samples
    y = x
    assert peak_parabola(x, y) <= x.max()

    # a convex curve falls back to the best sample
    y = np.square(x - 0.3)
    assert peak_parabola(x, y) == x.min()


def test_fit_distortion_scan(tmp_path: pathlib.Path):
    instrument = esis.flights.f1.optics.design(
        grid=_grid_coarse,
        num_distribution=0,
    )[dict(channel=0)]
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)

    scene = _scene()

    observation = instrument.system.image(
        scene=scene,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
        noise=False,
    ).outputs

    grids = [
        dict(
            pitch=np.linspace(-20, 20, 5) * u.arcsec,
            yaw=np.linspace(-20, 20, 5) * u.arcsec,
        ),
        dict(
            pitch=np.linspace(-4, 4, 5) * u.arcsec,
        ),
    ]

    result = esis.optics.fit_distortion_scan(
        instrument=instrument,
        scene=scene,
        observation=observation,
        grids=grids,
        parameters=parameters,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
        sigma_psf=1.0,
        directory=tmp_path / "scan",
    )

    assert isinstance(result, esis.optics.DistortionParameters)

    # the fitted offsets must stay within the scanned ranges
    assert np.abs(result.pitch - parameters.pitch) <= 24 * u.arcsec
    assert np.abs(result.yaw - parameters.yaw) <= 20 * u.arcsec

    assert (tmp_path / "scan" / "scan.json").exists()
    assert (tmp_path / "scan" / "scan.log").exists()


def test_fit_distortion_scan_channel():
    instrument = esis.flights.f1.optics.design(
        grid=_grid_coarse,
        num_distribution=0,
    )
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)

    scene = _scene()

    observation = instrument.system.image(
        scene=scene,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
        noise=False,
    ).outputs

    result = esis.optics.fit_distortion_scan(
        instrument=instrument,
        scene=scene,
        observation=observation,
        grids=[dict(pitch=np.linspace(-20, 20, 5) * u.arcsec)],
        parameters=parameters,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
        axis_channel="channel",
        sigma_psf=1.0,
    )

    assert isinstance(result, esis.optics.DistortionParameters)

    # each channel's peak is read off its own curve, so the fitted pointing
    # gains a channel axis even though the start was a scalar
    num_channel = na.shape(observation)["channel"]
    assert na.shape(result.pitch) == dict(channel=num_channel)


def test_fit_distortion_series(tmp_path: pathlib.Path):
    instrument = esis.flights.f1.optics.design(
        grid=_grid_coarse,
        num_distribution=0,
    )[dict(channel=0)]
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)

    scene = _scene()
    observation = instrument.system.image(
        scene=scene,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
        noise=False,
    ).outputs

    grids = [dict(pitch=np.linspace(-10, 10, 5) * u.arcsec)]

    results = esis.optics.fit_distortion_series(
        instrument=instrument,
        scenes=[scene, scene],
        observations=[observation, observation],
        grids=grids,
        parameters=parameters,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
        sigma_psf=1.0,
        directory=tmp_path / "series",
    )

    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert isinstance(result, esis.optics.DistortionParameters)

    # identical frames fit from an identical start give identical results
    assert _allclose(results[0].pitch, results[1].pitch)

    assert (tmp_path / "series" / "frame_000" / "scan.json").exists()
    assert (tmp_path / "series" / "frame_001" / "scan.json").exists()


def test_fit_distortion_series_parallel():
    instrument = esis.flights.f1.optics.design(
        grid=_grid_coarse,
        num_distribution=0,
    )[dict(channel=0)]
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)

    scene = _scene()
    observation = instrument.system.image(
        scene=scene,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
        noise=False,
    ).outputs

    results = esis.optics.fit_distortion_series(
        instrument=instrument,
        scenes=[scene, scene],
        observations=[observation, observation],
        grids=[dict(pitch=np.linspace(-10, 10, 3) * u.arcsec)],
        parameters=parameters,
        axis_wavelength="wavelength",
        axis_field=("scene_x", "scene_y"),
        sigma_psf=1.0,
        workers=2,
    )

    assert len(results) == 2
    for result in results:
        assert isinstance(result, esis.optics.DistortionParameters)


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
