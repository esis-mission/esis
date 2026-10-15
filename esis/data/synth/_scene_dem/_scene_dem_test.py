from typing import Any
import pytest
import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na
import fiasco
import sdo
import utu
import esis
from esis.data.synth._scene_dem._scene_dem import _emission_measure


def _database() -> bool:
    """Whether the CHIANTI database has been downloaded on this machine."""
    try:
        return len(fiasco.list_ions()) > 0
    except Exception:  # pragma: nocover
        return False


needs_database = pytest.mark.skipif(
    not _database(),
    reason="the CHIANTI database is not available",
)

channels = na.ScalarArray([94, 131, 171, 193, 211, 335] * u.AA, axes="wavelength")


def _response() -> na.FunctionArray:
    """Return the response the DEM is inverted against, from log T 5.5 to 7.5."""
    response = sdo.aia.temperature_response(
        wavelength=channels,
        emissivity="CHIANTI 11",
    )
    return response[{"temperature": slice(30, 71)}]


def _dem(
    temperature: na.AbstractScalar,
    amplitude: na.AbstractScalar,
) -> na.AbstractScalar:
    """Return a Gaussian DEM per unit log T, at half a million kelvin."""
    logt = np.log10(temperature.to(u.K).value)
    return amplitude * np.exp(-np.square((logt - 5.7) / 0.15) / 2) / u.cm**5


def test_emission_measure_constant():
    """A constant DEM holds its value times the span of log T."""
    response = _response()
    temperature = response.inputs
    dem = na.FunctionArray(
        inputs=temperature,
        outputs=na.ScalarArray(
            ndarray=np.full(temperature.shape["temperature"], 3e27) / u.cm**5,
            axes="temperature",
        ),
    )

    result = _emission_measure(dem, "temperature")

    logt = np.log10(temperature.to(u.K).value)
    span = logt[{"temperature": -1}] - logt[{"temperature": 0}]

    assert np.allclose(result.sum("temperature"), 3e27 / u.cm**5 * span)

    # the ends stand for half an interval, and the rest for a whole one
    step = 0.05
    assert np.allclose(result[{"temperature": 0}], 3e27 / u.cm**5 * step / 2)
    assert np.allclose(result[{"temperature": 5}], 3e27 / u.cm**5 * step)


def test_emission_measure_plowman():
    """
    Check the emission measures predict the intensities the inversion fit.

    Their reduced chi squared is the one it reports.
    """
    response = _response()

    amplitude = na.ScalarArray(np.array([1e27, 3e27, 1e28]), axes="x")
    em = _emission_measure(
        na.FunctionArray(response.inputs, _dem(response.inputs, amplitude)),
        axis="temperature",
    )
    intensity = (response.outputs * em).sum("temperature")

    # noise, so that the fit stops short of the data
    rng = np.random.default_rng(0)
    uncertainty = 0.05 * intensity
    noise = na.ScalarArray(rng.normal(size=(6, 3)), axes=("wavelength", "x"))
    intensity = intensity + noise * uncertainty

    dem, chi2 = utu.dem.plowman(
        intensity=intensity,
        uncertainty=uncertainty,
        response=response,
        axis_channel="wavelength",
        axis_temperature="temperature",
    )

    model = (response.outputs * _emission_measure(dem, "temperature")).sum(
        "temperature"
    )
    expected = np.square((intensity - model) / uncertainty).mean("wavelength")

    assert np.all(chi2 >= 0)
    assert np.allclose(chi2, expected, rtol=1e-6)


def _filtergram() -> sdo.aia.Filtergram:
    """
    AIA images three pixels by two, in each of the six channels.

    The counts are those of a uniform plasma. They only need to be positive,
    since the tests which use these replace the inversion.
    """
    time = astropy.time.Time("2016-07-15T00:00:00") + np.arange(6) * u.s
    position = na.Cartesian2dVectorLinearSpace(
        start=-0.9 * u.arcsec,
        stop=0.9 * u.arcsec,
        axis=na.Cartesian2dVectorArray("x", "y"),
        num=na.Cartesian2dVectorArray(4, 3),
    ).explicit
    return sdo.aia.Filtergram(
        inputs=na.TemporalSpectralPositionalVectorArray(
            time=na.ScalarArray(time, axes="wavelength"),
            wavelength=channels,
            position=position,
        ),
        outputs=na.ScalarArray(np.full((6, 3, 2), 1000.0), ("wavelength", "x", "y"))
        * u.DN,
        axis_detector_x="x",
        axis_detector_y="y",
        timedelta=2 * u.s,
    )


@needs_database
def test_scene_dem(monkeypatch: pytest.MonkeyPatch):
    """
    Check the lines are those of the plasma the DEM describes.

    The rest of the scene is that of the images.
    The inversion is replaced by a known DEM, so that the radiance of every
    line is known exactly. One pixel fails, and has to emit nothing.
    """
    calls = []

    def plowman(**kwargs: Any) -> tuple[na.FunctionArray, na.ScalarArray]:
        """Stand in for the inversion, returning a known DEM."""
        calls.append(kwargs)
        temperature = kwargs["response"].inputs
        amplitude = na.ScalarArray(np.array([[1, 2], [3, 4], [5, 6]]), ("x", "y"))
        amplitude = 1e27 * amplitude
        dem = na.FunctionArray(temperature, _dem(temperature, amplitude))
        chi2 = na.ScalarArray(np.array([[1, 1], [1, -1], [1, 1]]), ("x", "y"))
        return dem, chi2

    monkeypatch.setattr(utu.dem, "plowman", plowman)

    filtergram = _filtergram()
    wavelength_min = 460 * u.AA
    wavelength_max = 470 * u.AA
    velocity = na.linspace(-200, 200, axis="v", num=41) * u.km / u.s

    result = esis.data.synth.scene_dem(
        filtergram=filtergram,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        velocity=velocity,
        ions=["Ne 7"],
        axis_line="l",
        axis_velocity="v",
    )

    (call,) = calls
    assert call["axis_channel"] == "wavelength"
    temperature = call["response"].inputs
    logt = np.log10(temperature.to(u.K).value)
    assert np.isclose(logt.min().ndarray, 5.5)
    assert np.isclose(logt.max().ndarray, 7.5)

    assert isinstance(result.inputs, na.TemporalDopplerPositionalVectorArray)
    assert np.all(result.inputs.position == filtergram.inputs.position)
    assert result.inputs.time.ndarray == filtergram.inputs.time.ndarray.min()

    rest = result.inputs.wavelength_rest
    assert np.all((rest > wavelength_min) & (rest < wavelength_max))
    assert np.any(np.abs(rest - 465.221 * u.AA) < 0.001 * u.AA)
    assert na.shape(result.outputs) == dict(l=rest.size, v=40, x=3, y=2)

    # the radiance of each line, from the contribution functions of the
    # plasma and the emission measure of the known DEM
    (axis_temperature,) = na.shape(temperature)
    ion = fiasco.Ion(
        "Ne 7", temperature.ndarray, abundance=esis.data.synth._scene_dem.abundance
    )
    g = utu.spectrum.contribution_function(
        ion=ion,
        density=esis.data.synth._scene_dem.pressure / temperature,
        axis_temperature=axis_temperature,
        axis="l",
    )
    dem, chi2 = plowman(response=call["response"])
    em = _emission_measure(dem, axis_temperature)
    em = np.where(chi2 >= 0, em, 0 * em)

    radiance = (result.outputs * np.diff(result.inputs.wavelength, axis="v")).sum("v")
    for i in range(rest.size):
        w = rest[{"l": i}]
        j = np.argmin(np.abs(g.inputs - w), axis="l")
        expected = (g.outputs[j] * em).sum(axis_temperature) / (4 * np.pi * u.sr)
        assert np.allclose(radiance[{"l": i}], expected, rtol=1e-6)

    # the failed pixel emits nothing
    assert np.all(result.outputs[{"x": 1, "y": 1}] == 0)

    # and every line kept carries at least half a percent of the passband,
    # as the default asks
    total = (g.outputs * em.mean(("x", "y"))).sum(axis_temperature)
    total = total[(g.inputs > wavelength_min) & (g.inputs < wavelength_max)]
    fraction = radiance.mean(("x", "y")) * 4 * np.pi * u.sr / total.sum("l")
    assert np.all(fraction > 0.005)
    assert rest.size < total.size
