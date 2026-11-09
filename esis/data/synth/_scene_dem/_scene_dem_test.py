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
from esis.data.synth._scene_dem._scene_dem import _lines


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


def _dem(
    temperature: na.AbstractScalar,
    amplitude: na.AbstractScalar,
) -> na.AbstractScalar:
    """Return a Gaussian DEM per unit log T, at half a million kelvin."""
    logt = np.log10(temperature.to(u.K).value)
    return amplitude * np.exp(-np.square((logt - 5.7) / 0.15) / 2) / u.cm**5


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
    em = utu.dem.emission_measure(dem, axis_temperature)
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


@needs_database
def test_lines_ions(monkeypatch: pytest.MonkeyPatch):
    """
    Check the ions default to those with a line in the passband.

    :func:`utu.spectrum.ions` is replaced by a list of one ion, so that only
    that ion is computed, and the cache is bypassed, so that the search runs.
    """
    calls = []

    def ions(**kwargs: Any) -> list[str]:
        """Stand in for the search, finding one ion."""
        calls.append(kwargs)
        return ["Ne 7"]

    monkeypatch.setattr(utu.spectrum, "ions", ions)

    wavelength_min = 465 * u.AA
    wavelength_max = 466 * u.AA

    contribution, mass = _lines.func(
        temperature=np.geomspace(1e5, 1e6, num=6) * u.K,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        ions=None,
        abundance=esis.data.synth._scene_dem.abundance,
        pressure=esis.data.synth._scene_dem.pressure,
        versions=dict(),
        axis_temperature="t",
        axis_line="l",
    )

    (call,) = calls
    assert call["wavelength_min"] == wavelength_min
    assert call["wavelength_max"] == wavelength_max
    assert call["abundance"] == esis.data.synth._scene_dem.abundance

    rest = contribution.inputs
    assert np.all((rest > wavelength_min) & (rest < wavelength_max))
    assert na.shape(contribution.outputs) == dict(t=6, l=rest.size)
    assert np.all(mass == fiasco.Ion("Ne 7", 1e6 * u.K).mass)


def test_scene_emission_measure_shift(monkeypatch: pytest.MonkeyPatch):
    """
    Check a velocity-resolved emission measure adds up its bins.

    An emission measure resolved in line-of-sight velocity is the plasma in
    each of its bins of velocity, each at its own shift and width, added up.

    The contribution functions are made up, so that the test runs without
    the CHIANTI database.
    """
    temperature = na.geomspace(1e5, 10**6.5, axis="t", num=6) * u.K
    contribution = na.FunctionArray(
        inputs=na.ScalarArray([465.221, 499.406] * u.AA, axes="l"),
        outputs=na.ScalarArray(
            ndarray=np.array([[1, 3, 2, 1, 0.5, 0.1], [0.1, 0.2, 1, 4, 3, 1]])
            * 1e-24
            * u.erg
            * u.cm**3
            / u.s,
            axes=("l", "t"),
        ),
    )
    mass = na.ScalarArray([20.18, 28.09] * u.u, axes="l")

    def lines(**kwargs: Any) -> tuple[na.FunctionArray, na.ScalarArray]:
        """Stand in for the contribution functions of the passband."""
        return contribution, mass

    monkeypatch.setattr(esis.data.synth._scene_dem._scene_dem, "_lines", lines)

    em = na.ScalarArray(np.array([1, 2, 4, 2, 1, 0.5]) * 1e26 / u.cm**5, axes="t")
    em = em * na.ScalarArray(np.array([0.2, 0.5, 0.3]), axes="s")
    em = em * na.ScalarArray(np.array([1, 2]), axes="x")
    shift = na.ScalarArray([-40, 0, 25] * u.km / u.s, axes="s")
    width = na.ScalarArray([10, 5, 15] * u.km / u.s, axes="s")
    position = na.Cartesian2dVectorArray(
        x=na.ScalarArray([-1, 0, 1] * u.arcsec, axes="x"),
        y=0 * u.arcsec,
    )

    kwargs = dict(
        temperature=temperature,
        position=position,
        wavelength_min=460 * u.AA,
        wavelength_max=530 * u.AA,
        velocity=na.linspace(-300, 300, axis="v", num=61) * u.km / u.s,
        abundance="sun_coronal_2021_chianti",
        pressure=3e15 * u.K / u.cm**3,
        axis_temperature="t",
        axis_line="l",
        axis_velocity="v",
    )

    result = esis.data.synth.scene_emission_measure(
        emission_measure=em,
        shift=shift,
        width_nonthermal=width,
        axis_shift="s",
        **kwargs,
    )

    assert isinstance(result.inputs, na.DopplerPositionalVectorArray)
    assert result.inputs.position is position
    assert na.shape(result.outputs) == dict(l=2, v=60, x=2)

    expected = 0
    for i in range(3):
        index = {"s": i}
        part = esis.data.synth.scene_emission_measure(
            emission_measure=em[index],
            shift=shift[index],
            width_nonthermal=width[index],
            **kwargs,
        )
        expected = expected + part.outputs

    assert np.allclose(result.outputs, expected, rtol=1e-12)
