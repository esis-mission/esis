import pathlib
from typing import Any
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import zarr
import esis
from esis.data.synth._scene_vdem import _scene_vdem

logt = np.array([5.5, 5.6, 5.7], dtype=np.float32)
"""The logarithm of the temperature of each bin of the DEM."""

vbins = np.array([-15, -5, 5, 25], dtype=np.float64)
"""The vertices of the bins of line-of-sight velocity, in km/s."""

pitch = 192e5
"""The width of a pixel of the simulation, in centimeters."""


def _store(path: pathlib.Path, abundance: str = "Coronal_Chianti_2021") -> None:
    """
    Write a velocity-resolved DEM into a zarr store, in the format of MUSE.

    It is three pixels by two, in three bins of temperature and of velocity.
    """
    rng = np.random.default_rng(271828)
    vdem = rng.uniform(0, 1, size=(3, 3, 3, 2)).astype(np.float32)

    group = zarr.open_group(path, mode="w", zarr_format=2)

    def create(name: str, data: np.ndarray, dims: list[str], **attrs: str) -> None:
        """Write one array, with the names of its axes as xarray does."""
        array = group.create_array(name, data=data)
        array.attrs["_ARRAY_DIMENSIONS"] = dims
        array.attrs.update(attrs)

    create("vdem", vdem, ["logT", "vdop", "x", "y"], units="1e27 / cm5")
    create("logT", logt, ["logT"])
    create("vbins", vbins, ["vbins"], units="km/s")
    create("vdop", np.array([-10, 0, 10]), ["vdop"])
    create("x", pitch * np.arange(3), ["x"], units="cm")
    create("y", pitch * np.arange(2), ["y"], units="cm")
    create("Abundance", np.array(abundance), [])


def _scene(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> dict[str, Any]:
    """
    Call :func:`esis.data.synth.scene_vdem`, and return what it passes on.

    :func:`esis.data.synth.scene_emission_measure` is replaced, and what it is
    called with is returned.
    """
    calls = []

    def scene_emission_measure(**kw: Any) -> na.FunctionArray:
        """Stand in for the synthesis, recording its arguments."""
        calls.append(kw)
        return na.FunctionArray(None, None)

    monkeypatch.setattr(_scene_vdem, "scene_emission_measure", scene_emission_measure)

    arguments = dict(
        wavelength_min=460 * u.AA,
        wavelength_max=470 * u.AA,
        velocity=na.linspace(-50, 50, axis="v", num=11) * u.km / u.s,
        pressure=3e15 * u.K / u.cm**3,
    )
    esis.data.synth.scene_vdem(**(arguments | kwargs))

    (call,) = calls
    return call


def test_scene_vdem(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    """
    Check the DEM is read into an emission measure and bins of velocity.

    The emission measure is resolved in temperature and in velocity, with a
    Gaussian at the center of each bin of velocity as wide as plasma spread
    evenly across it.
    """
    path = tmp_path / "vdem.zarr"
    _store(path)

    call = _scene(monkeypatch, path=path, axis_x="sx", axis_y="sy")

    assert call["abundance"] == "sun_coronal_2021_chianti"
    assert call["pressure"] == 3e15 * u.K / u.cm**3
    assert call["wavelength_min"] == 460 * u.AA
    assert call["wavelength_max"] == 470 * u.AA

    axis_temperature = call["axis_temperature"]
    axis_shift = call["axis_shift"]

    temperature = call["temperature"]
    assert na.shape(temperature) == {axis_temperature: 3}
    assert np.allclose(temperature.ndarray, 10 ** logt.astype(float) * u.K)

    em = call["emission_measure"]
    assert em.axes == (axis_temperature, axis_shift, "sx", "sy")
    assert em.unit == u.Unit(1e27 * u.cm**-5)
    assert em.ndarray.dtype == np.float32
    assert np.all(em.ndarray.value == zarr.open_group(path)["vdem"][:])

    center = (vbins[1:] + vbins[:-1]) / 2
    width = np.diff(vbins)
    assert np.allclose(call["shift"].ndarray, center * u.km / u.s)
    assert np.allclose(
        call["width_nonthermal"].ndarray, width / np.sqrt(12) * u.km / u.s
    )
    assert na.shape(call["shift"]) == {axis_shift: 3}

    # the pixels, centered on the middle of the simulation, at the angle a
    # pixel spans from one astronomical unit away
    position = call["position"]
    scale = (pitch * u.cm / u.au).to(u.arcsec, u.dimensionless_angles())
    assert np.allclose(position.x.ndarray, np.array([-1.5, -0.5, 0.5, 1.5]) * scale)
    assert np.allclose(position.y.ndarray, np.array([-1, 0, 1]) * scale)
    assert na.shape(position) == dict(sx=4, sy=3)


def test_scene_vdem_plate_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
    """A plate scale stretches each pixel to the angle it gives."""
    path = tmp_path / "vdem.zarr"
    _store(path)

    call = _scene(monkeypatch, path=path, plate_scale=0.5 * u.arcsec)

    position = call["position"]
    assert np.allclose(position.x.ndarray, [-0.75, -0.25, 0.25, 0.75] * u.arcsec)
    assert np.allclose(position.y.ndarray, [-0.5, 0, 0.5] * u.arcsec)


def test_scene_vdem_abundance(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    """Check abundances fiasco isn't known to have are an error, unless given."""
    path = tmp_path / "vdem.zarr"
    _store(path, abundance="Photospheric_Somebody")

    with pytest.raises(ValueError, match="Photospheric_Somebody"):
        _scene(monkeypatch, path=path)

    call = _scene(monkeypatch, path=path, abundance="sun_photospheric_2021_asplund")
    assert call["abundance"] == "sun_photospheric_2021_asplund"
