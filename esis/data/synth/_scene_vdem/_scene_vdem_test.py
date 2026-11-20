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

scale = (pitch * u.cm / u.au).to(u.arcsec, u.dimensionless_angles())
"""The angle a pixel of the simulation spans from one astronomical unit away."""

units_default = dict(vdem="1e27 / cm5", vbins="km/s", x="cm", y="cm")
"""The unit of each array of the DEM, in its attribute ``units``."""


def _store(
    path: pathlib.Path,
    abundance: None | str = "Coronal_Chianti_2021",
    zarr_format: int = 2,
    x: None | np.ndarray = None,
    y: None | np.ndarray = None,
    units: None | dict[str, None | str] = None,
    dimensions: None | tuple[str, ...] = ("logT", "vdop", "x", "y"),
) -> None:
    """
    Write a velocity-resolved DEM into a zarr store, in the format of MUSE.

    It is three pixels by two, unless `x` and `y` say otherwise, in three
    bins of temperature and of velocity.

    Parameters
    ----------
    path
        Where to write the store.
    abundance
        The name of the abundances of the simulation, or :obj:`None` for a
        DEM which names none, as the quiet-Sun DEM of Rempel does.
    zarr_format
        The version of the format of the store. A v2 store names the axes of
        each array in an attribute, as xarray writes it, and a v3 store in
        the metadata of the array.
    x
        The position of each pixel in :math:`x`, in the unit of ``x``.
    y
        The position of each pixel in :math:`y`, in the unit of ``y``.
    units
        The unit of each array, by name, in place of those of
        :data:`units_default`. :obj:`None` leaves the array without a unit.
    dimensions
        The names of the axes of ``vdem``, or :obj:`None` to leave them
        unnamed.
    """
    if x is None:
        x = pitch * np.arange(3)
    if y is None:
        y = pitch * np.arange(2)
    units = units_default | ({} if units is None else units)

    rng = np.random.default_rng(271828)
    vdem = rng.uniform(0, 1, size=(3, 3, x.size, y.size)).astype(np.float32)

    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)

    def create(name: str, data: np.ndarray, dims: None | tuple[str, ...]) -> None:
        """Write one array, with the names of its axes and its unit."""
        if zarr_format == 2:
            array = group.create_array(name, data=data)
            if dims is not None:
                array.attrs["_ARRAY_DIMENSIONS"] = list(dims)
        else:
            array = group.create_array(name, data=data, dimension_names=dims)
        unit = units.get(name)
        if unit is not None:
            array.attrs["units"] = unit

    create("vdem", vdem, dimensions)
    create("logT", logt, ("logT",))
    create("vbins", vbins, ("vbins",))
    create("vdop", np.array([-10, 0, 10]), ("vdop",))
    create("x", x, ("x",))
    create("y", y, ("y",))
    if abundance is not None:
        create("Abundance", np.array(abundance), None if zarr_format == 3 else ())


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
        velocity=na.linspace(-50, 50, axis="velocity", num=11) * u.km / u.s,
        pressure=3e15 * u.K / u.cm**3,
    )
    esis.data.synth.scene_vdem(**(arguments | kwargs))

    (call,) = calls
    return call


@pytest.mark.parametrize(
    argnames="zarr_format",
    argvalues=[2, 3],
)
def test_scene_vdem(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    zarr_format: int,
) -> None:
    """
    Check the DEM is read into an emission measure and bins of velocity.

    The emission measure is resolved in temperature and in velocity, with a
    Gaussian at the center of each bin of velocity as wide as plasma spread
    evenly across it. The axes are named in the attributes of a zarr v2
    store, and in the metadata of a zarr v3 store.
    """
    path = tmp_path / "vdem.zarr"
    _store(path, zarr_format=zarr_format)

    call = _scene(monkeypatch, path=path, axis_x="sx", axis_y="sy")

    assert call["abundance"] == "sun_coronal_2021_chianti"
    assert call["pressure"] == 3e15 * u.K / u.cm**3
    assert call["wavelength_min"] == 460 * u.AA
    assert call["wavelength_max"] == 470 * u.AA
    assert call["fraction_min"] == 0.005
    assert call["ions"] is None
    assert call["axis_x"] == "sx"
    assert call["axis_y"] == "sy"
    assert call["axis_line"] == "line"
    assert call["axis_velocity"] == "velocity"

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
    assert np.allclose(position.x.ndarray, np.array([-1.5, -0.5, 0.5, 1.5]) * scale)
    assert np.allclose(position.y.ndarray, np.array([-1, 0, 1]) * scale)
    assert na.shape(position) == dict(sx=4, sy=3)


def test_scene_vdem_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Check the arguments the DEM has no say in are passed on as they are."""
    path = tmp_path / "vdem.zarr"
    _store(path)

    velocity = na.linspace(-50, 50, axis="v", num=11) * u.km / u.s

    call = _scene(
        monkeypatch,
        path=path,
        velocity=velocity,
        fraction_min=0.01,
        ions=["Ne 7"],
        axis_line="l",
        axis_velocity="v",
    )

    assert call["velocity"] is velocity
    assert call["fraction_min"] == 0.01
    assert call["ions"] == ["Ne 7"]
    assert call["axis_line"] == "l"
    assert call["axis_velocity"] == "v"
    assert call["axis_x"] == "x"
    assert call["axis_y"] == "y"


def test_scene_vdem_width_nonthermal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Check a nonthermal width is added in quadrature to that of each bin."""
    path = tmp_path / "vdem.zarr"
    _store(path)

    call = _scene(monkeypatch, path=path, width_nonthermal=10 * u.km / u.s)

    width = np.sqrt(np.square(np.diff(vbins) / np.sqrt(12)) + 10**2) * u.km / u.s
    assert np.allclose(call["width_nonthermal"].ndarray, width)


def test_scene_vdem_plate_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A plate scale stretches each pixel to the angle it gives."""
    path = tmp_path / "vdem.zarr"
    _store(path)

    call = _scene(monkeypatch, path=path, plate_scale=0.5 * u.arcsec)

    position = call["position"]
    assert np.allclose(position.x.ndarray, [-0.75, -0.25, 0.25, 0.75] * u.arcsec)
    assert np.allclose(position.y.ndarray, [-0.5, 0, 0.5] * u.arcsec)

    # one along each axis
    plate_scale = na.Cartesian2dVectorArray(0.5, 2) * u.arcsec
    call = _scene(monkeypatch, path=path, plate_scale=plate_scale)

    position = call["position"]
    assert np.allclose(position.x.ndarray, [-0.75, -0.25, 0.25, 0.75] * u.arcsec)
    assert np.allclose(position.y.ndarray, [-2, 0, 2] * u.arcsec)


@pytest.mark.parametrize(
    argnames="x,y,units_pixel,expected_x,expected_y",
    argvalues=[
        (None, None, dict(), [-1.5, -0.5, 0.5, 1.5], [-1, 0, 1]),
        # positions which decrease along each axis
        (
            pitch * np.arange(3)[::-1],
            pitch * np.arange(2)[::-1],
            dict(),
            [-1.5, -0.5, 0.5, 1.5],
            [-1, 0, 1],
        ),
        # pixels twice as tall as they are wide
        (None, 2 * pitch * np.arange(2), dict(), [-1.5, -0.5, 0.5, 1.5], [-2, 0, 2]),
        # positions in kilometers
        (
            pitch / 1e5 * np.arange(3),
            pitch / 1e5 * np.arange(2),
            dict(x="km", y="km"),
            [-1.5, -0.5, 0.5, 1.5],
            [-1, 0, 1],
        ),
        # positions without a unit, which are in centimeters
        (None, None, dict(x=None, y=None), [-1.5, -0.5, 0.5, 1.5], [-1, 0, 1]),
        # a simulation one pixel wide along x, or along y
        (np.array([5.0]), None, dict(), [-0.5, 0.5], [-1, 0, 1]),
        (None, np.array([5.0]), dict(), [-1.5, -0.5, 0.5, 1.5], [-0.5, 0.5]),
    ],
    ids=["square", "decreasing", "rectangular", "km", "unitless", "column", "row"],
)
def test_scene_vdem_pitch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    x: None | np.ndarray,
    y: None | np.ndarray,
    units_pixel: dict[str, None | str],
    expected_x: list[float],
    expected_y: list[float],
) -> None:
    """
    Check each axis of the pixels has its own pitch, in its own unit.

    The positions of the result increase along each axis whichever way those
    of the store run, and a simulation one pixel wide along an axis has the
    pitch of the other axis along it.
    """
    path = tmp_path / "vdem.zarr"
    _store(path, x=x, y=y, units=units_pixel)

    call = _scene(monkeypatch, path=path)

    position = call["position"]
    assert np.allclose(position.x.ndarray, np.array(expected_x) * scale)
    assert np.allclose(position.y.ndarray, np.array(expected_y) * scale)


def test_scene_vdem_pixel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Check a single pixel needs a plate scale, which has nothing to go on."""
    path = tmp_path / "vdem.zarr"
    _store(path, x=np.array([0.0]), y=np.array([0.0]))

    with pytest.raises(ValueError, match="plate_scale"):
        _scene(monkeypatch, path=path)

    call = _scene(monkeypatch, path=path, plate_scale=0.5 * u.arcsec)
    position = call["position"]
    assert np.allclose(position.x.ndarray, [-0.25, 0.25] * u.arcsec)
    assert np.allclose(position.y.ndarray, [-0.25, 0.25] * u.arcsec)


def test_scene_vdem_units(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """
    Check the bins of velocity are read in their own unit, or in km/s.

    They are in kilometers per second if they have no unit, but the emission
    measure has to have one.
    """
    center = (vbins[1:] + vbins[:-1]) / 2 * u.km / u.s
    width = np.diff(vbins) / np.sqrt(12) * u.km / u.s

    path = tmp_path / "m.zarr"
    _store(path, units=dict(vbins="m/s"))
    call = _scene(monkeypatch, path=path)
    assert np.allclose(call["shift"].ndarray, center / 1000)
    assert np.allclose(call["width_nonthermal"].ndarray, width / 1000)

    path = tmp_path / "unitless.zarr"
    _store(path, units=dict(vbins=None))
    call = _scene(monkeypatch, path=path)
    assert np.allclose(call["shift"].ndarray, center)
    assert np.allclose(call["width_nonthermal"].ndarray, width)

    path = tmp_path / "unknown.zarr"
    _store(path, units=dict(vdem=None))
    with pytest.raises(ValueError, match="vdem"):
        _scene(monkeypatch, path=path)


@pytest.mark.parametrize(
    argnames="zarr_format",
    argvalues=[2, 3],
)
@pytest.mark.parametrize(
    argnames="dimensions",
    argvalues=[None, ("logT", "vdop", "x", "z")],
    ids=["unnamed", "unknown"],
)
def test_scene_vdem_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    zarr_format: int,
    dimensions: None | tuple[str, ...],
) -> None:
    """Check the axes of the DEM have to be named, and named as expected."""
    path = tmp_path / "vdem.zarr"
    _store(path, zarr_format=zarr_format, dimensions=dimensions)

    with pytest.raises(ValueError, match="axes"):
        _scene(monkeypatch, path=path)


def test_scene_vdem_abundance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Check abundances fiasco isn't known to have are an error, unless given."""
    path = tmp_path / "vdem.zarr"
    _store(path, abundance="Photospheric_Somebody")

    with pytest.raises(ValueError, match="Photospheric_Somebody"):
        _scene(monkeypatch, path=path)

    call = _scene(monkeypatch, path=path, abundance="sun_photospheric_2021_asplund")
    assert call["abundance"] == "sun_photospheric_2021_asplund"


def test_scene_vdem_abundance_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """
    Check a DEM which names no abundances needs them to be given.

    The quiet-Sun DEM of Rempel has no array ``Abundance``.
    """
    path = tmp_path / "vdem.zarr"
    _store(path, abundance=None)

    with pytest.raises(ValueError, match="`abundance`"):
        _scene(monkeypatch, path=path)

    call = _scene(monkeypatch, path=path, abundance="sun_coronal_2021_chianti")
    assert call["abundance"] == "sun_coronal_2021_chianti"
