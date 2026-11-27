import typing
import pytest
import numpy as np
import astropy.units as u
import astropy.time
import astropy.table
import named_arrays as na
import soho
import esis
from esis.flights.f1.data._sumer import _sumer
from esis.flights.f1.data.synth._scene_sumer import _scene_sumer
from esis.flights.f1.spectrum import O_V

_time_test = "2005-04-15T15:43"
"""The start of the pass of the smallest file of the release, 4 MB."""

_unit = u.erg / (u.cm**2 * u.sr * u.nm * u.s)
"""The unit of the radiance of the scenes."""


def test_passes_sumer() -> None:
    result = esis.flights.f1.data.synth.passes_sumer()
    assert isinstance(result, astropy.table.QTable)
    assert len(result) == 161
    assert result["used"].dtype == bool
    assert np.count_nonzero(result["used"]) == 147
    # the sequence which holds the slit still
    start = result["time_start"].isot
    (still,) = result[np.char.startswith(start.astype(str), "1997-04-23T13:44")]
    assert not still["used"]
    start = result["time_start"]
    assert np.all(start[1:] > start[:-1])
    assert np.all(result["time_start"] < result["time_stop"])
    assert set(result["file"]) <= {row["file"] for row in _sumer._manifest()}
    assert np.all(result["exposure"] > 0 * u.s)
    assert np.all(result["num_exposure"] >= 10)
    assert set(result["kind"]) == {"qs", "ar"}
    assert set(result["split"]) == {"train", "validation", "test"}
    assert result["polar"].dtype == bool
    assert result["tilt"].dtype == bool
    assert np.count_nonzero(result["tilt"]) == 47
    # never the passes of active regions or the scans of polar coronal holes
    assert not np.any(result["tilt"] & ((result["kind"] == "ar") | result["polar"]))
    tilt = {row["time_start"].isot[:16]: bool(row["tilt"]) for row in result}
    # the coronal holes and filaments, two of which reach past 900 arcseconds
    # but show no limb
    marked = [
        "2004-11-30T13:27",
        "2007-11-04T10:41",
        "2007-11-09T03:56",
        "2007-11-10T16:41",
    ]
    assert all(tilt[time] for time in marked)
    # the jet and the studies of active regions, whose flows it would remove,
    # and passes near the limb
    kept = [
        "1996-10-08T06:46",
        "2005-05-17T08:57",
        "2005-05-19T08:39",
        "1996-04-16T13:06",
        "2005-10-10T15:51",
        "2009-04-24T04:55",
        "2009-04-24T06:58",
    ]
    assert not any(tilt[time] for time in kept)


class TestPass:

    @pytest.mark.parametrize("offset", [-50, 0, 50] * u.s)
    def test_start(self, offset: u.Quantity) -> None:
        """A pass is named by its start, given to the minute."""
        (row,) = [
            r
            for r in esis.flights.f1.data.synth.passes_sumer()
            if r["time_start"].isot.startswith(_time_test)
        ]
        result = _scene_sumer._pass(row["time_start"] + offset)
        assert result["time_start"] == row["time_start"]

    def test_during(self) -> None:
        """A time during a pass names it."""
        table = esis.flights.f1.data.synth.passes_sumer()
        row = table[len(table) // 2]
        middle = row["time_start"] + (row["time_stop"] - row["time_start"]) / 2
        result = _scene_sumer._pass(middle)
        assert result["time_start"] == row["time_start"]

    def test_none(self) -> None:
        with pytest.raises(ValueError, match="No pass"):
            _scene_sumer._pass(astropy.time.Time("2019-09-30T18:00"))


@pytest.mark.parametrize("remove_stripes", [None, True])
@pytest.mark.parametrize(
    argnames="velocity_max",
    argvalues=[
        125 * u.km / u.s,
        None,
    ],
)
def test_scene_sumer(
    velocity_max: None | u.Quantity,
    remove_stripes: None | bool,
) -> None:
    # Names other than the defaults, so an argument that isn't passed through
    # changes the result.
    axis_x = "x"
    axis_y = "y"
    axis_velocity = "v"

    result = esis.flights.f1.data.synth.scene_sumer(
        time_start=_time_test,
        velocity_max=velocity_max,
        fill=0 * _unit,
        remove_stripes=remove_stripes,
        axis_detector_x=axis_x,
        axis_detector_y=axis_y,
        axis_velocity=axis_velocity,
    )

    assert isinstance(result, soho.sumer.SpectrographObservation)
    assert result.outputs.unit.is_equivalent(_unit)
    assert set(result.outputs.shape) == {axis_x, axis_y, axis_velocity}
    # the 76 exposures of the pass
    assert result.outputs.shape[axis_x] == 76
    assert np.all(np.isfinite(result.outputs))

    assert np.all(result.inputs.wavelength_rest == O_V.wavelength)

    radiance = result.outputs.sum(axis_velocity)
    assert np.median(radiance.ndarray) > 0

    velocity = result.inputs.velocity.cell_centers(axis_velocity)
    if velocity_max is not None:
        assert np.all(np.abs(velocity) <= velocity_max)
    else:
        # as far on both sides, to within a column
        width = np.abs(np.diff(velocity.ndarray)).max()
        assert np.abs(velocity.max() + velocity.min()) <= width


def test_scene_sumer_trim_slit() -> None:
    """The rows beyond the ends of the slit are left out unless asked not to."""
    axis_y = "detector_y"
    trimmed = esis.flights.f1.data.synth.scene_sumer(_time_test)
    whole = esis.flights.f1.data.synth.scene_sumer(_time_test, trim_slit=False)
    assert 0 < trimmed.outputs.shape[axis_y] < whole.outputs.shape[axis_y]


def test_scene_sumer_time_stop() -> None:
    """The start and end of a pass give the same scene as its start alone."""
    row = _scene_sumer._pass(astropy.time.Time(_time_test))
    result = esis.flights.f1.data.synth.scene_sumer(
        time_start=row["time_start"],
        time_stop=row["time_stop"],
    )
    expected = esis.flights.f1.data.synth.scene_sumer(_time_test)
    assert result.outputs.shape == expected.outputs.shape
    assert np.all(na.nominal(result.outputs) == na.nominal(expected.outputs))


class _Loaded(Exception):
    """Raised in place of loading the files of a pass."""


def test_scene_sumer_velocity_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """O V is looked for as far from where the files put it as the passes need."""
    arguments = dict()

    def from_level1(cls: type, **kwargs: object) -> typing.NoReturn:
        """
        Keep the arguments instead of loading the files.

        Raises
        ------
        _Loaded
            Always, once the arguments are kept.
        """
        arguments.update(kwargs)
        raise _Loaded

    monkeypatch.setattr(
        soho.sumer.SpectrographObservation,
        "from_level1",
        classmethod(from_level1),
    )
    with pytest.raises(_Loaded):
        esis.flights.f1.data.synth.scene_sumer(_time_test)
    assert arguments["velocity_search"] == _scene_sumer._velocity_search
    assert _scene_sumer._velocity_search >= 200 * u.km / u.s


def _tilt(scene: soho.sumer.SpectrographObservation) -> u.Quantity:
    """
    Return the change of the centroid of the line along the slit.

    The change is from one end of the slit to the other, of a straight line
    fitted to the centroid of each row, weighted by its radiance over the
    raster.
    """
    axis_x = scene.axis_detector_x
    axis_y = scene.axis_detector_y
    axis_velocity = scene.axis_wavelength
    radiance = na.nominal(scene.outputs)
    velocity = scene.inputs.velocity.cell_centers(axis_velocity)
    total = radiance.sum((axis_x, axis_velocity))
    centroid = (radiance * velocity).sum((axis_x, axis_velocity)) / total
    num = scene.outputs.shape[axis_y]
    index = na.arange(0, num, axis=axis_y)
    index = index - index.mean()
    slope = (index * (centroid - centroid.mean())).sum() / (index**2).sum()
    return (slope * (num - 1)).ndarray


def test_scene_sumer_tilt() -> None:
    """The tilt is removed from the default scene of a pass the table marks."""
    assert _scene_sumer._pass(astropy.time.Time(_time_test))["tilt"]
    kwargs = dict(velocity_max=60 * u.km / u.s, fill=0 * _unit)
    result = esis.flights.f1.data.synth.scene_sumer(_time_test, **kwargs)
    kept = esis.flights.f1.data.synth.scene_sumer(
        _time_test,
        remove_tilt=False,
        **kwargs,
    )
    assert np.abs(_tilt(kept)) > 3 * u.km / u.s
    assert np.abs(_tilt(result)) < np.abs(_tilt(kept)) / 3


class _Observation:
    """A stand-in for an observation, which counts the corrections made to it."""

    def __init__(self) -> None:
        """Start with no corrections made and no arguments kept."""
        self.calls = {"remove_stripes": 0, "remove_tilt": 0}
        self.arguments = dict()

    def remove_bad_steps(self) -> "_Observation":
        """Return the observation as it is."""
        return self

    def trim_slit(self) -> "_Observation":
        """Return the observation as it is."""
        return self

    def remove_stripes(self) -> "_Observation":
        """Count the call and return the observation as it is."""
        self.calls["remove_stripes"] += 1
        return self

    def remove_tilt(self) -> "_Observation":
        """Count the call and return the observation as it is."""
        self.calls["remove_tilt"] += 1
        return self


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> _Observation:
    """
    Return the stand-in which :func:`scene_sumer` loads in place of the files.

    The stand-in keeps the arguments it was loaded with,
    and :func:`scene_sumer` returns it instead of making a scene of it.
    """
    observation = _Observation()

    def path_sumer(
        time_start: astropy.time.Time,
        time_stop: astropy.time.Time,
    ) -> None:
        """Return nothing instead of the paths of the files."""

    def from_level1(cls: type, **kwargs: object) -> _Observation:
        """Keep the arguments and return the stand-in instead of loading the files."""
        observation.arguments.update(kwargs)
        return observation

    def observation_only(observation: _Observation, **kwargs: object) -> _Observation:
        """Return the observation instead of making a scene of it."""
        return observation

    monkeypatch.setattr(_scene_sumer, "path_sumer", path_sumer)
    monkeypatch.setattr(
        soho.sumer.SpectrographObservation,
        "from_level1",
        classmethod(from_level1),
    )
    monkeypatch.setattr(esis.data.synth, "scene_sumer_observation", observation_only)
    return observation


def _times(
    name: None | str,
) -> tuple[str | astropy.time.Time, None | astropy.time.Time]:
    """
    Return a start and a stop which name the test pass in one of several ways.

    Parameters
    ----------
    name
        :obj:`None` for its start alone, ``"pass"`` for its start and stop,
        ``"shorter"`` for its start and a stop 5 minutes before its own,
        ``"during"`` for a time during it and its stop,
        and ``"none"`` for an hour of a day of no pass.
    """
    row = _scene_sumer._pass(astropy.time.Time(_time_test))
    start = row["time_start"]
    stop = row["time_stop"]
    if name is None:
        return _time_test, None
    if name == "pass":
        return start, stop
    if name == "shorter":
        return start, stop - 5 * u.min
    if name == "during":
        return start + (stop - start) / 2, stop
    start = astropy.time.Time("2019-09-30T18:00")
    return start, start + 1 * u.h


@pytest.mark.parametrize("name", [None, "pass", "shorter", "during", "none"])
def test_scene_sumer_named(stand_in: _Observation, name: None | str) -> None:
    """
    The pass named by the start gives its slit and exposure time.

    It does whatever the stop is,
    and only the start alone gives its start and stop.
    """
    row = _scene_sumer._pass(astropy.time.Time(_time_test))
    time_start, time_stop = _times(name)
    esis.flights.f1.data.synth.scene_sumer(
        time_start=time_start,
        time_stop=time_stop,
    )
    arguments = stand_in.arguments
    if name == "none":
        assert arguments["slit"] is None
        assert arguments["timedelta"] is None
    else:
        assert arguments["slit"] == row["slit"]
        assert arguments["timedelta"] == row["exposure"]
    if name is None:
        assert arguments["time_start"] == row["time_start"]
        assert arguments["time_stop"] == row["time_stop"]
    else:
        assert arguments["time_start"] == time_start
        assert arguments["time_stop"] == time_stop


@pytest.mark.parametrize(
    argnames="column,method",
    argvalues=[
        ("polar", "remove_stripes"),
        ("tilt", "remove_tilt"),
    ],
)
@pytest.mark.parametrize(
    argnames="marked,name,value,expected",
    argvalues=[
        (True, None, None, True),
        (True, None, False, False),
        (False, None, None, False),
        (False, None, True, True),
        (True, "pass", None, True),
        (False, "pass", None, False),
        (True, "shorter", None, True),
        (True, "during", None, True),
        (True, "none", None, False),
        (True, "none", True, True),
    ],
)
def test_scene_sumer_defaults(
    monkeypatch: pytest.MonkeyPatch,
    stand_in: _Observation,
    column: str,
    method: str,
    marked: bool,
    name: None | str,
    value: None | bool,
    expected: bool,
) -> None:
    """
    The stripes and the tilt are removed by default where the table says.

    They are removed from the passes whose column of the table marks them,
    whatever the stop, and not if the start names no pass,
    unless asked otherwise.
    """
    rows = []
    for row in _scene_sumer._rows():
        if row["time_start"].startswith(_time_test):
            row = dict(row, polar="false", tilt="false")
            row[column] = "true" if marked else "false"
        rows.append(row)
    monkeypatch.setattr(_scene_sumer, "_rows", lambda: tuple(rows))

    time_start, time_stop = _times(name)
    esis.flights.f1.data.synth.scene_sumer(
        time_start=time_start,
        time_stop=time_stop,
        **{method: value},
    )
    assert stand_in.calls[method] == int(expected)
