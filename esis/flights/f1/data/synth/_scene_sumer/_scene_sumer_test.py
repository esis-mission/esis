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


class TestPassBetween:

    @pytest.mark.parametrize("offset", [-50, 0, 50] * u.s)
    def test_pass(self, offset: u.Quantity) -> None:
        """A pass is named by its start and stop, given to the minute."""
        row = _scene_sumer._pass(astropy.time.Time(_time_test))
        result = _scene_sumer._pass_between(
            row["time_start"] + offset,
            row["time_stop"] - offset,
        )
        assert result is not None
        assert result["time_start"] == row["time_start"]

    def test_none(self) -> None:
        """Times which are not the start and stop of a pass name none."""
        row = _scene_sumer._pass(astropy.time.Time(_time_test))
        stop = row["time_stop"] - 5 * u.min
        assert _scene_sumer._pass_between(row["time_start"], stop) is None


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


@pytest.mark.parametrize(
    argnames="column,method",
    argvalues=[
        ("polar", "remove_stripes"),
        ("tilt", "remove_tilt"),
    ],
)
@pytest.mark.parametrize(
    argnames="marked,stop,value,expected",
    argvalues=[
        (True, None, None, True),
        (True, None, False, False),
        (False, None, None, False),
        (False, None, True, True),
        (True, "pass", None, True),
        (False, "pass", None, False),
        (True, "other", None, False),
        (True, "other", True, True),
    ],
)
def test_scene_sumer_defaults(
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    method: str,
    marked: bool,
    stop: None | str,
    value: None | bool,
    expected: bool,
) -> None:
    """
    The stripes and the tilt are removed by default where the table says.

    They are removed from the passes whose column of the table marks them,
    named by their start alone or by their start and stop,
    and not if no pass is named, unless asked otherwise.
    """
    rows = []
    for row in _scene_sumer._rows():
        if row["time_start"].startswith(_time_test):
            row = dict(row, polar="false", tilt="false")
            row[column] = "true" if marked else "false"
        rows.append(row)
    monkeypatch.setattr(_scene_sumer, "_rows", lambda: tuple(rows))

    calls = {"remove_stripes": 0, "remove_tilt": 0}

    def spy(name: str) -> typing.Callable[..., soho.sumer.SpectrographObservation]:
        """Return a method which counts its calls and changes nothing."""

        def counted(
            self: soho.sumer.SpectrographObservation,
            **kwargs: object,
        ) -> soho.sumer.SpectrographObservation:
            """Count the call and return the observation as it is."""
            calls[name] += 1
            return self

        return counted

    def observation_only(
        observation: soho.sumer.SpectrographObservation,
        **kwargs: object,
    ) -> soho.sumer.SpectrographObservation:
        """Return the observation instead of making a scene of it."""
        return observation

    for name in calls:
        monkeypatch.setattr(soho.sumer.SpectrographObservation, name, spy(name))
    monkeypatch.setattr(esis.data.synth, "scene_sumer_observation", observation_only)

    time_start = _time_test
    time_stop = None
    if stop is not None:
        row = _scene_sumer._pass(astropy.time.Time(_time_test))
        time_start = row["time_start"]
        time_stop = row["time_stop"]
        if stop == "other":
            time_stop = time_stop - 5 * u.min
    esis.flights.f1.data.synth.scene_sumer(
        time_start=time_start,
        time_stop=time_stop,
        **{method: value},
    )
    assert calls[method] == int(expected)
