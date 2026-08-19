import pathlib
import named_arrays as na
import esis


def test_path_fits():
    axis_time = "time"
    axis_chan = "channel"
    result = esis.flights.f1.data.path_fits(axis_time, axis_chan)
    assert isinstance(result, na.ScalarArray)
    assert result.shape[axis_chan] == 4
    for item in result.ndarray.flat:
        assert isinstance(item, pathlib.Path)


def test_path_directory():
    result = esis.flights.f1.data.path_directory()
    assert isinstance(result, pathlib.Path)
    assert result.is_dir()
    assert list(result.glob("*.fit.gz"))


def test_path_cache(monkeypatch, tmp_path: pathlib.Path):
    """The download directory can be moved with an environment variable."""
    monkeypatch.setenv("ESIS_DATA_DIR", str(tmp_path))
    assert esis.flights.f1.data._fits._fits._path_cache() == tmp_path


def test_doi():
    assert esis.flights.f1.data.doi.startswith("10.5281/zenodo.")
