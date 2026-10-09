import pathlib
import pytest
import numpy as np
import astropy.units as u
import astropy.time
import astropy.io.fits
import named_arrays as na
import esis

_axis_time = "time"


def _header(path: pathlib.Path) -> astropy.io.fits.Header:
    """Read the header of the compressed image in an AIA Level 1 file."""
    return astropy.io.fits.getheader(path, ext=1)


def _time(path: na.AbstractScalarArray) -> na.ScalarArray:
    """Read when the exposure of each image in `path` began."""
    time = astropy.time.Time([_header(p)["DATE-OBS"] for p in path.ndarray.flat])
    return na.ScalarArray(
        ndarray=time.reshape(path.ndarray.shape),
        axes=path.axes,
    )


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        304 * u.AA,
        na.ScalarArray([304, 193, 304] * u.AA, axes="wavelength"),
    ],
)
class TestPathAIA:

    def test_path_aia(
        self,
        wavelength: u.Quantity | na.AbstractScalarArray,
    ) -> None:
        result = esis.flights.f1.data.path_aia(wavelength, _axis_time)
        assert isinstance(result, na.ScalarArray)
        assert result.shape == na.shape(wavelength) | {_axis_time: 30}
        for index in result.ndindex():
            path = result[index].ndarray
            assert isinstance(path, pathlib.Path)
            assert path.is_file()
            index_wavelength = {a: index[a] for a in na.shape(wavelength)}
            channel = na.as_named_array(wavelength)[index_wavelength].ndarray
            assert _header(path)["WAVELNTH"] * u.AA == channel
        time = _time(result)
        time_next = time[{_axis_time: slice(1, None)}]
        time_previous = time[{_axis_time: slice(None, ~0)}]
        assert np.all(time_next > time_previous)

    def test_time_range(
        self,
        wavelength: u.Quantity | na.AbstractScalarArray,
    ) -> None:
        time_start = astropy.time.Time("2019-09-30T18:07:00")
        time_stop = astropy.time.Time("2019-09-30T18:08:00")
        result = esis.flights.f1.data.path_aia(
            wavelength=wavelength,
            axis_time=_axis_time,
            time_start=time_start,
            time_stop=time_stop,
        )
        # one image every 12 seconds
        assert result.shape[_axis_time] == 5
        time = _time(result)
        assert np.all(time >= time_start)
        assert np.all(time <= time_stop)

    def test_time_range_empty(
        self,
        wavelength: u.Quantity | na.AbstractScalarArray,
    ) -> None:
        time = astropy.time.Time("2019-09-30T18:00:00")
        with pytest.raises(ValueError, match="no images"):
            esis.flights.f1.data.path_aia(
                wavelength=wavelength,
                axis_time=_axis_time,
                time_start=time,
                time_stop=time,
            )

    def test_limit(
        self,
        wavelength: u.Quantity | na.AbstractScalarArray,
    ) -> None:
        result = esis.flights.f1.data.path_aia(wavelength, _axis_time, limit=3)
        everything = esis.flights.f1.data.path_aia(wavelength, _axis_time)
        index = na.ScalarArray(np.array([5, 15, 25]), axes=_axis_time)
        assert np.all(result == everything[{_axis_time: index}])


def test_path_aia_channel() -> None:
    with pytest.raises(ValueError, match="channels"):
        esis.flights.f1.data.path_aia(171 * u.AA, _axis_time)


def test_path_aia_level_1() -> None:
    """The images span the Level-1 observations of ESIS."""
    l1 = esis.flights.f1.data.level_1()
    time_start = l1.inputs.time_start[{l1.axis_time: 0}].ndarray.min()
    time_stop = l1.inputs.time_end[{l1.axis_time: ~0}].ndarray.max()
    wavelength = na.ScalarArray([193, 304] * u.AA, axes="wavelength")
    time = _time(esis.flights.f1.data.path_aia(wavelength, _axis_time))
    assert np.all(time[{_axis_time: 0}] <= time_start)
    assert np.all(time[{_axis_time: ~0}] >= time_stop)
