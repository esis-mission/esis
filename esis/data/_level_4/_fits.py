"""
Self-describing FITS serialization of the Level-4 data product.

The in-memory product carries its coordinates implicitly: the scene grid,
the per-line velocity axes, and the intensity units are all derived from
the instrument model at construction time.  A FITS file has to stand on
its own, so this module writes one image extension per spectral line,
each a ``(time, velocity, y, x)`` cube with a full world coordinate
system, plus extensions holding the exact coordinate vertices and the
per-frame diagnostics.
"""

from typing import TYPE_CHECKING, Self

import numpy as np

import astropy.constants
import astropy.io.fits
import astropy.time
import astropy.units as u
import named_arrays as na

if TYPE_CHECKING:  # pragma: no cover
    from ._level_4 import Level_4

__all__ = [
    "to_fits",
    "from_fits",
]

#: The name of the extension holding the wavelength vertices.
EXTNAME_WAVELENGTH = "WAVELENGTH"

#: The name of the extension holding the horizontal scene vertices.
EXTNAME_X = "POSITION_X"

#: The name of the extension holding the vertical scene vertices.
EXTNAME_Y = "POSITION_Y"

#: The name of the extension holding the per-frame diagnostics.
EXTNAME_DIAGNOSTICS = "DIAGNOSTICS"

#: The name of the extension holding the shaded-pixel mask.
EXTNAME_SHADOW = "SHADOW"

#: The name of the extension holding the inter-window cells.
EXTNAME_GAPS = "GAPS"


def _centers(vertices: np.ndarray) -> np.ndarray:
    """
    Convert an array of cell vertices to cell centers.

    Parameters
    ----------
    vertices
        The cell vertices.
    """
    return (vertices[:-1] + vertices[1:]) / 2


def _ndarray(a: na.AbstractScalarArray, axes: tuple[str, ...]) -> np.ndarray:
    """
    Extract the ndarray of `a` with the given axes leading, in order.

    Parameters
    ----------
    a
        The array to extract.
    axes
        The logical axes to move to the front, in the desired order.
    """
    source = [a.axes.index(ax) for ax in axes]
    return np.moveaxis(np.asarray(a.ndarray), source, range(len(axes)))


def to_fits(
    self: "Level_4",
    path,
    overwrite: bool = False,
    split: bool = False,
) -> None:
    """
    Write the product to a self-describing FITS file.

    Each spectral line becomes one image extension: a ``(time, velocity,
    y, x)`` cube whose world coordinate system gives helioprojective
    longitude and latitude in arcseconds, Doppler velocity in km/s about
    the rest wavelength of the line, and time in seconds from a reference
    epoch.  The scene coordinates are helioprojective because the
    distortion fit registers them to the AIA frame.

    The exact coordinate vertices are also written, as one-dimensional
    extensions, so :func:`from_fits` can restore the product without
    recomputing anything from the instrument model.

    Parameters
    ----------
    self
        The product to write.
    path
        The path of the file to write, or of the directory to fill if
        `split`.
    overwrite
        Whether to overwrite an existing file.
    split
        Whether to write one file per spectral line rather than one file
        holding every line.  Each file is a complete single-line product,
        so a reader interested in one line need not fetch the others.
        The cells between adjacent line windows are dropped, since they
        belong to no line and carry no calibrated signal.
    """
    if split:
        return _to_fits_split(self, path=path, overwrite=overwrite)

    axis_time = self.axis_time
    axis_wavelength = self.axis_wavelength
    axis_x = self.axis_x
    axis_y = self.axis_y

    outputs = self.outputs
    unit_outputs = na.unit(outputs)
    bunit = f"{unit_outputs:fits}" if unit_outputs is not None else ""

    time = self.inputs.time
    time = astropy.time.Time(time.ndarray if hasattr(time, "ndarray") else time)
    epoch = time[0]
    seconds = (time - epoch).to_value(u.s)
    seconds = np.atleast_1d(np.asarray(seconds, dtype=float))

    x = self.inputs.position.x.ndarray.to_value(u.arcsec)
    y = self.inputs.position.y.ndarray.to_value(u.arcsec)
    x_center = _centers(x)
    y_center = _centers(y)

    wavelength = self.inputs.wavelength.ndarray.to_value(u.AA)

    num_time = seconds.size
    num_wavelength = outputs.shape[axis_wavelength]

    primary = astropy.io.fits.PrimaryHDU()
    header = primary.header
    header["ORIGIN"] = ("esis", "the package that wrote this file")
    header["TELESCOP"] = ("ESIS", "EUV Snapshot Imaging Spectrograph")
    header["DATALVL"] = (4, "MART inversions on the scene grid")
    header["DATE"] = (astropy.time.Time.now().isot, "file creation time")
    header["NLINE"] = (self.num_line, "number of reconstructed spectral lines")
    header["NVEL"] = (self.num_velocity, "velocity bins per line window")
    header["NWAVE"] = (num_wavelength, "cells on the concatenated wavelength axis")
    header["NTIME"] = (num_time, "number of exposures")
    header["MJDREF"] = (epoch.mjd, "reference epoch of the time axis")
    header["DATE-OBS"] = (epoch.isot, "time of the first exposure")
    header["BUNIT"] = (bunit, "unit of the reconstructed radiance")
    header.add_comment("Level-4 ESIS product: time-dependent MART inversions.")
    header.add_comment("One image extension per spectral line, each a 4-D cube")
    header.add_comment("with axes (time, velocity, latitude, longitude).")
    header.add_comment("Scene coordinates are helioprojective, registered to")
    header.add_comment("the AIA frame by the instrument distortion fit.")

    hdus = [primary]

    used = np.zeros(num_wavelength, dtype=bool)
    for i in range(self.num_line):
        window = self.window(i)[axis_wavelength]
        used[window] = True

        cube = outputs[{axis_wavelength: window}]
        data = _ndarray(cube, (axis_time, axis_wavelength, axis_y, axis_x))

        hdu = astropy.io.fits.ImageHDU(np.asarray(data, dtype=np.float32))
        h = hdu.header
        h["EXTNAME"] = _extname(self, i)
        h["EXTVER"] = (i + 1, "index of this line, so the name is unambiguous")
        h["BUNIT"] = (bunit, "unit of the reconstructed radiance")

        rest = self.wavelength_center[{self.axis_line: i}].ndarray.to(u.AA)
        velocity = _window_velocity(self, i)
        velocity_center = _centers(velocity)

        h["CTYPE1"] = ("HPLN-TAN", "helioprojective longitude")
        h["CUNIT1"] = "arcsec"
        h["CRPIX1"] = (1.0, "reference pixel, 1-indexed at the first cell")
        h["CRVAL1"] = (float(x_center[0]), "coordinate of the reference pixel")
        h["CDELT1"] = (float(x_center[1] - x_center[0]), "scene pitch")

        h["CTYPE2"] = ("HPLT-TAN", "helioprojective latitude")
        h["CUNIT2"] = "arcsec"
        h["CRPIX2"] = (1.0, "reference pixel, 1-indexed at the first cell")
        h["CRVAL2"] = (float(y_center[0]), "coordinate of the reference pixel")
        h["CDELT2"] = (float(y_center[1] - y_center[0]), "scene pitch")

        h["CTYPE3"] = ("VOPT", "Doppler velocity, optical convention")
        h["CUNIT3"] = "km/s"
        h["CRPIX3"] = (1.0, "reference pixel, 1-indexed at the first cell")
        h["CRVAL3"] = (float(velocity_center[0]), "coordinate of the reference pixel")
        h["CDELT3"] = (float(velocity_center[1] - velocity_center[0]), "velocity bin")
        h["RESTWAV"] = (
            float(rest.to_value(u.m)),
            "rest wavelength of the velocity axis, meters",
        )
        _write_members(h, self, i)

        h["CTYPE4"] = ("TIME", "seconds from MJDREF")
        h["CUNIT4"] = "s"
        h["CRPIX4"] = 1.0
        h["CRVAL4"] = (float(seconds[0]), "coordinate of the reference pixel")
        cadence = float(np.median(np.diff(seconds))) if num_time > 1 else 1.0
        h["CDELT4"] = (cadence, "median exposure cadence")
        h["MJDREF"] = (epoch.mjd, "reference epoch of the time axis")
        if num_time > 1:
            residual = np.abs(seconds - (seconds[0] + cadence * np.arange(num_time)))
            h["TIMEDEV"] = (
                float(residual.max()),
                "max deviation from the linear time axis, seconds",
            )
            h.add_comment(
                f"The time axis is linear to {residual.max():.3g} s; the exact"
            )
            h.add_comment(f"exposure times are in the {EXTNAME_DIAGNOSTICS} table.")

        h["LINE"] = (
            _ascii(self.label(i)),
            "spectral line, including blends in this window",
        )
        h["WSLICE0"] = (int(window.start or 0), "first cell on the concatenated axis")
        h["WSLICE1"] = (
            int(window.stop if window.stop is not None else num_wavelength),
            "one past the last cell on the concatenated axis",
        )
        hdus.append(hdu)

    if not used.all():
        # cells between adjacent line windows, kept only so the concatenated
        # wavelength axis can be restored exactly; they are an artifact of
        # the concatenation and carry no calibrated signal, and are omitted
        # entirely when empty, since :func:`from_fits` restores zeros anyway
        index_gap = np.flatnonzero(~used)
        gaps = outputs[{axis_wavelength: na.ScalarArray(index_gap, axes="_gap")}]
        data = _ndarray(gaps, (axis_time, "_gap", axis_y, axis_x))
        if np.any(data):
            hdu = astropy.io.fits.ImageHDU(np.asarray(data, dtype=np.float32))
            hdu.header["EXTNAME"] = EXTNAME_GAPS
            hdu.header["BUNIT"] = bunit
            hdu.header.add_comment("Inter-window cells of the concatenated wavelength")
            hdu.header.add_comment("axis. Not a calibrated product; present so the")
            hdu.header.add_comment("original array can be restored exactly.")
            hdus.append(hdu)
            hdus.append(
                _column_hdu(index_gap.astype(np.int32), "GAPINDEX", "", "cell indices")
            )

    hdus.append(
        _column_hdu(wavelength, EXTNAME_WAVELENGTH, "Angstrom", "cell vertices")
    )
    hdus.append(_column_hdu(x, EXTNAME_X, "arcsec", "cell vertices"))
    hdus.append(_column_hdu(y, EXTNAME_Y, "arcsec", "cell vertices"))

    hdus.append(_diagnostics_hdu(self, time))

    if self.where_shadow is not None:
        shadow = self.where_shadow
        axes = tuple(shadow.axes)
        data = np.asarray(shadow.ndarray, dtype=np.uint8)
        hdu = astropy.io.fits.ImageHDU(data)
        hdu.header["EXTNAME"] = EXTNAME_SHADOW
        hdu.header["AXES"] = (",".join(axes), "logical axes of this mask")
        hdu.header.add_comment("Detector pixels shaded by the frame-transfer mask,")
        hdu.header.add_comment("excluded from the inversion. 1 where shaded.")
        hdus.append(hdu)

    astropy.io.fits.HDUList(hdus).writeto(path, overwrite=overwrite)


def line_filename(self: "Level_4", index_line: int) -> str:
    """
    Build the filename holding one spectral line of a split product.

    Parameters
    ----------
    self
        The product being written.
    index_line
        The index of the spectral line.
    """
    return f"esis_level_4_{_extname(self, index_line).lower()}.fits"


def single_line(self: "Level_4", index_line: int) -> "Level_4":
    """
    Extract one spectral line as a standalone single-line product.

    The window is re-based so it starts at the beginning of the
    wavelength axis, which is what the default
    :meth:`~esis.data.Level_4.window` expects of a one-line product.

    Parameters
    ----------
    self
        The product to extract from.
    index_line
        The index of the spectral line.
    """
    import dataclasses

    axis_wavelength = self.axis_wavelength
    axis_line = self.axis_line

    window = self.window(index_line)[axis_wavelength]
    start = window.start or 0
    stop = (
        window.stop if window.stop is not None else self.outputs.shape[axis_wavelength]
    )

    inputs = dataclasses.replace(
        self.inputs,
        wavelength=self.inputs.wavelength[{axis_wavelength: slice(start, stop + 1)}],
    )

    index = {axis_line: slice(index_line, index_line + 1)}
    return dataclasses.replace(
        self,
        inputs=inputs,
        outputs=self.outputs[{axis_wavelength: window}],
        wavelength_center=self.wavelength_center[index],
        label_line=(None if self.label_line is None else [self.label_line[index_line]]),
        members_line=(
            None if self.members_line is None else [self.members_line[index_line]]
        ),
    )


def _to_fits_split(
    self: "Level_4",
    path,
    overwrite: bool = False,
) -> None:
    """
    Write one file per spectral line into a directory.

    Parameters
    ----------
    self
        The product to write.
    path
        The directory to fill.
    overwrite
        Whether to overwrite existing files.
    """
    import pathlib

    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    for i in range(self.num_line):
        file = path / line_filename(self, i)
        to_fits(single_line(self, i), file, overwrite=overwrite)
        with astropy.io.fits.open(file, mode="update") as hdul:
            header = hdul[0].header
            header["LINEIDX"] = (i, "index of this line in the full product")
            header["NLINEALL"] = (
                self.num_line,
                "lines in the full product this file came from",
            )
            header.add_comment("One line of a product split across files; the")
            header.add_comment("companion files hold the remaining lines.")


def _ascii(value: str) -> str:
    """
    Escape a string so it can be stored in a FITS header.

    FITS headers hold printable ASCII only, but a line label may carry a
    unit symbol such as the angstrom sign.  Escaping rather than
    transliterating keeps :func:`from_fits` exact.

    Parameters
    ----------
    value
        The string to escape.
    """
    return value.encode("unicode_escape").decode("ascii")


def _unascii(value: str) -> str:
    """
    Invert :func:`_ascii`.

    Parameters
    ----------
    value
        The escaped string.
    """
    return value.encode("ascii").decode("unicode_escape")


def _extname(self: "Level_4", index_line: int) -> str:
    """
    Build a FITS extension name for the given line.

    Parameters
    ----------
    self
        The product being written.
    index_line
        The index of the spectral line.
    """
    label = self.label(index_line)
    result = "".join(c if (c.isascii() and c.isalnum()) else "_" for c in label).upper()
    result = "_".join(part for part in result.split("_") if part)
    return (result or f"LINE{index_line}")[:68]


def _write_members(
    header: astropy.io.fits.Header,
    self: "Level_4",
    index_line: int,
) -> None:
    """
    Record the spectral lines tied into one window.

    A tied window holds a single reconstructed scene shared by several
    spectral lines, so the file needs every member's rest wavelength and
    photon ratio, not just the one the velocity axis is measured against.

    Parameters
    ----------
    header
        The header of the window's extension.
    self
        The product being written.
    index_line
        The index of the spectral line.
    """
    if self.members_line is None:
        return
    members = self.members_line[index_line]
    header["NMEMBER"] = (len(members), "spectral lines tied into this window")
    for j, (wavelength, ratio) in enumerate(members):
        header[f"MEMWAV{j + 1}"] = (
            float(u.Quantity(wavelength).to_value(u.m)),
            f"rest wavelength of member {j + 1}, meters",
        )
        header[f"MEMRAT{j + 1}"] = (
            float(ratio),
            f"photon ratio of member {j + 1} to member 1",
        )
    if len(members) > 1:
        header.add_comment("This window is a tied combination: the members")
        header.add_comment("share one reconstructed scene, their relative")
        header.add_comment("brightness fixed at the ratios above. The cube is")
        header.add_comment("that shared solution, not any one member alone.")


def _read_members(
    header: astropy.io.fits.Header,
) -> None | list[tuple[u.Quantity, float]]:
    """
    Read the tied members of one window, if the file records them.

    Parameters
    ----------
    header
        The header of the window's extension.
    """
    if "NMEMBER" not in header:
        return None
    result = []
    for j in range(header["NMEMBER"]):
        result.append(
            (
                (header[f"MEMWAV{j + 1}"] * u.m).to(u.AA),
                float(header[f"MEMRAT{j + 1}"]),
            )
        )
    return result


def _window_velocity(self: "Level_4", index_line: int) -> np.ndarray:
    """
    Compute the velocity vertices of one line window, in km/s.

    Parameters
    ----------
    self
        The product being written.
    index_line
        The index of the spectral line.
    """
    axis = self.axis_wavelength
    window = self.window(index_line)[axis]
    start = window.start or 0
    stop = window.stop if window.stop is not None else self.outputs.shape[axis]
    wavelength = self.inputs.wavelength.ndarray.to(u.AA)[start : stop + 1]
    rest = self.wavelength_center[{self.axis_line: index_line}].ndarray.to(u.AA)
    result = (wavelength / rest - 1) * astropy.constants.c
    return result.to_value(u.km / u.s)


def _column_hdu(
    data: np.ndarray,
    name: str,
    unit: str,
    comment: str,
) -> astropy.io.fits.ImageHDU:
    """
    Build a one-dimensional image extension holding a coordinate array.

    Parameters
    ----------
    data
        The array to store.
    name
        The extension name.
    unit
        The unit of the array.
    comment
        A comment describing the array.
    """
    hdu = astropy.io.fits.ImageHDU(np.asarray(data))
    hdu.header["EXTNAME"] = name
    hdu.header["BUNIT"] = (unit, comment)
    return hdu


def _diagnostics_hdu(
    self: "Level_4",
    time: astropy.time.Time,
) -> astropy.io.fits.BinTableHDU:
    """
    Build the per-frame diagnostics table.

    Parameters
    ----------
    self
        The product being written.
    time
        The exposure times.
    """
    columns = [
        astropy.io.fits.Column(
            name="MJD",
            format="D",
            unit="d",
            array=np.atleast_1d(time.mjd),
        )
    ]

    axis_time = self.axis_time
    num_time = np.atleast_1d(time.mjd).size

    def _add(value, name: str, unit: str) -> None:
        """
        Append a column for a per-frame quantity.

        Parameters
        ----------
        value
            The quantity to store, or :obj:`None` to skip.
        name
            The column name.
        unit
            The unit of the column.
        """
        if value is None:
            return
        axes = [ax for ax in value.axes if ax != axis_time]
        data = _ndarray(value, (axis_time,) + tuple(axes))
        data = np.asarray(data, dtype=float).reshape(num_time, -1)
        width = data.shape[1]
        columns.append(
            astropy.io.fits.Column(
                name=name,
                format=f"{width}D" if width > 1 else "D",
                unit=unit,
                array=data if width > 1 else data[:, 0],
            )
        )

    _add(self.mean_chi_squared, "CHI2", "")
    _add(self.factor_norm, "FACTOR_NORM", "")
    if self.num_iteration is not None:
        columns.append(
            astropy.io.fits.Column(
                name="NITER",
                format="J",
                array=np.asarray(
                    _ndarray(self.num_iteration, (axis_time,)), dtype=np.int32
                ),
            )
        )

    hdu = astropy.io.fits.BinTableHDU.from_columns(columns)
    hdu.header["EXTNAME"] = EXTNAME_DIAGNOSTICS
    hdu.header.add_comment("Per-exposure inversion diagnostics. CHI2 and")
    hdu.header.add_comment("FACTOR_NORM have one entry per camera channel.")
    return hdu


def from_fits(
    cls: type["Level_4"],
    path,
    instrument=None,
) -> Self:
    """
    Read a product written by :func:`to_fits`.

    The optical model is not stored in the file, so `instrument` is
    :obj:`None` unless supplied; every coordinate, the reconstructed
    radiance, and the diagnostics are restored exactly.

    Given a directory, every single-line file it holds is read and
    reassembled into one multi-line product, in the line order the files
    record.  Given the file of one line of a split product, that line is
    returned on its own, which is the point of splitting.

    Parameters
    ----------
    cls
        The :class:`~esis.data.Level_4` class.
    path
        The path of the file to read, or of a directory of single-line
        files to reassemble.
    instrument
        A model of the optical system to attach to the result.
    """
    import pathlib

    if pathlib.Path(path).is_dir():
        return _from_fits_split(cls, path=path, instrument=instrument)

    with astropy.io.fits.open(path) as hdul:
        header = hdul[0].header

        num_line = header["NLINE"]
        num_velocity = header["NVEL"]
        num_wavelength = header["NWAVE"]
        unit_outputs = (
            u.Unit(header["BUNIT"]) if header["BUNIT"] else u.dimensionless_unscaled
        )

        axis_time = "time"
        axis_wavelength = "wavelength"
        axis_x = "field_x"
        axis_y = "field_y"
        axis_line = "line"

        wavelength = hdul[EXTNAME_WAVELENGTH].data * u.AA
        x = hdul[EXTNAME_X].data * u.arcsec
        y = hdul[EXTNAME_Y].data * u.arcsec

        diagnostics = hdul[EXTNAME_DIAGNOSTICS].data
        time = astropy.time.Time(np.asarray(diagnostics["MJD"]), format="mjd")

        windows = []
        labels = []
        rest = []
        members = []
        for i in range(num_line):
            hdu = hdul[i + 1]
            windows.append((hdu.header["WSLICE0"], hdu.header["WSLICE1"]))
            labels.append(_unascii(hdu.header["LINE"]))
            rest.append(hdu.header["RESTWAV"])
            members.append(_read_members(hdu.header))
        members_line = members if all(m is not None for m in members) else None

        num_time = len(time)
        num_x = x.size - 1
        num_y = y.size - 1
        outputs = np.zeros(
            (num_time, num_wavelength, num_x, num_y),
            dtype=np.float32,
        )
        for i in range(num_line):
            start, stop = windows[i]
            # stored as (time, velocity, y, x); the product is (..., x, y)
            outputs[:, start:stop] = np.moveaxis(hdul[i + 1].data, 3, 2)

        if EXTNAME_GAPS in hdul:
            index_gap = np.asarray(hdul["GAPINDEX"].data, dtype=int)
            outputs[:, index_gap] = np.moveaxis(hdul[EXTNAME_GAPS].data, 3, 2)

        outputs = na.ScalarArray(
            outputs * unit_outputs,
            axes=(axis_time, axis_wavelength, axis_x, axis_y),
        )

        wavelength_center = na.ScalarArray(
            (np.array(rest) * u.m).to(u.AA),
            axes=(axis_line,),
        )

        mean_chi_squared = None
        if "CHI2" in diagnostics.names:
            mean_chi_squared = na.ScalarArray(
                np.asarray(diagnostics["CHI2"]),
                axes=(axis_time, "channel"),
            )
        factor_norm = None
        if "FACTOR_NORM" in diagnostics.names:
            factor_norm = na.ScalarArray(
                np.asarray(diagnostics["FACTOR_NORM"]),
                axes=(axis_time, "channel"),
            )
        num_iteration = None
        if "NITER" in diagnostics.names:
            num_iteration = na.ScalarArray(
                np.asarray(diagnostics["NITER"]),
                axes=(axis_time,),
            )

        where_shadow = None
        if EXTNAME_SHADOW in hdul:
            hdu = hdul[EXTNAME_SHADOW]
            where_shadow = na.ScalarArray(
                np.asarray(hdu.data, dtype=bool),
                axes=tuple(hdu.header["AXES"].split(",")),
            )

        inputs = na.TemporalSpectralPositionalVectorArray(
            time=na.ScalarArray(time, axes=(axis_time,)),
            wavelength=na.ScalarArray(wavelength, axes=(axis_wavelength,)),
            position=na.Cartesian2dVectorArray(
                x=na.ScalarArray(x, axes=(axis_x,)),
                y=na.ScalarArray(y, axes=(axis_y,)),
            ),
        )

        return cls(
            inputs=inputs,
            outputs=outputs,
            instrument=instrument,
            wavelength_center=wavelength_center,
            label_line=labels,
            members_line=members_line,
            num_velocity=num_velocity,
            mean_chi_squared=mean_chi_squared,
            num_iteration=num_iteration,
            factor_norm=factor_norm,
            where_shadow=where_shadow,
            axis_time=axis_time,
            axis_wavelength=axis_wavelength,
            axis_x=axis_x,
            axis_y=axis_y,
            axis_line=axis_line,
        )


def _from_fits_split(
    cls: type["Level_4"],
    path,
    instrument=None,
) -> Self:
    """
    Reassemble a product split across one file per spectral line.

    The single-line files hold only their own window, so the windows are
    restored onto the standard concatenated layout, which separates
    adjacent windows by one empty cell.

    Parameters
    ----------
    cls
        The :class:`~esis.data.Level_4` class.
    path
        The directory holding the single-line files.
    instrument
        A model of the optical system to attach to the result.

    Raises
    ------
    FileNotFoundError
        If the directory holds no FITS files.
    """
    import dataclasses
    import pathlib

    files = sorted(pathlib.Path(path).glob("*.fits"))
    if not files:
        raise FileNotFoundError(f"no FITS files in {path}")

    def _index(file) -> int:
        """
        Look up the line index a file records.

        Parameters
        ----------
        file
            The file to inspect.
        """
        with astropy.io.fits.open(file) as hdul:
            return hdul[0].header.get("LINEIDX", 0)

    files = sorted(files, key=_index)
    lines = [from_fits(cls, file, instrument=instrument) for file in files]

    first = lines[0]
    axis_wavelength = first.axis_wavelength
    axis_line = first.axis_line
    num_line = len(lines)
    num_velocity = first.num_velocity
    num = num_velocity + 1

    outputs = None
    wavelength = []
    for i, line in enumerate(lines):
        if outputs is None:
            shape = dict(line.outputs.shape)
            shape[axis_wavelength] = num_line * num - 1
            outputs = np.zeros(
                tuple(shape.values()),
                dtype=np.asarray(line.outputs.ndarray).dtype,
            )
            axes = tuple(shape)
        index = [slice(None)] * len(axes)
        index[axes.index(axis_wavelength)] = slice(i * num, i * num + num_velocity)
        outputs[tuple(index)] = _ndarray(line.outputs, axes)
        wavelength.append(line.inputs.wavelength.ndarray.to_value(u.AA))

    unit_outputs = na.unit(first.outputs)
    inputs = dataclasses.replace(
        first.inputs,
        wavelength=na.ScalarArray(
            np.concatenate(wavelength) * u.AA,
            axes=(axis_wavelength,),
        ),
    )

    return dataclasses.replace(
        first,
        inputs=inputs,
        outputs=na.ScalarArray(outputs * unit_outputs, axes=axes),
        wavelength_center=na.concatenate(
            [line.wavelength_center for line in lines],
            axis=axis_line,
        ),
        label_line=[line.label(0) for line in lines],
        members_line=(
            None
            if any(line.members_line is None for line in lines)
            else [line.members_line[0] for line in lines]
        ),
    )
