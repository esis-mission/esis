import functools
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import matplotlib.pyplot as plt
import esis

ctis = pytest.importorskip("ctis")

if not hasattr(ctis.instruments, "OptikaInstrument"):  # pragma: nocover
    pytest.skip(
        reason="ctis is missing OptikaInstrument (needs ctis PR #18)",
        allow_module_level=True,
    )


@functools.cache
def _level_4() -> esis.data.Level_4:
    """Invert three frames of the flight data on a deliberately coarse grid."""
    spectrum = esis.flights.f1.spectrum
    wavelength_center = na.ScalarArray(
        ndarray=u.Quantity(
            [
                spectrum.He_I.wavelength,
                spectrum.Mg_X.wavelength,
                spectrum.O_V.wavelength,
            ]
        ),
        axes="line",
    )
    width_doppler = na.ScalarArray(
        ndarray=u.Quantity(
            [
                spectrum.He_I.width_doppler,
                spectrum.Mg_X.width_doppler,
                spectrum.O_V.width_doppler,
            ]
        ),
        axes="line",
    )
    return esis.data.Level_4.from_level_1(
        a=esis.flights.f1.data.level_1()[dict(time=slice(14, 17))],
        wavelength_center=wavelength_center,
        width_doppler=width_doppler,
        instrument=esis.flights.f1.optics.distortion_fit(num_distribution=0),
        pitch_scene=16 * u.arcsec,
        num_velocity=2,
        index_time_reference=1,
        num_iteration=2,
    )


class TestLevel_4:
    def test_type(self):
        assert isinstance(_level_4(), esis.data.Level_4)

    def test_outputs(self):
        a = _level_4()
        shape = a.outputs.shape
        assert shape[a.axis_time] == 3
        assert shape[a.axis_wavelength] == a.num_line * (a.num_velocity + 1) - 1
        assert shape[a.axis_x] > 0
        assert shape[a.axis_y] > 0
        assert np.all(a.outputs >= 0)

    def test_mean_chi_squared(self):
        a = _level_4()
        assert a.axis_time in a.mean_chi_squared.shape
        assert np.all(np.isfinite(a.mean_chi_squared))

    def test_num_iteration(self):
        a = _level_4()
        assert a.num_iteration.shape == {a.axis_time: 3}
        assert np.all(a.num_iteration <= 2)

    def test_factor_norm(self):
        a = _level_4()
        assert np.all(a.factor_norm > 0)

    def test_where_shadow(self):
        a = _level_4()
        assert a.where_shadow is not None
        assert a.where_shadow.dtype == bool
        assert np.any(a.where_shadow)

    def test_window(self):
        a = _level_4()
        num = a.num_velocity + 1
        for i in range(a.num_line):
            window = a.window(i)[a.axis_wavelength]
            assert window.stop - window.start == a.num_velocity
            assert window.start == i * num

    def test_velocity(self):
        a = _level_4()
        velocity = a.velocity
        assert velocity.shape == {a.axis_wavelength: a.num_velocity + 1}
        velocity_first = velocity[{a.axis_wavelength: 0}].ndarray
        velocity_last = velocity[{a.axis_wavelength: ~0}].ndarray
        assert np.allclose(velocity_first, -200 * u.km / u.s)
        assert np.allclose(velocity_last, +200 * u.km / u.s)

    def test_intensity(self):
        a = _level_4()
        intensity = a.intensity
        assert a.axis_line in intensity.shape
        assert intensity.shape[a.axis_line] == a.num_line
        assert np.all(intensity >= 0)

    def test_velocity_mean(self):
        a = _level_4()
        velocity = a.velocity_mean
        assert a.axis_line in velocity.shape
        assert np.all(np.abs(velocity) <= 200 * u.km / u.s)

    def test_velocity_width(self):
        a = _level_4()
        width = a.velocity_width
        assert a.axis_line in width.shape
        assert np.all(width >= 0)

    def test_to_fits_from_fits(self, tmp_path):
        a = _level_4()
        path = tmp_path / "level_4.fits"
        a.to_fits(path)

        b = esis.data.Level_4.from_fits(path)

        assert b.num_line == a.num_line
        assert b.num_velocity == a.num_velocity
        assert b.label_line == [a.label(i) for i in range(a.num_line)]

        assert np.allclose(
            b.outputs.ndarray.value,
            a.outputs.ndarray.value,
            rtol=1e-6,
        )
        assert na.unit(b.outputs) == na.unit(a.outputs)

        assert np.allclose(
            b.inputs.wavelength.ndarray.to_value(u.AA),
            a.inputs.wavelength.ndarray.to_value(u.AA),
        )
        assert np.allclose(
            b.inputs.position.x.ndarray.to_value(u.arcsec),
            a.inputs.position.x.ndarray.to_value(u.arcsec),
        )
        assert np.allclose(
            b.inputs.time.ndarray.mjd,
            a.inputs.time.ndarray.mjd,
        )
        assert np.allclose(
            b.wavelength_center.ndarray.to_value(u.AA),
            a.wavelength_center.ndarray.to_value(u.AA),
        )
        assert np.allclose(
            b.mean_chi_squared.ndarray,
            np.asarray(a.mean_chi_squared.ndarray),
        )
        assert np.all(b.num_iteration.ndarray == np.asarray(a.num_iteration.ndarray))
        assert b.where_shadow is not None
        assert np.all(
            b.where_shadow.ndarray
            == a.where_shadow.broadcast_to(a.where_shadow.shape).ndarray
        )

        # the derived quantities must survive the round trip
        assert np.allclose(
            b.intensity.ndarray.value,
            a.intensity.ndarray.value,
            rtol=1e-6,
        )
        assert np.allclose(
            b.velocity.ndarray.to_value(u.km / u.s),
            a.velocity.ndarray.to_value(u.km / u.s),
        )

    def test_to_fits_wcs(self, tmp_path):
        """The written WCS must place the scene where the product does."""
        import astropy.io.fits
        import astropy.wcs

        a = _level_4()
        path = tmp_path / "level_4.fits"
        a.to_fits(path)

        with astropy.io.fits.open(path) as hdul:
            hdu = hdul[1]
            assert hdu.header["EXTNAME"]
            wcs = astropy.wcs.WCS(hdu.header)

            # the cube is (time, velocity, y, x); FITS axes run the other way
            num_time, num_velocity, num_y, num_x = hdu.data.shape
            assert num_velocity == a.num_velocity
            assert num_time == a.shape[a.axis_time]

            # astropy normalizes celestial axes to degrees and spectral axes
            # to SI, whatever CUNIT says
            world = wcs.pixel_to_world_values(0, 0, 0, 0)
            # the field straddles disc centre, so half of it has negative
            # helioprojective longitude, which wcslib reports on the
            # [0, 360) branch; wrap it back the way sunpy's Helioprojective
            # frame does before comparing
            longitude = ((world[0] + 180) % 360 - 180) * u.deg
            longitude = longitude.to_value(u.arcsec)
            latitude = (world[1] * u.deg).to_value(u.arcsec)
            velocity_world = (world[2] * u.m / u.s).to_value(u.km / u.s)

            x = a.inputs.position.x.ndarray.to_value(u.arcsec)
            y = a.inputs.position.y.ndarray.to_value(u.arcsec)
            # tolerance well below the scene pitch, but above the departure
            # of the tangent projection from the linear scene grid
            assert np.isclose(longitude, (x[0] + x[1]) / 2, atol=1e-2)
            assert np.isclose(latitude, (y[0] + y[1]) / 2, atol=1e-2)

            # the tangent point sits mid-field, so the reference pixel
            # reproduces its own coordinate exactly
            world = wcs.pixel_to_world_values(
                hdu.header["CRPIX1"] - 1,
                hdu.header["CRPIX2"] - 1,
                0,
                0,
            )
            longitude = ((world[0] + 180) % 360 - 180) * u.deg
            assert np.isclose(
                longitude.to_value(u.arcsec),
                hdu.header["CRVAL1"],
                atol=1e-6,
            )

            velocity = a.velocity.ndarray.to_value(u.km / u.s)
            assert np.isclose(
                velocity_world,
                (velocity[0] + velocity[1]) / 2,
                rtol=1e-6,
            )

    def test_to_fits_members(self, tmp_path):
        """Tied windows must record every member line and its ratio."""
        import dataclasses

        a = _level_4()
        # a two-member window, as the tied production configuration uses:
        # two lines of one ion sharing a solution at a fixed photon ratio
        members = [[(w, 1.0)] for w in [584.334, 599.590, 629.732] * u.AA]
        members[1] = [(609.793 * u.AA, 1.0), (624.941 * u.AA, 0.52)]
        a = dataclasses.replace(a, members_line=members)

        path = tmp_path / "level_4_tied.fits"
        a.to_fits(path)
        b = esis.data.Level_4.from_fits(path)

        assert b.members_line is not None
        assert len(b.members_line) == a.num_line
        for expected, found in zip(members, b.members_line):
            assert len(found) == len(expected)
            for (w0, r0), (w1, r1) in zip(expected, found):
                assert np.isclose(
                    u.Quantity(w1).to_value(u.AA),
                    u.Quantity(w0).to_value(u.AA),
                )
                assert np.isclose(r1, r0)

        import astropy.io.fits

        with astropy.io.fits.open(path) as hdul:
            header = hdul[2].header
            assert header["NMEMBER"] == 2
            # the velocity axis is measured against the first member
            assert np.isclose(
                (header["RESTWAV"] * u.m).to_value(u.AA),
                609.793,
            )
            assert np.isclose(header["MEMRAT2"], 0.52)

    def test_to_fits_split(self, tmp_path):
        """One file per line, each readable on its own and reassemblable."""
        import astropy.io.fits

        a = _level_4()
        directory = tmp_path / "split"
        a.to_fits(directory, split=True)

        files = sorted(directory.glob("*.fits"))
        assert len(files) == a.num_line

        # each file must stand alone as a single-line product
        for file in files:
            with astropy.io.fits.open(file) as hdul:
                index = hdul[0].header["LINEIDX"]
                assert hdul[0].header["NLINEALL"] == a.num_line
            one = esis.data.Level_4.from_fits(file)
            assert one.num_line == 1
            assert one.label(0) == a.label(index)
            assert np.allclose(
                one.intensity[{one.axis_line: 0}].ndarray.value,
                a.intensity[{a.axis_line: index}].ndarray.value,
                rtol=1e-6,
            )
            assert np.allclose(
                one.velocity.ndarray.to_value(u.km / u.s),
                a.velocity.ndarray.to_value(u.km / u.s),
            )

        # and the directory must reassemble into the whole product
        b = esis.data.Level_4.from_fits(directory)
        assert b.num_line == a.num_line
        assert b.label_line == [a.label(i) for i in range(a.num_line)]
        for i in range(a.num_line):
            assert np.allclose(
                b.outputs[b.window(i)].ndarray.value,
                a.outputs[a.window(i)].ndarray.value,
                rtol=1e-6,
            )
        assert np.allclose(
            b.intensity.ndarray.value,
            a.intensity.ndarray.value,
            rtol=1e-6,
        )

    def test_drift(self):
        """A synthetic shift must be recovered with the right sign."""
        import dataclasses

        a = _level_4()
        pitch = (
            a.inputs.position.x.ndarray.to_value(u.arcsec)[1]
            - a.inputs.position.x.ndarray.to_value(u.arcsec)[0]
        )

        # slide frame 0 by a known whole number of cells, leave the rest
        outputs = np.asarray(a.outputs.ndarray.value).copy()
        axes = a.outputs.axes
        index_time = axes.index(a.axis_time)
        index_x = axes.index(a.axis_x)
        shifted = np.moveaxis(outputs, (index_time, index_x), (0, 1))
        shifted[0] = np.roll(shifted[0], 2, axis=0)
        outputs = np.moveaxis(shifted, (0, 1), (index_time, index_x))
        b = dataclasses.replace(
            a,
            outputs=na.ScalarArray(outputs * na.unit(a.outputs), axes=axes),
        )

        drift = b.drift(index_time_reference=1)
        offset = drift.to_value(u.arcsec) / pitch

        assert np.isclose(offset[0, 0], 2, atol=0.1)
        assert np.isclose(offset[0, 1], 0, atol=0.1)
        assert np.allclose(offset[1], 0, atol=0.1)

    def test_animate_drift_corrected(self):
        """Passing a drift must undo it rather than double it."""
        a = _level_4()
        drift = a.drift()
        assert drift.shape == (a.shape[a.axis_time], 2)
        result = a.animate_doppler(index_line=0, drift=drift)
        result._fig.canvas.draw()
        plt.close(result._fig)

    def test_coregister(self):
        """Coregistration must remove the drift and record what it removed."""
        import dataclasses

        a = _level_4()
        pitch = (
            a.inputs.position.x.ndarray.to_value(u.arcsec)[1]
            - a.inputs.position.x.ndarray.to_value(u.arcsec)[0]
        )

        outputs = np.asarray(a.outputs.ndarray.value).copy()
        axes = a.outputs.axes
        shifted = np.moveaxis(
            outputs, (axes.index(a.axis_time), axes.index(a.axis_x)), (0, 1)
        )
        shifted[0] = np.roll(shifted[0], 2, axis=0)
        outputs = np.moveaxis(
            shifted, (0, 1), (axes.index(a.axis_time), axes.index(a.axis_x))
        )
        b = dataclasses.replace(
            a,
            outputs=na.ScalarArray(outputs * na.unit(a.outputs), axes=axes),
        )

        c = b.coregister(index_time_reference=1)

        assert c.drift_applied is not None
        assert np.isclose(c.drift_applied.to_value(u.arcsec)[0, 0] / pitch, 2, atol=0.1)
        # the residual drift of the coregistered product must be ~zero
        residual = c.drift(index_time_reference=1).to_value(u.arcsec) / pitch
        assert np.allclose(residual, 0, atol=0.15)
        assert np.all(c.outputs.ndarray.value >= 0)

        with pytest.raises(ValueError):
            c.coregister()

    def test_to_fits_coregistered(self, tmp_path):
        """The applied drift must survive a round trip."""
        import astropy.io.fits

        a = _level_4().coregister()
        path = tmp_path / "level_4_sky.fits"
        a.to_fits(path)

        with astropy.io.fits.open(path) as hdul:
            assert hdul[0].header["COREGIST"]

        b = esis.data.Level_4.from_fits(path)
        assert b.drift_applied is not None
        assert np.allclose(
            b.drift_applied.to_value(u.arcsec),
            a.drift_applied.to_value(u.arcsec),
        )
