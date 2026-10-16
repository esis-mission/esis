import dataclasses
import pathlib
import pytest
import numpy as np
import astropy.units as u
import astropy.table
import named_arrays as na
import esis


def _is_uncertain(a, seen: None | set[int] = None) -> bool:
    """Whether any part of a (possibly nested) object is an uncertain array."""
    if seen is None:
        seen = set()
    if id(a) in seen:
        return False
    seen.add(id(a))
    if isinstance(a, na.AbstractUncertainScalarArray):
        return True
    if isinstance(a, dict):
        return any(_is_uncertain(v, seen) for v in a.values())
    if isinstance(a, (list, tuple)):
        return any(_is_uncertain(v, seen) for v in a)
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return any(
            _is_uncertain(getattr(a, f.name), seen) for f in dataclasses.fields(a)
        )
    return False


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_full(num_distribution: int):
    result = esis.flights.f1.optics.design_full(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
    assert _is_uncertain(result) == (num_distribution != 0)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design(num_distribution: int):
    result = esis.flights.f1.optics.design(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
    assert _is_uncertain(result) == (num_distribution != 0)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_single(num_distribution: int):
    result = esis.flights.f1.optics.design_single(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
    assert _is_uncertain(result) == (num_distribution != 0)


def test_design_distance_grating() -> None:
    """
    Check the arms of the grating against the prescription.

    The design centers the field stop on the axis and places the grating and
    the sensor of a channel at the same azimuth, each at its own distance from
    the axis, so each arm is the hypotenuse of the distance along the axis and
    the distance across it.
    """
    design = esis.flights.f1.optics.design_single(num_distribution=0)
    field_stop = design.field_stop
    grating = design.grating
    sensor = design.camera.sensor

    expected_input = np.hypot(
        grating.translation.z - field_stop.translation.z,
        grating.distance_radial,
    )
    expected_output = np.hypot(
        sensor.translation.z - grating.translation.z,
        sensor.distance_radial - grating.distance_radial,
    )

    assert np.isclose(design.distance_grating_input, expected_input)
    assert np.isclose(design.distance_grating_output, expected_output)


def test_design_magnification() -> None:
    """
    Check the magnification against the image of the field stop.

    The field stop sits at the focus of the primary mirror, so every ray
    through a point on its edge lands at the same point of the detector, and
    the outline of the stop traced through the center of the pupil is the
    outline of its image. Across the dispersion, along the :math:`y` axis of
    the detector, the image is as tall as the stop is wide from edge to edge,
    times the magnification.
    """
    design = esis.flights.f1.optics.design_single(num_distribution=0)
    wavelength = esis.flights.f1.spectrum.O_V.wavelength
    system = design.system

    # The vertices of the outline are along an axis named the same as those
    # of every polygonal aperture they are traced through, which would
    # broadcast against each other, so it is renamed.
    outline = system.field_stop_polygon(wavelength).vertices.xy
    outline = outline.combine_axes(axes=("vertex",), axis_new="outline")

    rays = system.rayfunction(
        wavelength=wavelength,
        field=outline,
        pupil=na.Cartesian2dVectorArray(0, 0),
        normalized_field=False,
        efficiency=False,
    )
    y = rays.outputs.position.y
    height = y.max() - y.min()

    expected = design.magnification * design.field_stop.width_clear
    assert np.isclose(height, expected, rtol=0.005)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_as_built_unfocused(num_distribution: int):
    result = esis.flights.f1.optics.as_built_unfocused(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
    assert _is_uncertain(result) == (num_distribution != 0)


def test_as_built_unfocused_camera() -> None:
    """
    Check the measured properties of each tap of each camera.

    The gain, read noise, and dark current were measured for each tap of each
    camera, and the read noise reaches the surface as one value for each
    camera.
    """
    camera = esis.flights.f1.optics.as_built_unfocused(num_distribution=0).camera
    shape = {"channel": 4, camera.axis_tap_y: 2, camera.axis_tap_x: 2}

    assert camera.gain.shape == shape
    assert camera.sensor.readout_noise.shape == shape
    assert camera.dark_current.shape == shape

    # the range the verification decks give for every tap of every camera
    assert camera.gain.min() == 2.50 * u.electron / u.DN
    assert camera.gain.max() == 2.63 * u.electron / u.DN

    assert camera.surface.read_noise.shape == {"channel": 4}


def test_design_focus_grating():
    """The design is already focused, so focusing it should move nothing."""
    design = esis.flights.f1.optics.design(num_distribution=0)
    result = design.focus_grating(
        wavelength=esis.flights.f1.spectrum.O_V.wavelength,
    )
    dz = result.grating.translation.z - design.grating.translation.z
    assert np.all(np.abs(dz) < 10 * u.um)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_as_built_focused(num_distribution: int):
    result = esis.flights.f1.optics._as_built_focused(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)

    as_built = esis.flights.f1.optics.as_built_unfocused(
        num_distribution=num_distribution,
    )
    dz = na.nominal(result.grating.translation.z) - na.nominal(
        as_built.grating.translation.z
    )
    # the measured radii are shorter than the design radius, so every
    # grating moves toward the field stop by a fraction of a millimeter
    assert np.all(dz > 0.3 * u.mm)
    assert np.all(dz < 0.9 * u.mm)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_position_image(num_distribution: int):
    """The design puts the O V line where the sensor says it does."""
    design = esis.flights.f1.optics.design(num_distribution=num_distribution)

    position = design.position_line(esis.flights.f1.spectrum.O_V.wavelength)
    error = na.nominal(position - design.camera.sensor.position_image)

    # within a tenth of a pixel, which is what keeps the recorded value
    # honest without pinning it to a particular raytrace
    assert np.all(np.abs(error.x) < design.camera.sensor.width_pixel / 10)
    assert np.all(np.abs(error.y) < design.camera.sensor.width_pixel / 10)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_as_built(num_distribution: int):
    result = esis.flights.f1.optics.as_built(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)

    sensor = result.camera.sensor
    wavelength = esis.flights.f1.spectrum.O_V.wavelength

    # the line is put where the sensor says it should be
    error = na.nominal(result.position_line(wavelength) - sensor.position_image)
    assert np.all(np.abs(error.x) < sensor.width_pixel / 100)

    # which the model it was aligned from is eight pixels away from
    as_built = esis.flights.f1.optics.as_built_unfocused(
        num_distribution=num_distribution
    )
    error_built = na.nominal(as_built.position_line(wavelength) - sensor.position_image)
    assert np.all(np.abs(error_built.x) > 5 * sensor.width_pixel)

    # Focusing alone lands much closer than that, since moving the grating to
    # correct the focus also moves the image, and the two nearly cancel. It
    # is not close enough: a couple of pixels is tens of kilometers per
    # second of apparent Doppler shift.
    focused = esis.flights.f1.optics._as_built_focused(
        num_distribution=num_distribution,
    )
    error_focused = na.nominal(
        focused.position_line(wavelength) - sensor.position_image
    )
    assert np.all(np.abs(error_focused.x) > np.abs(error.x))

    # and it is still focused: the gratings moved along z as they did for
    # `_as_built_focused`, and only rotated by seconds of arc
    dz = na.nominal(result.grating.translation.z - as_built.grating.translation.z)
    assert np.all(dz > 0.3 * u.mm)
    assert np.all(dz < 0.9 * u.mm)

    dyaw = na.nominal(result.grating.yaw - as_built.grating.yaw)
    assert np.all(np.abs(dyaw) < 1 * u.arcmin)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_as_built_radius_grating(num_distribution: int):
    """The gratings which were built are concave, as the designed ones are."""
    # the radius does not depend on where the alignment puts the grating, so
    # this asks the model which has not paid to be placed
    result = esis.flights.f1.optics.as_built_unfocused(
        num_distribution=num_distribution,
    )
    design = esis.flights.f1.optics.design(
        num_distribution=num_distribution,
    )

    radius = result.grating.sag.radius

    # a concave surface has a negative radius of curvature in the sag
    # convention, and a grating stored with a positive one is convex, which
    # sends the light somewhere else entirely
    assert np.all(radius < 0)

    # The gratings which were built are the gratings which were designed, to
    # within a millimeter. Compared against the radius the design asked for
    # rather than against a sample of its tolerance, since the samples are
    # gratings which might have been made and these are the ones which were.
    radius_design = na.nominal(design.grating.sag.radius)
    assert np.all(np.abs(radius - radius_design) < 1 * u.mm)


@pytest.mark.parametrize("optic", ["primary_mirror", "grating"])
@pytest.mark.parametrize("name", ["slope_error", "roughness", "microroughness"])
def test_as_built_figure_and_finish(optic: str, name: str) -> None:
    """
    The figure and finish of each optic as built meet its specifications.

    The measurements are those of the primary mirror and of each grating, and
    the instrument which flew carries them through its alignment. A grating
    is measured on its own, so the measurements of the gratings are along the
    axis of the channels, beside the serial number of each.
    """
    design = esis.flights.f1.optics.design(num_distribution=0)
    as_built = esis.flights.f1.optics.as_built(num_distribution=0)

    specified = getattr(getattr(design, optic), name)
    measured = getattr(getattr(as_built, optic), name)
    assert specified is not None
    assert measured is not None
    assert np.all(measured <= specified)

    if optic == "grating":
        assert na.shape(measured) == na.shape(as_built.grating.serial_number)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_distortion_fit(num_distribution: int):
    result = esis.flights.f1.optics.distortion_fit(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)


def test_distortion_fit_axis_time():
    result = esis.flights.f1.optics.distortion_fit(num_distribution=0, axis_time="time")
    assert na.shape(result.pitch) == dict(channel=4, time=30)
    reference = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    delta = result.yaw - reference.yaw
    # the yaw drifted monotonically through the flight, changing sign at the
    # reference frame
    assert np.all(delta[dict(time=0, channel=0)] > 0 * u.arcsec)
    assert np.all(delta[dict(time=~0, channel=0)] < 0 * u.arcsec)
    assert np.all(np.abs(delta[dict(time=15)]) < 0.01 * u.arcsec)
    # the pointing is the payload's, common to the channels
    assert np.all(np.abs(delta - delta.mean("channel")) < 1e-6 * u.arcsec)
    # each channel views the Sun through its own sector of the primary, and
    # the focus of each sector has its own history: zero at the reference
    # frame, tens of microns apart at the first frame, and channel 1's
    # sector moves by more than 30 um over the flight
    z = (
        result.primary_mirror.translation.z - reference.primary_mirror.translation.z
    ).to(u.um)
    assert na.shape(z) == dict(time=30, channel=4)
    assert np.all(np.abs(z[dict(time=15)]) < 1e-6 * u.um)
    first = z[dict(time=0)]
    assert first.max() - first.min() > 15 * u.um
    assert np.abs(z[dict(time=~0, channel=1)] - z[dict(time=0, channel=1)]) > 30 * u.um


def test_distortion_fit_tables_record_environment():
    # every committed table names the environment that produced it
    directory = pathlib.Path(esis.flights.f1.optics._instruments._instruments.__file__)
    paths = sorted((directory.parent / "_data").glob("*.ecsv"))
    assert paths
    for path in paths:
        table = astropy.table.QTable.read(path, format="ascii.ecsv")
        environment = table.meta.get("environment")
        assert environment is not None, path.name
        assert "optika" in environment["checkouts"], path.name
        assert "optika" in environment["packages"], path.name


def test_distortion_fit_sensor_terms():
    # the loaded reference carries the sensor placement (zero for a file
    # written before it existed) and the primary displacement is measured
    # from the nominal focal length
    result = esis.flights.f1.optics.distortion_fit(num_distribution=0)
    parameters = esis.optics.DistortionParameters.from_instrument(result)
    assert na.shape(parameters) == dict(channel=4)
    z = na.value(parameters.z_sensor)
    assert np.all(np.isfinite(np.asarray(getattr(z, "ndarray", z))))


def test_distortion_fit_bounds():
    instrument = esis.flights.f1.optics.design(num_distribution=0)
    parameters = esis.optics.DistortionParameters.from_instrument(instrument)
    lower, upper = esis.flights.f1.optics.distortion_fit_bounds(parameters)
    lb, ub = na.pack(lower).ndarray, na.pack(upper).ndarray
    x = na.pack(parameters).ndarray
    assert lb.shape == ub.shape == x.shape
    # every parameter starts within its box, and no box is degenerate
    assert np.all(lb <= x)
    assert np.all(x <= ub)
    assert np.all(ub > lb)
    # the sensor terms are bounded about wherever the sensor is
    assert np.all(upper.yaw_sensor - parameters.yaw_sensor == 3 * u.deg)
    assert np.all(parameters.yaw_sensor - lower.yaw_sensor == 3 * u.deg)
    assert np.all(upper.z_sensor == parameters.z_sensor + 10 * u.mm)


def test_distortion_fit_windows():
    """
    Check that the committed reference puts the windows where it says.

    The reference table records the centre of every window as the fit
    left it; a change in the optics library that moves the image, such as
    the roll bug of optika 2.7, shows up here as a shift of every window.
    """
    import astropy.table
    from esis.flights.f1.optics._instruments import _instruments

    table = astropy.table.QTable.read(
        _instruments._directory_data / "distortion_reference.ecsv",
        format="ascii.ecsv",
    )
    if "windows" not in table.meta:
        pytest.skip("the committed table predates the window record")
    from esis.flights.f1.optics._fits import _fits

    # the fit traces the idealized as-built model, whose uncertain ruling
    # coefficients are stripped; the committed parameters go back onto it
    model = esis.optics.DistortionParameters.from_file(
        _instruments._directory_data / "distortion_reference.ecsv"
    ).to_instrument(_fits._base(None))
    for c, centers in enumerate(table.meta["windows"]):
        channel = model[dict(channel=c)]
        linear = channel.system.linearize(
            wavelength=channel.wavelength, degree=2, field_stop=True
        )
        for i, (x, y) in enumerate(centers):
            footprint = linear.footprint(channel.wavelength[dict(wavelength=i)])
            cx, cy = _fits._center(footprint)
            assert abs(cx - x) < 1
            assert abs(cy - y) < 1


def test_distortion_fit_channel_offsets():
    reference = esis.flights.f1.optics.distortion_fit(
        num_distribution=0, axis_time="time"
    )
    result = esis.flights.f1.optics.distortion_fit(
        num_distribution=0,
        axis_time="time",
        channel_offsets=True,
    )
    for name in ("pitch", "yaw"):
        delta = getattr(result, name) - getattr(reference, name)
        assert na.shape(delta) == dict(channel=4, time=30)
        # zero at the reference frame and about the mean of the channels,
        # so the payload's pointing is left where the scene put it
        assert np.all(np.abs(delta[dict(time=15)]) < 1e-6 * u.arcsec)
        # (the centering is done on the sky, so in angle it holds to a
        # ten-thousandth of a pixel)
        assert np.all(np.abs(delta.mean("channel")) < 1e-4 * u.arcsec)
        # a few hundredths of a pixel, well under a pixel of 0.76 arcsec
        assert np.all(np.abs(delta) < 0.5 * u.arcsec)
    assert np.any(np.abs(result.yaw - reference.yaw) > 0.01 * u.arcsec)
    # the optics are untouched
    z = result.primary_mirror.translation.z - reference.primary_mirror.translation.z
    assert np.all(np.abs(z) < 1e-9 * u.um)
