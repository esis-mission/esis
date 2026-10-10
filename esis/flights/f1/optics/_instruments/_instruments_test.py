import dataclasses
import pytest
import numpy as np
import astropy.units as u
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


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_as_built_unfocused(num_distribution: int):
    result = esis.flights.f1.optics.as_built_unfocused(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
    assert _is_uncertain(result) == (num_distribution != 0)


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


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_distortion_fit(num_distribution: int):
    result = esis.flights.f1.optics.distortion_fit(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractInstrument)
