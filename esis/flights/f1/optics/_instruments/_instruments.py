import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import esis
from esis.flights.f1.spectrum import He_I, Mg_X, O_V
from .. import primaries
from .. import gratings
from .. import filters
from .._uncertainty import spawn_seeds, uniform

__all__ = [
    "design_full",
    "design",
    "design_single",
    "as_built_unfocused",
    "as_built",
    "distortion_fit",
]


def design_full(
    grid: None | optika.vectors.ObjectVectorArray = None,
    axis_channel: str = "channel",
    num_distribution: int = 11,
    seed: None | int = None,
) -> esis.optics.Instrument:
    """
    Load the entire optical design including the inactive channels.

    This instance includes all six channels instead of the four active channels
    included in :func:`design`.

    Parameters
    ----------
    grid
        sampling of wavelength, field, and pupil positions that will be used to
        characterize the optical system.
    axis_channel
        The name of the logical axis corresponding to changing camera channel.
    num_distribution
        number of Monte Carlo samples to draw when computing uncertainties
    seed
        seed for the random number generator which draws the Monte Carlo
        samples, or :obj:`None` to draw different samples every time
    """
    num_folds = 8
    num_channels = 6

    seeds = spawn_seeds(seed)

    name_channel = na.arange(0, num_channels, axis=axis_channel)

    angle_per_channel = (360 * u.deg) / num_folds
    cos_per_channel = np.cos(angle_per_channel / 2)
    angle_channel_offset = -angle_per_channel / 2
    angle_channel = na.linspace(
        start=0 * u.deg,
        stop=num_channels * angle_per_channel,
        axis=axis_channel,
        num=num_channels,
        endpoint=False,
    )
    angle_channel = angle_channel + angle_channel_offset

    # dashstyle = (0, (1, 3))
    # dashstyle_channels = na.ScalarArray(
    #     ndarray=np.array(
    #         object=[dashstyle, "solid", "solid", "solid", "solid", dashstyle],
    #         dtype=object,
    #     ),
    #     axes="channel",
    # )
    # alpha_channels = na.ScalarArray(np.array([0, 1, 1, 1, 1, 0]), axes="channel")

    radius_primary_clear = 77.9 * u.mm
    primary = esis.optics.PrimaryMirror(
        sag=optika.sags.ParabolicSag(
            focal_length=-1000 * u.mm,
            parameters_slope_error=optika.metrology.SlopeErrorParameters(
                step_size=4 * u.mm,
                kernel_size=2 * u.mm,
            ),
            parameters_roughness=optika.metrology.RoughnessParameters(
                period_min=0.06 * u.mm,
                period_max=6 * u.mm,
            ),
            parameters_microroughness=optika.metrology.RoughnessParameters(
                period_min=1.6 * u.um,
                period_max=70 * u.um,
            ),
        ),
        num_folds=8,
        width_clear=2 * radius_primary_clear * cos_per_channel,
        width_border=(83.7 * u.mm - radius_primary_clear) * cos_per_channel,
        material=primaries.materials.multilayer_design(),
        translation=na.Cartesian3dVectorArray(
            x=uniform(
                nominal=0 * u.mm,
                width=1 * u.mm,
                num_distribution=num_distribution,
                seed=next(seeds),
            ),
            y=uniform(
                nominal=0 * u.mm,
                width=1 * u.mm,
                num_distribution=num_distribution,
                seed=next(seeds),
            ),
            z=0 * u.mm,
        ),
    )

    front_aperture = esis.optics.FrontAperture(
        translation=na.Cartesian3dVectorArray(
            x=0 * u.mm,
            y=0 * u.mm,
            z=primary.sag.focal_length - 500 * u.mm,
        ),
    )

    point_tuffet_1 = na.Cartesian2dVectorArray(2.54, 37.1707) * u.mm
    point_tuffet_2 = na.Cartesian2dVectorArray(24.4876, 28.0797) * u.mm
    difference_tuffet = point_tuffet_2 - point_tuffet_1
    slope_tuffet = difference_tuffet.y / difference_tuffet.x
    radius_tuffet = point_tuffet_1.y - slope_tuffet * point_tuffet_1.x
    central_obscuration = esis.optics.CentralObscuration(
        num_folds=num_folds,
        halfwidth=radius_tuffet * cos_per_channel,
        remove_last_vertex=True,
        translation=na.Cartesian3dVectorArray(z=-1404.270) * u.mm,
    )

    field_stop = esis.optics.FieldStop(
        num_folds=num_folds,
        radius_clear=1.82 * u.mm,
        radius_mechanical=2.81 * u.mm,
        translation=na.Cartesian3dVectorArray(
            x=primary.translation.x.copy(),
            y=primary.translation.y.copy(),
            z=primary.sag.focal_length,
        ),
    )

    radius_grating = 597.830 * u.mm
    error_radius_grating = 0.4 * u.percent
    width_grating_border = 2 * u.mm
    width_grating_border_inner = 4.58 * u.mm
    var_grating_z_single = np.square(2.5e-5 * u.m)
    var_grating_z_systematic = np.square(5e-6 * u.m)
    var_grating_z = var_grating_z_single / 3 + var_grating_z_systematic
    error_grating_z = np.sqrt(var_grating_z)
    grating = esis.optics.Grating(
        angle_input=1.301 * u.deg,
        angle_output=8.057 * u.deg,
        sag=optika.sags.SphericalSag(
            radius=uniform(
                nominal=-radius_grating,
                width=radius_grating * error_radius_grating,
                num_distribution=num_distribution,
                seed=next(seeds),
            ),
            parameters_slope_error=optika.metrology.SlopeErrorParameters(
                step_size=2 * u.mm,
                kernel_size=1 * u.mm,
            ),
            parameters_roughness=optika.metrology.RoughnessParameters(
                period_min=0.024 * u.mm,
                period_max=2.4 * u.mm,
            ),
            parameters_microroughness=optika.metrology.RoughnessParameters(
                period_min=0.02 * u.um,
                period_max=2 * u.um,
            ),
        ),
        material=gratings.materials.multilayer_design(),
        rulings=gratings.rulings.ruling_design(
            num_distribution=num_distribution,
            seed=next(seeds),
        ),
        num_folds=num_folds,
        halfwidth_inner=13.02 * u.mm - width_grating_border_inner,
        halfwidth_outer=10.49 * u.mm - width_grating_border,
        width_border=width_grating_border,
        width_border_inner=width_grating_border_inner,
        clearance=1.25 * u.mm,
        distance_radial=2.074999998438000e1 * u.mm,
        azimuth=angle_channel.copy(),
        translation=na.Cartesian3dVectorArray(
            x=uniform(
                nominal=0 * u.mm,
                width=1 * u.mm,
                num_distribution=num_distribution,
                seed=next(seeds),
            ),
            y=uniform(
                nominal=0 * u.mm,
                width=1 * u.mm,
                num_distribution=num_distribution,
                seed=next(seeds),
            ),
            z=uniform(
                nominal=primary.sag.focal_length - 374.7 * u.mm,
                width=error_grating_z,
                num_distribution=num_distribution,
                seed=next(seeds),
            ),
        ),
        yaw=-4.469567242792327 * u.deg,
        roll=uniform(
            nominal=0 * u.deg,
            width=1.3e-2 * u.rad,
            num_distribution=num_distribution,
            seed=next(seeds),
        ),
    )

    filter = esis.optics.Filter(
        material=filters.materials.thin_film_design(),
        radius_clear=15 * u.mm,
        width_border=0 * u.mm,
        distance_radial=95.9 * u.mm,
        azimuth=angle_channel.copy(),
        translation=na.Cartesian3dVectorArray(
            x=0 * u.mm,
            y=0 * u.mm,
            # placed relative to where the grating is designed to be
            z=na.nominal(grating.translation.z) + 1.301661998854058 * u.m,
        ),
        yaw=-3.45 * u.deg,
        roll=45 * u.deg,
    )

    # The sensors were placed at the focus found by experiment and shimmed
    # there, so where each one sits along the optic axis is uncertain by the
    # error of the experiment, the step between the thicknesses of the shims,
    # and the defocus of the edges of the sensor caused by its tip and tilt.
    var_sensor_z_experiment = np.square(0.5 * u.mm)
    var_sensor_z_shim = np.square(50 * u.um)
    var_sensor_z_edge = np.square(0.1 * u.mm)
    var_sensor_z = var_sensor_z_experiment + var_sensor_z_shim + var_sensor_z_edge
    error_sensor_z = np.sqrt(var_sensor_z)
    sensor = esis.optics.Sensor(
        # The physical mask on the ESIS-I detectors was undersized, leaving
        # readout-buffer rows exposed to light.  Science data extends into
        # those rows, so the active area is 2 x 1040 rows, matching the
        # 1040-row halves of the Level-1 frames.
        num_pixel_y=2 * 1040,
        distance_radial=108 * u.mm,
        azimuth=angle_channel.copy(),
        translation=na.Cartesian3dVectorArray(
            x=uniform(
                nominal=0 * u.mm,
                width=1 * u.mm,
                num_distribution=num_distribution,
                seed=next(seeds),
            ),
            y=uniform(
                nominal=0 * u.mm,
                width=1 * u.mm,
                num_distribution=num_distribution,
                seed=next(seeds),
            ),
            z=uniform(
                nominal=filter.translation.z + 200 * u.mm,
                width=error_sensor_z,
                num_distribution=num_distribution,
                seed=next(seeds),
            ),
        ),
        yaw=-12.252 * u.deg,
        # Where the center of the field of view lands at the O V line, which
        # is what the gratings were rotated about y to achieve. Traced from
        # this model rather than asserted: `test_position_image` checks that
        # the design still puts the line here, so the number cannot drift
        # away from the prescription it describes.
        #
        # The Zemax design this was ported from carried the same quantity as
        # a CENY operand, targeting 7.2091 mm. It used 629.7 A for O V where
        # this model uses 629.732 A, and traced at 629.7 A this model puts
        # the line within 1.6 um of that target.
        position_image=na.Cartesian2dVectorArray(
            x=7.2206 * u.mm,
            y=0 * u.mm,
        ),
        material=optika.sensors.materials.e2v_ccd97(
            temperature=-55 * u.deg_C,
        ).replace(
            num_interpolation=esis.optics.num_interpolation,
        ),
    )

    camera = esis.optics.Camera(
        sensor=sensor,
        gain=2.5 * u.electron / u.DN,
        channel=name_channel,
        channel_trigger=1,
        timedelta_sync=1 * u.ms,
    )

    if grid is None:
        grid = optika.vectors.ObjectVectorArray(
            wavelength=629.77 * u.AA,
            field=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=11,
                centers=True,
            ),
            pupil=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
                num=11,
                centers=True,
            ),
        )

    return esis.optics.Instrument(
        name="ESIS 1 final design (all channels)",
        axis_channel=axis_channel,
        front_aperture=front_aperture,
        central_obscuration=central_obscuration,
        primary_mirror=primary,
        field_stop=field_stop,
        grating=grating,
        filter=filter,
        camera=camera,
        wavelength=grid.wavelength,
        field=grid.field,
        pupil=grid.pupil,
    )


def design(
    grid: None | optika.vectors.ObjectVectorArray = None,
    axis_channel: str = "channel",
    num_distribution: int = 11,
    seed: None | int = None,
) -> esis.optics.Instrument:
    """
    Load the final optical design prepared by Charles Kankelborg and Hans Courrier.

    Parameters
    ----------
    grid
        sampling of wavelength, field, and pupil positions that will be used to
        characterize the optical system.
    axis_channel
        The name of the logical axis corresponding to changing camera channel.
    num_distribution
        number of Monte Carlo samples to draw when computing uncertainties
    seed
        seed for the random number generator which draws the Monte Carlo
        samples, or :obj:`None` to draw different samples every time
    """
    result = design_full(
        grid=grid,
        axis_channel=axis_channel,
        num_distribution=num_distribution,
        seed=seed,
    )

    slice_active = {axis_channel: slice(1, 5)}

    result.grating.azimuth = result.grating.azimuth[slice_active]
    result.filter.azimuth = result.filter.azimuth[slice_active]

    result.camera.channel = result.camera.channel[slice_active]
    result.camera.sensor.azimuth = result.camera.sensor.azimuth[slice_active]

    return result


def design_single(
    grid: None | optika.vectors.ObjectVectorArray = None,
    axis_channel: str = "channel",
    num_distribution: int = 11,
    seed: None | int = None,
) -> esis.optics.Instrument:
    """
    Load only a single channel of the optical design.

    Since the system is rotationally symmetric, sometimes it's nice to model
    only one channel

    Parameters
    ----------
    grid
        sampling of wavelength, field, and pupil positions that will be used to
        characterize the optical system.
    axis_channel
        The name of the logical axis corresponding to changing camera channel.
    num_distribution
        number of Monte Carlo samples to draw when computing uncertainties
    seed
        seed for the random number generator which draws the Monte Carlo
        samples, or :obj:`None` to draw different samples every time
    """
    result = design(
        grid=grid,
        axis_channel=axis_channel,
        num_distribution=num_distribution,
        seed=seed,
    )

    index = {axis_channel: 0}

    result.grating.azimuth = result.grating.azimuth[index]
    result.filter.azimuth = result.filter.azimuth[index]

    result.camera.channel = result.camera.channel[index]
    result.camera.sensor.azimuth = result.camera.sensor.azimuth[index]

    result.roll = -result.grating.azimuth

    return result


def as_built_unfocused(
    grid: None | optika.vectors.ObjectVectorArray = None,
    axis_channel: str = "channel",
    num_distribution: int = 11,
    seed: None | int = None,
) -> esis.optics.Instrument:
    """
    Load the as-built optical model before it has been focused or pointed.

    Based on :func:`design`, but includes efficiency and figure measurements of the
    primary mirror and gratings, as well as the gain, read noise, and dark current
    measured for each camera.

    The gratings carry their measured radii of curvature but sit where the
    design put them, which is not where the instrument that flew carried
    them: it was focused and aligned with the gratings it actually had.
    :func:`as_built` is that instrument, and is the model to compare against
    an image.

    This one is for everything which wants the measurements without the
    solve that places them: the properties of the camera and the sensor, the
    coatings, the rulings, and the question of how far the alignment had to
    move the gratings, which is measured from here.

    Parameters
    ----------
    grid
        sampling of wavelength, field, and pupil positions that will be used to
        characterize the optical system.
    axis_channel
        The name of the logical axis corresponding to changing camera channel.
    num_distribution
        number of Monte Carlo samples to draw when computing uncertainties
    seed
        seed for the random number generator which draws the Monte Carlo
        samples, or :obj:`None` to draw different samples every time

    Examples
    --------
    Load the as-built optical model and print its parameters.

    .. jupyter-execute::

        import esis

        esis.flights.f1.optics.as_built()
    """
    seeds = spawn_seeds(seed)

    result = design(
        grid=grid,
        axis_channel=axis_channel,
        num_distribution=num_distribution,
        seed=next(seeds),
    )

    result.primary_mirror.material = primaries.materials.multilayer_fit()

    result.grating.serial_number = na.stack(
        arrays=[
            "89025",
            "89024",
            "89026",
            "89027",
        ],
        axis=axis_channel,
    )
    result.grating.manufacturing_number = na.stack(
        arrays=[
            "UBO-16-024",
            "UBO-16-017",
            "UBO-16-019",
            "UBO-16-014",
        ],
        axis=axis_channel,
    )

    radius_014 = [597.170, 597.210, 597.195] * u.mm
    radius_017 = [597.065, 597.045, 597.050] * u.mm
    radius_019 = [597.055, 597.045, 597.030] * u.mm
    radius_024 = [596.890, 596.870, 596.880] * u.mm
    # The measurements report the magnitude of the radius of curvature. In the
    # sag convention it is negative for a concave surface, which these are,
    # as the design radius of -597.83 mm records.
    result.grating.sag.radius = -na.stack(
        arrays=[
            radius_024.mean(),
            radius_017.mean(),
            radius_019.mean(),
            radius_014.mean(),
        ],
        axis=axis_channel,
    )

    result.grating.material = gratings.materials.multilayer_fit()

    result.grating.rulings = gratings.rulings.ruling_measurement(
        num_distribution=num_distribution,
        seed=next(seeds),
    )

    result.camera.sensor.serial_number = na.stack(
        arrays=[
            "SN6",
            "SN7",
            "SN9",
            "SN10",
        ],
        axis=axis_channel,
    )

    axis_tap_x = result.camera.axis_tap_x
    axis_tap_y = result.camera.axis_tap_y

    # The cameras were tested at MSFC by Laurel Rachmeler on 2017-07-06 and
    # 2017-07-12, and her verification decks for the two tests list their
    # properties by quadrant. Quadrants 1 and 4 are read out by the first row
    # of taps and quadrants 2 and 3 by the second, as
    # `msfc_ccd.TapData.amplifier` lays them out, so each channel below is
    # [[Q1, Q4], [Q2, Q3]].

    # Fitted to the K-alpha peak of an Fe-55 source, on 2017-07-06 for SN6
    # and SN7 (slide 18 of its verification deck) and on 2017-07-12 for SN9
    # and SN10 (slide 24). An earlier summary of the 2017-07-06 test gave the
    # gains of SN6 to SN7 and those of SN7 to SN6.
    result.camera.gain = na.ScalarArray(
        ndarray=[
            [
                [2.57, 2.53],
                [2.50, 2.52],
            ],
            [
                [2.55, 2.63],
                [2.58, 2.57],
            ],
            [
                [2.57, 2.59],
                [2.53, 2.52],
            ],
            [
                [2.60, 2.58],
                [2.60, 2.54],
            ],
        ]
        * u.electron
        / u.DN,
        axes=(axis_channel, axis_tap_y, axis_tap_x),
    )

    # The standard deviation of the pixels outside the image area, in the
    # first set of dark frames of the test on 2017-07-12, which the decks
    # plot but do not list. These are the values the analysis behind the
    # decks saved for that set, in `noise_params_20170712_SN*_quad*.sav`.
    readout_noise = na.ScalarArray(
        ndarray=[
            [
                [4.02, 3.85],
                [4.20, 4.24],
            ],
            [
                [4.08, 4.11],
                [4.20, 4.18],
            ],
            [
                [4.13, 4.27],
                [4.11, 4.12],
            ],
            [
                [3.90, 4.07],
                [3.87, 4.21],
            ],
        ]
        * u.DN,
        axes=(axis_channel, axis_tap_y, axis_tap_x),
    )
    result.camera.sensor.readout_noise = result.camera.gain * readout_noise

    # The slope of the dark level against the exposure time, in the same set
    # of dark frames, with the sensors held at about -50 C (slide 16 of the
    # verification deck of the test on 2017-07-12).
    dark_current = na.ScalarArray(
        ndarray=[
            [
                [2.37e-5, 2.41e-5],
                [2.04e-5, 1.95e-5],
            ],
            [
                [4.58e-5, 6.66e-5],
                [3.95e-5, 6.06e-5],
            ],
            [
                [2.29e-5, 2.59e-5],
                [1.98e-5, 2.15e-5],
            ],
            [
                [4.50e-5, 5.94e-5],
                [3.83e-5, 4.93e-5],
            ],
        ]
        * u.DN
        / u.ms,
        axes=(axis_channel, axis_tap_y, axis_tap_x),
    )
    result.camera.dark_current = (result.camera.gain * dark_current).to(
        u.electron / u.s
    )

    return result


def _moved_sensor(
    instrument: esis.optics.Instrument,
    translation: u.Quantity | na.AbstractCartesian3dVectorArray,
) -> esis.optics.Instrument:
    """
    Copy an instrument, putting its sensors somewhere else.

    Built rather than changed in place, like
    :meth:`esis.optics.abc.AbstractInstrument.moved_grating`, so that the
    raytrace of the copy is not one cached while the sensors were where they
    were.

    Parameters
    ----------
    instrument
        The instrument to copy.
    translation
        Where to put the sensors.
    """
    camera = instrument.camera
    return instrument.replace(
        camera=camera.replace(
            sensor=camera.sensor.replace(translation=translation),
        ),
    )


def _as_built_focused(
    grid: None | optika.vectors.ObjectVectorArray = None,
    axis_channel: str = "channel",
    num_distribution: int = 11,
    seed: None | int = None,
) -> esis.optics.Instrument:
    r"""
    Load the as-built optical model with the gratings moved to their best focus.

    :func:`as_built_unfocused` replaces the design radius of curvature of each grating
    with its measured value but leaves the grating where the design put it.
    The measured radii are 0.6 to 0.9 mm shorter than the design radius, so
    each grating images the field stop short of the sensor and the model is
    defocused by about two pixels RMS, whereas the flight instrument was
    focused with the gratings it actually carried.

    This model moves each grating along the optic axis to the position which
    minimizes the spot size of the :math:`\text{O\,V}\;630\,\AA` line, the
    brightest line in the passband, using
    :meth:`esis.optics.abc.AbstractInstrument.focus_grating`, which restores
    the focus of the design. The offsets come out to roughly
    :math:`+0.7`, :math:`+0.6`, :math:`+0.6`, and :math:`+0.5` mm toward the
    field stop for channels 0 through 3, in the same order as the errors in
    the measured radii. With ``num_distribution > 0`` every Monte Carlo
    sample of the model is focused independently, so the uncertainty in the
    other parameters (the placement of the gratings and the measured rulings,
    for example) carries through to a spread of about :math:`\pm 0.2` mm in
    the focus of each channel.

    The gratings are focused on the sensors where the design puts them, and
    the tolerance on where each sensor sits is added afterward, since along
    the optic axis that tolerance is the error of the focus itself, from the
    experiment which found it, the shims which set it, and the tip and tilt
    of the sensor.

    Parameters
    ----------
    grid
        sampling of wavelength, field, and pupil positions that will be used to
        characterize the optical system.
    axis_channel
        The name of the logical axis corresponding to changing camera channel.
    num_distribution
        number of Monte Carlo samples to draw when computing uncertainties
    seed
        seed for the random number generator which draws the Monte Carlo
        samples, or :obj:`None` to draw different samples every time

    Examples
    --------
    Load the focused as-built model and print the displacement of each
    grating from its design position.

    .. jupyter-execute::

        import esis

        instrument = esis.flights.f1.optics._as_built_focused(num_distribution=0)
        design = esis.flights.f1.optics.design(num_distribution=0)

        instrument.grating.translation.z - design.grating.translation.z
    """
    result = as_built_unfocused(
        grid=grid,
        axis_channel=axis_channel,
        num_distribution=num_distribution,
        seed=seed,
    )
    translation = result.camera.sensor.translation
    result = _moved_sensor(result, na.nominal(translation))
    result = result.focus_grating(wavelength=O_V.wavelength)
    return _moved_sensor(result, translation)


def as_built(
    grid: None | optika.vectors.ObjectVectorArray = None,
    axis_channel: str = "channel",
    num_distribution: int = 11,
    seed: None | int = None,
) -> esis.optics.Instrument:
    r"""
    Load the as-built optical model, focused and pointed at the sensor.

    The measured radii leave the as-built model imaging the O V line about
    eight pixels from where the design puts it.
    :func:`_as_built_focused` moves each grating along the optic axis until
    the spots are as small as the design's, which happens to carry the line
    most of the way back, since the same error in the radius causes both the
    defocus and the displacement. It stops a pixel or two short.

    The instrument which flew was aligned as well as focused, so this model
    rotates each grating about :math:`y` afterwards, by a few arcseconds,
    until the center of the field of view lands where
    :attr:`esis.optics.Sensor.position_image` says it should.

    Like the focus, the alignment is found for the sensors where the design
    puts them, and the tolerance on where each sensor sits is added
    afterward, since it is an error which remained after the instrument was
    focused and aligned.

    This is the model of the instrument that flew, and the one to use.
    :func:`as_built_unfocused` and :func:`_as_built_focused` are the steps on the way
    to it, kept for comparison rather than for use.

    Parameters
    ----------
    grid
        sampling of wavelength, field, and pupil positions that will be used to
        characterize the optical system.
    axis_channel
        The name of the logical axis corresponding to changing camera channel.
    num_distribution
        number of Monte Carlo samples to draw when computing uncertainties
    seed
        seed for the random number generator which draws the Monte Carlo
        samples, or :obj:`None` to draw different samples every time

    Examples
    --------
    Confirm the O V line lands where the sensor says it should.

    .. jupyter-execute::

        import astropy.units as u
        import esis

        instrument = esis.flights.f1.optics.as_built(num_distribution=0)

        error = instrument.position_line(
            esis.flights.f1.spectrum.O_V.wavelength,
        ) - instrument.camera.sensor.position_image

        error.length.to(u.um)
    """
    result = as_built_unfocused(
        grid=grid,
        axis_channel=axis_channel,
        num_distribution=num_distribution,
        seed=seed,
    )
    translation = result.camera.sensor.translation
    result = _moved_sensor(result, na.nominal(translation))
    result = result.align_grating(wavelength=O_V.wavelength)
    return _moved_sensor(result, translation)


def distortion_fit(
    grid: None | optika.vectors.ObjectVectorArray = None,
    axis_channel: str = "channel",
    num_distribution: int = 11,
) -> esis.optics.Instrument:
    """
    Apply the best-fit distortion parameters to the ESIS-I :func:`design`.

    The parameters are hard-coded from the best distortion fit of the ESIS-I
    flight data, optimized against the ``time=15`` frame of the 2019-09-30
    flight (:func:`esis.flights.f1.data.level_1`, with a start time of
    2019-09-30T18:08:41.642 UTC). The values are per-channel and were produced
    by the ``ESISI_distortion_optimization_20260213_151715`` run.

    Parameters
    ----------
    grid
        sampling of wavelength, field, and pupil positions that will be used to
        characterize the optical system.
    axis_channel
        The name of the logical axis corresponding to changing camera channel.
    num_distribution
        number of Monte Carlo samples to draw when computing uncertainties

    Examples
    --------
    Overplot the ray-traced detector footprint of each spectral line onto the
    Level-1 frame that the distortion fit was optimized against.
    Each line's footprint should land on its corresponding image of the
    field stop.

    .. jupyter-execute::

        import numpy as np
        import astropy.units as u
        import named_arrays as na
        import esis

        l1 = esis.flights.f1.data.level_1()[dict(time=15)]
        model = esis.flights.f1.optics.distortion_fit(num_distribution=0)

        rays = model.system.rayfunction_default.outputs
        position = rays.position.to(u.um).mean(axis=("pupil_x", "pupil_y"))
        position = position / model.camera.sensor.width_pixel * u.pixel

        fig, ax = na.plt.subplots(
            figsize=(8, 17),
            constrained_layout=True,
            axis_rows="channel",
            nrows=l1.shape["channel"],
            sharex=True,
            origin="upper",
        )
        fig.suptitle(
            "ESIS-I distortion fit vs. Level-1 data"
            " (2019-09-30 18:08:41 UTC)"
        )
        na.plt.set_xlabel("detector $x$ (pix)", ax=ax[dict(channel=~0)])
        na.plt.set_ylabel("detector $y$ (pix)", ax=ax)
        na.plt.set_aspect("equal", ax=ax)
        na.plt.pcolormesh(
            l1.inputs.pixel.x,
            l1.inputs.pixel.y,
            C=l1.outputs.value,
            ax=ax,
            vmax=np.percentile(l1.outputs.value, 99),
        )
        na.plt.text(
            x=0.5,
            y=1.01,
            s=l1.channel,
            transform=na.plt.transAxes(ax),
            ax=ax,
            ha="center",
            va="bottom",
        )
        spectral_lines = ["He I", "Mg X", "O V"]
        colors = ["red", "orange", "yellow"]
        for i in range(len(spectral_lines)):
            j = dict(wavelength=i)
            na.plt.scatter(
                position.x[j] + 1024 * u.pixel,
                position.y[j] + 512 * u.pixel,
                color=colors[i],
                ax=ax,
                s=8,
                where=rays.unvignetted[j],
                label=spectral_lines[i],
            )
        ax.ndarray[0].legend(loc="upper right");
    """
    model = design(
        grid=grid,
        axis_channel=axis_channel,
        num_distribution=num_distribution,
    )

    model.wavelength = na.ScalarArray(
        u.Quantity(
            [
                He_I.wavelength,
                Mg_X.wavelength,
                O_V.wavelength,
            ]
        ),
        axes="wavelength",
    )

    model.grating.yaw = (
        na.ScalarArray(
            np.array([-2.693e02, -2.681e02, -2.687e02, -2.680e02]),
            axes=axis_channel,
        )
        * u.arcmin
    )
    model.grating.pitch = (
        na.ScalarArray(
            np.array([3.704e00, 1.522e00, 1.316e00, 5.705e00]),
            axes=axis_channel,
        )
        * u.arcmin
    )
    model.grating.roll = (
        na.ScalarArray(
            np.array([1.027e00, 2.393e-01, 3.678e-01, 1.020e00]),
            axes=axis_channel,
        )
        * u.deg
    )
    model.field_stop.roll = (
        na.ScalarArray(
            np.array([-2.066e-01, -2.891e-01, -5.264e-01, 1.182e00]),
            axes=axis_channel,
        )
        * u.deg
    )
    model.grating.rulings.spacing.coefficients[0] = (
        na.ScalarArray(
            np.array([3.854e-01, 3.859e-01, 3.855e-01, 3.863e-01]),
            axes=axis_channel,
        )
        * u.um
    )

    # The fitted primary-mirror displacement relative to its -1000 mm nominal
    # focal length; this hard reference is what the fit is measured from.
    primary_displacement = (
        na.ScalarArray(
            np.array([-5.649e00, -2.207e-02, -2.795e00, -1.616e00]),
            axes=axis_channel,
        )
        * u.mm
    )
    model.primary_mirror.sag.focal_length = -1000 * u.mm + primary_displacement
    model.primary_mirror.translation.z = -primary_displacement

    model.pitch = (
        na.ScalarArray(
            np.array([-2.024e01, -2.096e01, -1.973e01, -2.119e01]),
            axes=axis_channel,
        )
        * u.arcsec
    )
    model.yaw = (
        na.ScalarArray(
            np.array([-1.832e01, -1.675e01, -1.604e01, -1.498e01]),
            axes=axis_channel,
        )
        * u.arcsec
    )
    model.roll = (
        na.ScalarArray(
            np.array([-8.681e-01, -3.391e-01, -3.378e-01, -1.109e00]),
            axes=axis_channel,
        )
        * u.deg
    )

    return model
