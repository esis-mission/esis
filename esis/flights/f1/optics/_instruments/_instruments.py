import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import esis
from esis.flights.f1.spectrum import He_I, Mg_X, O_V
from .. import primaries
from .. import gratings
from .. import filters

__all__ = [
    "design_full",
    "design",
    "design_single",
    "as_built",
    "distortion_fit",
    "distortion_fit_bounds",
]


def design_full(
    grid: None | optika.vectors.ObjectVectorArray = None,
    axis_channel: str = "channel",
    num_distribution: int = 11,
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
    """
    num_folds = 8
    num_channels = 6

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
            x=na.UniformUncertainScalarArray(
                nominal=0 * u.mm,
                width=1 * u.mm,
                num_distribution=num_distribution,
            ),
            y=na.UniformUncertainScalarArray(
                nominal=0 * u.mm,
                width=1 * u.mm,
                num_distribution=num_distribution,
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
            radius=na.UniformUncertainScalarArray(
                nominal=-radius_grating,
                width=radius_grating * error_radius_grating,
                num_distribution=num_distribution,
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
            x=na.UniformUncertainScalarArray(
                nominal=0 * u.mm,
                width=1 * u.mm,
                num_distribution=num_distribution,
            ),
            y=na.UniformUncertainScalarArray(
                nominal=0 * u.mm,
                width=1 * u.mm,
                num_distribution=num_distribution,
            ),
            z=na.UniformUncertainScalarArray(
                nominal=primary.sag.focal_length - 374.7 * u.mm,
                width=error_grating_z,
                num_distribution=num_distribution,
            ),
        ),
        yaw=-4.469567242792327 * u.deg,
        roll=na.UniformUncertainScalarArray(
            nominal=0 * u.deg,
            width=1.3e-2 * u.rad,
            num_distribution=num_distribution,
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
            z=grating.translation.z.nominal + 1.301661998854058 * u.m,
        ),
        yaw=-3.45 * u.deg,
        roll=45 * u.deg,
    )

    sensor = esis.optics.Sensor(
        # The physical mask on the ESIS-I detectors was undersized, leaving
        # readout-buffer rows exposed to light.  Science data extends into
        # those rows, so the active area is 2 x 1040 rows, matching the
        # 1040-row halves of the Level-1 frames.
        num_pixel_y=2 * 1040,
        distance_radial=108 * u.mm,
        azimuth=angle_channel.copy(),
        translation=na.Cartesian3dVectorArray(
            x=0 * u.mm,
            y=0 * u.mm,
            z=filter.translation.z + 200 * u.mm,
        ),
        yaw=-12.252 * u.deg,
        material=optika.sensors.materials.e2v_ccd97(
            temperature=-55 * u.deg_C,
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

    if num_distribution == 0:
        primary.translation = na.nominal(primary.translation)
        field_stop.translation = na.nominal(field_stop.translation)
        grating.sag.radius = na.nominal(grating.sag.radius)
        grating.rulings.spacing.coefficients[0] = na.nominal(
            grating.rulings.spacing.coefficients[0]
        )
        grating.rulings.spacing.coefficients[1] = na.nominal(
            grating.rulings.spacing.coefficients[1]
        )
        grating.rulings.spacing.coefficients[2] = na.nominal(
            grating.rulings.spacing.coefficients[2]
        )
        grating.rulings.depth = na.nominal(grating.rulings.depth)
        grating.rulings.ratio_duty = na.nominal(grating.rulings.ratio_duty)
        grating.translation = na.nominal(grating.translation)
        grating.roll = na.nominal(grating.roll)

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
    """
    result = design_full(
        grid=grid,
        axis_channel=axis_channel,
        num_distribution=num_distribution,
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
    """
    result = design(
        grid=grid,
        axis_channel=axis_channel,
        num_distribution=num_distribution,
    )

    index = {axis_channel: 0}

    result.grating.azimuth = result.grating.azimuth[index]
    result.filter.azimuth = result.filter.azimuth[index]

    result.camera.channel = result.camera.channel[index]
    result.camera.sensor.azimuth = result.camera.sensor.azimuth[index]

    result.roll = -result.grating.azimuth

    return result


def as_built(
    grid: None | optika.vectors.ObjectVectorArray = None,
    axis_channel: str = "channel",
    num_distribution: int = 11,
) -> esis.optics.Instrument:
    """
    Load the as-built optical model.

    Based on :func:`design`, but includes efficiency and figure measurements of the
    primary mirror and gratings, as well as gain measurements of the sensor.

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
    Load the as-built optical model and print its parameters.

    .. jupyter-execute::

        import esis

        esis.flights.f1.optics.as_built()
    """
    result = design(
        grid=grid,
        axis_channel=axis_channel,
        num_distribution=num_distribution,
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
    result.grating.sag.radius = na.stack(
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

    # Results from Laurel Rachmeler presented on 2017-07-06 and 2017-07-12.
    result.camera.gain = na.ScalarArray(
        ndarray=[
            [
                [2.55, 2.63],
                [2.57, 2.57],
            ],
            [
                [2.57, 2.53],
                [2.50, 2.52],
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

    result.camera.sensor.readout_noise = 6 * u.electron

    return result


def distortion_fit(
    grid: None | optika.vectors.ObjectVectorArray = None,
    axis_channel: str = "channel",
    num_distribution: int = 11,
    axis_time: None | str = None,
) -> esis.optics.Instrument:
    """
    Apply the best-fit distortion parameters to the ESIS-I :func:`design`.

    The parameters are hard-coded from the best distortion fit of the ESIS-I
    flight data, optimized against the ``time=15`` frame of the 2019-09-30
    flight (:func:`esis.flights.f1.data.level_1`, with a start time of
    2019-09-30T18:08:41.642 UTC). The values are per-channel and were produced
    by the ``ESISI_distortion_optimization_20260213_151715`` run.

    If `axis_time` is given, the instrument pointing additionally carries the
    fitted per-frame payload pointing along that axis, one element per frame
    of :func:`esis.flights.f1.data.level_1`. During the flight the payload
    pointing drifted by several arcseconds (dominated by yaw, which sweeps
    monotonically from :math:`+3.3''` at the first frame to :math:`-4.4''` at
    the last); the optics are otherwise held fixed at the reference fit. The
    offsets are common to all four channels (a rigid-payload model) and were
    measured independently for every frame on 2026-07-06 by scanning the
    correlation between the imaged AIA scene and each Level-1 frame over a
    common pitch/yaw/roll offset and refining each axis with a parabola fit
    (a :math:`401^2` scene, a 1-pixel-deviation Gaussian point-spread
    function applied to the modeled image, and two coarse-to-fine rounds per
    frame).

    Parameters
    ----------
    grid
        sampling of wavelength, field, and pupil positions that will be used to
        characterize the optical system.
    axis_channel
        The name of the logical axis corresponding to changing camera channel.
    num_distribution
        number of Monte Carlo samples to draw when computing uncertainties
    axis_time
        The name of the logical axis corresponding to changing time.
        If :obj:`None`, the pointing is that of the ``time=15`` reference fit;
        otherwise the pitch, yaw, and roll gain one element per Level-1 frame.

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

    # Per-channel pointing re-polished 2026-07-07 against the time=15 frame
    # with the deterministic PSF-smoothed correlation (401-pixel scene,
    # per-channel peaks read from coherent pointing scans), correcting
    # constant per-channel misregistrations of up to ~2.5 detector pixels
    # left by the original stochastic 201-pixel optimization.
    model.pitch = (
        na.ScalarArray(
            np.array([-19.7717, -21.25244, -22.08457, -21.68604]),
            axes=axis_channel,
        )
        * u.arcsec
    )
    model.yaw = (
        na.ScalarArray(
            np.array([-20.13878, -16.59384, -16.49289, -15.38459]),
            axes=axis_channel,
        )
        * u.arcsec
    )
    model.roll = (
        na.ScalarArray(
            np.array([-0.82038, -0.3028, -0.24703, -1.06117]),
            axes=axis_channel,
        )
        * u.deg
    )

    if axis_time is not None:
        model.pitch = model.pitch + (
            na.ScalarArray(np.array(_pointing_pitch), axes=axis_time) * u.arcsec
        )
        model.yaw = model.yaw + (
            na.ScalarArray(np.array(_pointing_yaw), axes=axis_time) * u.arcsec
        )
        model.roll = model.roll + (
            na.ScalarArray(np.array(_pointing_roll), axes=axis_time) * u.deg
        )

    return model


# The per-frame payload pointing offsets relative to the time=15 reference
# fit, one element per frame of esis.flights.f1.data.level_1(), measured
# 2026-07-07 by per-frame coherent pitch/yaw/roll merit scans against the
# re-polished per-channel reference (see the distortion_fit docstring).
# The time=15 elements are consistent with zero, confirming the reference
# and the time series agree.
_pointing_pitch = [
    0.9805,
    0.7607,
    0.4168,
    0.3388,
    0.3081,
    0.2087,
    0.0614,
    0.0413,
    0.0284,
    0.1326,
    0.0550,
    -0.0571,
    -0.3421,
    -0.1918,
    -0.1614,
    0.0705,
    0.1508,
    -0.0703,
    0.0320,
    -0.0064,
    0.0028,
    0.0424,
    0.0603,
    -0.0489,
    -0.0206,
    -0.1942,
    -0.2018,
    -0.0822,
    -0.0822,
    -0.0402,
]
_pointing_yaw = [
    3.6928,
    3.6224,
    3.2995,
    3.2085,
    2.9156,
    2.6023,
    2.4250,
    2.1710,
    1.9898,
    1.6427,
    1.4558,
    1.0418,
    0.9022,
    0.5417,
    0.3938,
    -0.0053,
    -0.1843,
    -0.4411,
    -0.8831,
    -0.9857,
    -1.1723,
    -1.7005,
    -1.9098,
    -2.0489,
    -2.6097,
    -2.7764,
    -3.1531,
    -3.4944,
    -3.7320,
    -4.0488,
]
_pointing_roll = [
    -0.00451,
    -0.05431,
    0.00271,
    0.02175,
    -0.03299,
    -0.02115,
    -0.00750,
    -0.03146,
    -0.00485,
    -0.00274,
    -0.04148,
    -0.02061,
    -0.04902,
    -0.02972,
    -0.08825,
    0.01418,
    -0.03381,
    -0.06777,
    -0.01660,
    -0.03867,
    0.01496,
    -0.01852,
    -0.01871,
    -0.03029,
    -0.03952,
    -0.05201,
    -0.02006,
    -0.05063,
    -0.01753,
    -0.01305,
]


def distortion_fit_bounds(
    parameters: esis.optics.DistortionParameters,
) -> tuple[esis.optics.DistortionParameters, esis.optics.DistortionParameters]:
    r"""
    Compute the parameter bounds used when fitting the ESIS-I distortion.

    Most parameters are bounded at :math:`\pm 20\%` of the given initial
    guess. The roll angles and the primary-mirror displacement are instead
    given the hand-tuned absolute bounds of the
    ``ESISI_distortion_optimization_20260213_151715`` run, the best fit of the
    ESIS-I flight data (and the source of the values in
    :func:`distortion_fit`), since their initial guesses are zero or nearly so.

    The bounds are expressed in the same units as `parameters` so that
    flattening both with :func:`named_arrays.pack` yields consistent vectors.

    Parameters
    ----------
    parameters
        The initial guess of the fit.

    Examples
    --------
    Compute the bounds for fitting the ESIS flight-1 design.

    .. jupyter-execute::

        import named_arrays as na
        import esis

        instrument = esis.flights.f1.optics.design(num_distribution=0)
        parameters = esis.optics.DistortionParameters.from_instrument(instrument)
        lower, upper = esis.flights.f1.optics.distortion_fit_bounds(parameters)

        na.pack(lower), na.pack(upper)
    """
    p = parameters

    lower = esis.optics.DistortionParameters(
        yaw_grating=np.minimum(0.8 * p.yaw_grating, 1.2 * p.yaw_grating),
        pitch_grating=np.minimum(0.8 * p.pitch_grating, 1.2 * p.pitch_grating),
        roll_grating=(-2 * u.deg).to(na.unit(p.roll_grating)),
        roll_field_stop=(-4 * u.deg).to(na.unit(p.roll_field_stop)),
        spacing_rulings=np.minimum(0.8 * p.spacing_rulings, 1.2 * p.spacing_rulings),
        displacement_primary=(-10 * u.mm).to(na.unit(p.displacement_primary)),
        pitch=np.minimum(0.8 * p.pitch, 1.2 * p.pitch),
        yaw=np.minimum(0.8 * p.yaw, 1.2 * p.yaw),
        roll=(-4 * u.deg).to(na.unit(p.roll)),
    )

    upper = esis.optics.DistortionParameters(
        yaw_grating=np.maximum(0.8 * p.yaw_grating, 1.2 * p.yaw_grating),
        pitch_grating=np.maximum(0.8 * p.pitch_grating, 1.2 * p.pitch_grating),
        roll_grating=(2 * u.deg).to(na.unit(p.roll_grating)),
        roll_field_stop=(4 * u.deg).to(na.unit(p.roll_field_stop)),
        spacing_rulings=np.maximum(0.8 * p.spacing_rulings, 1.2 * p.spacing_rulings),
        displacement_primary=(0 * u.mm).to(na.unit(p.displacement_primary)),
        pitch=np.maximum(0.8 * p.pitch, 1.2 * p.pitch),
        yaw=np.maximum(0.8 * p.yaw, 1.2 * p.yaw),
        roll=(0 * u.deg).to(na.unit(p.roll)),
    )

    return lower, upper
