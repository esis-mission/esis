import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import esis
from .. import primaries
from .. import gratings
from .. import filters

__all__ = [
    "design_full",
    "design",
    "design_single",
]


def design_full(
    grid: None | optika.vectors.ObjectVectorArray = None,
    axis_channel: str = "channel",
    num_distribution: int = 11,
) -> esis.optics.Instrument:
    """
    Final ESIS optical design prepared by Charles Kankelborg and Hans Courrier.

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
        temperature=-55 * u.deg_C,
        distance_radial=108 * u.mm,
        azimuth=angle_channel.copy(),
        translation=na.Cartesian3dVectorArray(
            x=0 * u.mm,
            y=0 * u.mm,
            z=filter.translation.z + 200 * u.mm,
        ),
        yaw=-12.252 * u.deg,
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
) -> esis.optics.Instrument:
    """
    Final ESIS optical design prepared by Charles Kankelborg and Hans Courrier.

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
    Final ESIS optical design prepared by Charles Kankelborg and Hans Courrier.

    This instance includes only one channel.
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

    Examples
    --------
    Plot the rays traveling through the optical system, as viewed from the side.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika
        import esis

        grid = optika.vectors.ObjectVectorArray(
            wavelength=na.linspace(-1, 1, axis="wavelength",  num=2) / 2,
            field=0,
            pupil=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
                num=5,
            ),
        )

        model = esis.flights.f1.optics.models.design_single(
            grid=grid,
            num_distribution=0,
        )

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(
                figsize=(8, 2),
                constrained_layout=True
            )
            ax.set_aspect("equal")
            model.system.plot(
                components=("z", "x"),
                color="black",
                kwargs_rays=dict(
                    color=na.ScalarArray(np.array(["tab:orange", "tab:blue"]), axes="wavelength"),
                    label=model.system.grid_input.wavelength.astype(int),
                ),
            );
            handles, labels = ax.get_legend_handles_labels()
            labels = dict(zip(labels, handles))
            fig.legend(labels.values(), labels.keys());
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
