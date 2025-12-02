import astropy.units as u
import optika
import esis

__all__ = [
    "design_proposed",
]


def design_proposed(
    grid: None | optika.vectors.ObjectVectorArray = None,
    axis_channel: str = "channel",
    num_distribution: int = 11,
) -> esis.optics.Instrument:
    """
    Load the proposed optical design for ESIS 2.

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
    Plot the rays traveling through the optical system, as viewed from the front.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika
        import esis

        grid = optika.vectors.ObjectVectorArray(
            wavelength=na.linspace(-1, 1, axis="wavelength", num=2, centers=True),
            field=0.99 * na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=5,
            ),
            pupil=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
                num=5,
            ),
        )

        instrument = esis.flights.f2.optics.design_proposed(
            grid=grid,
            num_distribution=0,
        )

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(
                figsize=(6, 6.5),
                constrained_layout=True
            )
            ax.set_aspect("equal")
            instrument.system.plot(
                components=("x", "y"),
                color="black",
                kwargs_rays=dict(
                    color=na.ScalarArray(
                        ndarray=np.array(["tab:orange", "tab:blue"]),
                        axes="wavelength",
                    ),
                    label=instrument.system.grid_input.wavelength.astype(int),
                ),
            );
            handles, labels = ax.get_legend_handles_labels()
            labels = dict(zip(labels, handles))
            fig.legend(labels.values(), labels.keys());
    """
    result = esis.flights.f1.optics.design_full(
        grid=grid,
        axis_channel=axis_channel,
        num_distribution=num_distribution,
    )

    c0 = 1 / (2700 / u.mm)
    c1 = -2.852e-5 * (u.um / u.mm)
    c2 = -2.112e-7 * (u.um / u.mm**2)

    if num_distribution == 0:
        result.grating.rulings.spacing.coefficients[0] = c0
        result.grating.rulings.spacing.coefficients[1] = c1
        result.grating.rulings.spacing.coefficients[2] = c2
        z_filter = result.grating.translation.z + 1291.012 * u.mm
    else:
        result.grating.rulings.spacing.coefficients[0].nominal = c0
        result.grating.rulings.spacing.coefficients[1].nominal = c1
        result.grating.rulings.spacing.coefficients[2].nominal = c2
        z_filter = result.grating.translation.z.nominal + 1291.012 * u.mm

    result.grating.yaw = -3.65 * u.deg

    dz = z_filter - result.filter.translation.z
    result.filter.translation.z += dz
    result.camera.sensor.translation.z += dz

    return result
