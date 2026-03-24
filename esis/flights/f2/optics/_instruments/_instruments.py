import astropy.units as u
import numpy as np
import named_arrays as na
import optika
import esis
from ... import wavelength_Ne_VII, wavelength_Si_XII

__all__ = [
    "design_proposed",
    "design_guess",
    "design_single",
    "design",
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


def design_guess(
    grid: None | optika.vectors.ObjectVectorArray = None,
    axis_channel: str = "channel",
    num_distribution: int = 11,
) -> esis.optics.Instrument:
    r"""
    Load the starting point (or guess) for optimization of the ESIS-II design.

    This model uses the entire optical bench length for increased resolution.

    The changes from ESIS-I are as follows:
     - The target wavelengths have changed to Ne VII 46.5 nm and Si XII 49.9 nm.
     - The primary mirror is moved back by 6 holes on the optical bench.
     - The gratings are moved forward by 2 holes on the optical bench.
     - The filter position and orientation has been adjusted to account
       for the shallower beam.

    To account for these changes, the angle, radius of curvature, and the
    ruling parameters of the grating need to be adjusted.
    This function uses the grating equation and :cite:t:`Poletto2004` to
    estimate these parameters.
    These estimates are intended to be used as a starting point for a local
    minimization procedure to find the best-fit values of these parameters.

    Parameters
    ----------
    grid
        sampling of wavelength, field, and pupil positions that will be used to
        characterize the optical system.
    axis_channel
        The name of the logical axis corresponding to changing camera channel.
    num_distribution
        number of Monte Carlo samples to draw when computing uncertainties

    Notes
    -----
    Let :math:`\mathbf{a}` be the vector pointing from the apex of the grating
    to the center of the field stop, and let :math:`\mathbf{b}_i` be the vector
    pointing from the apex of the grating to target location of wavelength
    :math:`i` on the sensor.

    The angle between :math:`\mathbf{a}` and the optic axis, :math:`\hat{\mathbf{z}}`,
    is then

    .. math::

        a = \arctan \left( \frac{\mathbf{a} \cdot \hat{\mathbf{x}}}
                                {\mathbf{a} \cdot \hat{\mathbf{z}}} \right),

    and similarly for :math:`\mathbf{b}_i`,

    .. math::

        b_i = \arctan \left( \frac{\mathbf{b}_i \cdot \hat{\mathbf{x}}}
                                  {\mathbf{b}_i \cdot \hat{\mathbf{z}}} \right).

    If the grating is rotated about the :math:`\hat{\mathbf{y}}` axis by
    :math:`\theta`, then the relationship between the incident/diffracted angles
    and :math:`a`/:math:`b` is

    .. math::

        \alpha &= a - \theta \\
        \beta_i &= b_i - \theta,

    and the grating equation becomes

    .. math::
        :label: grating-equation

        \sin{(a - \theta)} + \sin{(b_i - \theta)} = \frac{m \lambda_i}{d},

    where :math:`m` is the diffraction order,
    :math:`\lambda_i` is the wavelength,
    and :math:`d` the grating ruling spacing.

    If we consider two target wavelengths, :math:`\lambda_1` and :math:`\lambda_2`,
    Equation :eq:`grating-equation` is a system of two equations which can be
    solved to find the grating yaw angle,

    .. math::

        \theta = -\arccos \left[
            \frac{(\lambda_2 - \lambda_1) \cos a
                  + \lambda_2 \cos b_1
                  - \lambda_1 \cos b_2}
                 {\sqrt{2} \sqrt{
                    \lambda_1^2 - \lambda_1 \lambda_2 + \lambda_2^2
                    + \left( \lambda_1 - \lambda_2)(\lambda_1 \cos (a-b_2)
                        - \lambda_2 \cos(a-b1) \right)
                    - \lambda_1 \lambda_2 \cos(b1-b2)
                 }
            }
        \right],

    and the ruling spacing,

    .. math::

        d = \frac{m \lambda_1}{\sin(a - \theta) + \sin(b_1 - \theta)}.

    The radius of the grating surface can be found using Equation 31 of
    :cite:t:`Poletto2004`,

    .. math::

        R = r_a (\cos \alpha + \cos \beta) \frac{M_c}{1 + M_c},

    where :math:`r_a = |\mathbf{a}|` is the length of the entrance arm,
    :math:`\beta_c` is the diffracted angle of the center wavelength,
    :math:`\lambda_c = (\lambda_1 + \lambda_2) / 2`,
    :math:`M_c = r_b / r_a` is the magnification of the center wavelength,
    and :math:`r_b = |\mathbf{b}_c|` is the length of the exit arm at the center
    wavelength.

    Finally, the linear term for the ruling density VLS law can be found using
    Equation 26 of :cite:t:`Poletto2004`,

    .. math::

        \sigma_1 = -\frac{1}{m \lambda_c} \left(
            \frac{\cos^2 \alpha}{r_a}
            + \frac{\cos^2 \beta}{r_b}
            - \frac{\cos \alpha + \cos \beta}{R}
        \right).

    These equations taken together can be used to formulate a close guess
    to an optimal ESIS-II design.

    Examples
    --------
    Plot a spot diagram for this design.

    .. jupyter-execute::

        # Import this package
        import esis

        # Load this design into memory
        instrument = esis.flights.f2.optics.design_guess(num_distribution=0)

        # Lower the number of field angles for clearer plotting
        instrument.field.num = 5

        # Plot the spot diagram for each field angle
        fig, ax = instrument.system.spot_diagram()

    Print the calculated parameters of this design.

    .. jupyter-execute::

        instrument.grating.sag.radius.ndarray

    .. jupyter-execute::

        instrument.grating.yaw.ndarray

    .. jupyter-execute::

        instrument.grating.rulings.spacing.coefficients[0].ndarray

    .. jupyter-execute::

        instrument.grating.rulings.spacing.coefficients[1].ndarray
    """
    result = esis.flights.f1.optics.design_single(
        grid=grid,
        axis_channel=axis_channel,
        num_distribution=num_distribution,
    )

    w1 = wavelength_Ne_VII
    w2 = wavelength_Si_XII

    primary = result.primary_mirror
    grating = result.grating
    fs = result.field_stop
    filt = result.filter
    sensor = result.camera.sensor

    lots_hole_spacing = (4 * u.imperial.inch).to(u.mm)

    # Extend the focal length of the primary mirror
    primary.sag.focal_length = primary.sag.focal_length - 6 * lots_hole_spacing

    # Save the old distance between the field stop and the grating
    dz_grating_old = grating.translation.z - fs.translation.z

    # Compute the new distance between the field stop and grating
    dz_grating_new = dz_grating_old - 2 * lots_hole_spacing

    # Move the field stop to the new focus
    fs.translation.z = primary.sag.focal_length

    # Move the grating to its new position
    grating.translation.z = fs.translation.z + dz_grating_new

    result.central_obscuration.translation.z = grating.translation.z - 25 * u.mm
    result.front_aperture.translation.z = grating.translation.z - 100 * u.mm

    t_system = result.transformation

    zero = na.Cartesian3dVectorArray() * u.mm
    offset_Ne_VII = na.Cartesian3dVectorArray(x=-7.35) * u.mm
    offset_Si_XII = na.Cartesian3dVectorArray(x=+7.35) * u.mm

    position_fs = t_system(fs.transformation(zero))
    position_grating = t_system(grating.transformation(zero))
    position_sensor = t_system(sensor.transformation(zero))
    position_Ne_VII = t_system(sensor.transformation(offset_Ne_VII))
    position_Si_XII = t_system(sensor.transformation(offset_Si_XII))

    direction_fs = position_fs - position_grating
    direction_sensor = position_sensor - position_grating
    direction_Ne_VII = position_Ne_VII - position_grating
    direction_Si_XII = position_Si_XII - position_grating

    a = np.arctan2(direction_fs.x, direction_fs.z)
    b = np.arctan2(direction_sensor.x, direction_sensor.z)
    b1 = np.arctan2(direction_Ne_VII.x, direction_Ne_VII.z)
    b2 = np.arctan2(direction_Si_XII.x, direction_Si_XII.z)

    numerator = (w2 - w1) * np.cos(a) + w2 * np.cos(b1) - w1 * np.cos(b2)

    term_1 = np.square(w1) - w1 * w2 + np.square(w2)
    term_2 = (w1 - w2) * (-w2 * np.cos(a - b1) + w1 * np.cos(a - b2))
    term_3 = -w1 * w2 * np.cos(b1 - b2)

    denominator = np.sqrt(2) * np.sqrt(term_1 + term_2 + term_3)

    theta = -np.arccos(numerator / denominator)

    m = -grating.rulings.diffraction_order
    d = m * w1 / (np.sin(a - theta) + np.sin(b1 - theta))

    alpha = a - theta
    beta = b - theta

    r_A = direction_fs.length
    r_B = direction_sensor.length

    M_c = r_B / r_A

    R = r_A * (np.cos(alpha) + np.cos(beta)) * M_c / (1 + M_c)

    sigma_1 = -(
        np.square(np.cos(alpha)) / r_A
        + np.square(np.cos(beta)) / r_B
        - (np.cos(alpha) + np.cos(beta)) / R
    ) / (m * (w1 + w2) / 2)

    d_1 = -sigma_1 * np.square(d)

    yaw_grating = theta.to(u.deg)
    c0 = d
    c1 = d_1.to(u.um / u.mm)
    c2 = 0 * (u.um / u.mm**2)
    radius_grating = -R

    grating.yaw = yaw_grating

    if num_distribution == 0:
        result.grating.rulings.spacing.coefficients[0] = c0
        result.grating.rulings.spacing.coefficients[1] = c1
        result.grating.rulings.spacing.coefficients[2] = c2
        result.grating.sag.radius = radius_grating
    else:
        result.grating.rulings.spacing.coefficients[0].nominal = c0
        result.grating.rulings.spacing.coefficients[1].nominal = c1
        result.grating.rulings.spacing.coefficients[2].nominal = c2
        result.grating.sag.radius.nominal = radius_grating

    filt.yaw = b

    dz_filter = filt.translation.z - sensor.translation.z

    filt.distance_radial = sensor.distance_radial + dz_filter * np.tan(b)

    if grid is None or grid.wavelength is None:
        result.wavelength = na.stack(
            arrays=[w1, w2],
            axis="wavelength",
        )

    return result


def design_single(
    grid: None | optika.vectors.ObjectVectorArray = None,
    axis_channel: str = "channel",
    num_distribution: int = 11,
) -> esis.optics.Instrument:
    r"""
    Load a single channel of the ESIS-II design.

    This model starts with :func:`~esis.flights.f2.optics.design_guess`
    and modifies the grating yaw, radius, and VLS parameters to those
    found by Jacob D. Parker in July 2024.

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
    Plot a spot diagram for this design.

    .. jupyter-execute::

        # Import this package
        import esis

        # Load this design into memory
        instrument = esis.flights.f2.optics.design_single(num_distribution=0)

        # Lower the number of field angles for clearer plotting
        instrument.field.num = 5

        # Plot the spot diagram for each field angle
        fig, ax = instrument.system.spot_diagram()

    Print the calculated parameters of this design.

    .. jupyter-execute::

        instrument.grating.sag.radius

    .. jupyter-execute::

        instrument.grating.yaw

    .. jupyter-execute::

        instrument.grating.rulings.spacing.coefficients[0]

    .. jupyter-execute::

        instrument.grating.rulings.spacing.coefficients[1]

    .. jupyter-execute::

        instrument.grating.rulings.spacing.coefficients[2]
    """
    result = design_guess(
        grid=grid,
        axis_channel=axis_channel,
        num_distribution=num_distribution,
    )

    yaw_grating = -2.42796088e00 * u.deg
    c0 = 5.57902824e-04 * u.mm
    c1 = -1.79596543e-05 * u.um / u.mm
    c2 = -1.67614260e-07 * u.um / u.mm**2
    radius_grating = -9.24015556e02 * u.mm

    result.grating.yaw = yaw_grating

    if num_distribution == 0:
        result.grating.rulings.spacing.coefficients[0] = c0
        result.grating.rulings.spacing.coefficients[1] = c1
        result.grating.rulings.spacing.coefficients[2] = c2
        result.grating.sag.radius = radius_grating
    else:
        result.grating.rulings.spacing.coefficients[0].nominal = c0
        result.grating.rulings.spacing.coefficients[1].nominal = c1
        result.grating.rulings.spacing.coefficients[2].nominal = c2
        result.grating.sag.radius.nominal = radius_grating

    return result


def design(
    grid: None | optika.vectors.ObjectVectorArray = None,
    axis_channel: str = "channel",
    num_distribution: int = 11,
) -> esis.optics.Instrument:
    r"""
    Load all six channels of the ESIS-II design.

    This model starts with :func:`~esis.flights.f2.optics.design_single`
    and modifies the azimuth of the gratings, filters, and detectors
    to be a six-element vector.

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
    old = esis.flights.f1.optics.design_full(
        grid=grid,
        axis_channel=axis_channel,
        num_distribution=num_distribution,
    )

    result = design_single(
        grid=grid,
        axis_channel=axis_channel,
        num_distribution=num_distribution,
    )

    result.grating.azimuth = old.grating.azimuth
    result.filter.azimuth = old.filter.azimuth
    result.camera.sensor.azimuth = old.camera.sensor.azimuth

    result.camera.channel = old.camera.channel

    result.roll = old.roll

    return result
