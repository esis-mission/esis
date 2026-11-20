import astropy.units as u
import optika
import esis

__all__ = [
    "thin_film_design",
]


def thin_film_design() -> optika.materials.ThinFilmFilter:
    """
    Load a model of the thin-film filter design.

    The filters are 100 nm of Al on a nickel mesh. Al grows an oxide in air,
    on both sides of the film, and the oxide absorbs strongly in the EUV, so
    the transmission of a filter depends mostly on how old it is.
    :cite:t:`Powell1990` found that filters transmit about as a model with
    18 Å of Al2O3 does near the time they are made, and as one with 50 to
    100 Å does once they have aged for a year or more. The ESIS-I filters
    were made in early 2019, about seven months before the flight.

    This model puts 2 nm of amorphous Al2O3 on each side of the film. That is
    a single value for every channel, between the 1.4 and 3.0 nm per side
    which the brightness of the flight images at O V needs once the primary
    and the gratings are modeled as they were at the flight. The channels
    differ by more than one oxide can explain, and those differences are not
    modeled here.

    Examples
    --------
    Plot the transmissivity of the thin-film filter over the EUV wavelength
    range.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika
        from esis.flights.f1.optics import filters

        # Define an array of wavelengths with which to sample the efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incident rays from the wavelength array
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        )

        # Initialize the ESIS diffraction grating material
        material = filters.materials.thin_film_design()

        # Compute the reflectivity of the primary mirror
        transmissivity = material.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the transmissivity vs wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(wavelength, transmissivity, ax=ax);
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("transmissivity");

    Plot the schematic of the layers composing the filter

    .. jupyter-execute::

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            material.plot_layers(
                ax=ax,
            )
            ax.set_axis_off()
            ax.autoscale()
    """
    return optika.materials.ThinFilmFilter(
        num_interpolation=esis.optics.num_interpolation,
        layer=optika.materials.Layer(
            chemical="Al",
            thickness=100 * u.nm,
            kwargs_plot=dict(
                color="lightgray",
            ),
        ),
        layer_oxide=optika.materials.Layer(
            chemical=optika.chemicals.Chemical(
                formula="Al2O3",
                is_amorphous=True,
                table="palik",
            ),
            thickness=2 * u.nm,
            kwargs_plot=dict(
                color="gray",
            ),
            x_label=1.1,
        ),
        mesh=optika.materials.meshes.Mesh(
            chemical="Ni",
            efficiency=0.82,
            pitch=70 / u.imperial.inch,
        ),
    )
