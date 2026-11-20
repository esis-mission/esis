import pathlib
import numpy as np
import scipy.optimize
import astropy.units as u
import astropy.time
import named_arrays as na
import optika
import esis
from esis import _caching
from esis.flights.f1.data._fits._fits import _path_cache, _retrieve

__all__ = [
    "multilayer_AlSc",
    "multilayer_SiSc",
    "time_coating_sample",
    "multilayer_SiSc_sample_measured",
    "multilayer_SiSc_sample_fit",
]

_url = (
    "https://github.com/esis-mission/esis-data-f2/releases/download"
    "/multilayer-v1.0/esis-f2-multilayer.tar"
)
_hash = "sha256:6c6ef8ced0b890dff1c358d231472846d2f741460bdb7b175be4adc8f4dad3e1"

time_coating_sample = astropy.time.Time("2025-09-16")
"""
When the sample of :func:`multilayer_SiSc_sample_measured` was coated.

The date is the one in the name of the sample, ``CX250916B``, which heads its
measurement.
It was measured about two months later.
"""

_thickness_oxide = 1.5 * u.nm
"""
The thickness of the native oxide on the top Si layer of a Si/Sc coating.

It is Eric Gullikson's estimate for the sample of
:func:`multilayer_SiSc_sample_measured`.
"""

_ratio_oxide = 0.44
"""The thickness of Si consumed in growing a unit thickness of its oxide."""

_fraction_s = (1 + 0.98) / 2
"""
The fraction of the intensity of the beam which measured the sample of
:func:`multilayer_SiSc_sample_measured` that is s-polarized.

The beam is 0.98 linearly polarized, with its electric field perpendicular to
the plane of incidence.
"""

_packages = (
    "euv-snapshot-imaging-spectrograph",
    "optika",
    "named-arrays",
    "scipy",
)
"""
The packages whose versions the cache of :func:`multilayer_SiSc_sample_fit` is
keyed on.

They hold the model of the sample (this package), compute the reflectivity of
a multilayer stack (:mod:`optika` and :mod:`named_arrays`), and minimize its
difference from the measurement (:mod:`scipy`).
"""


def _path_directory() -> pathlib.Path:
    """
    Return the directory containing the measurements of multilayer coatings.

    They are measurements of coatings for the second flight, downloaded and
    unpacked into :func:`esis.flights.f1.data._fits._fits._path_cache` the
    first time they are needed.
    """
    _retrieve(
        url=_url,
        known_hash=_hash,
        fname="esis-f2-multilayer.tar",
        path=_path_cache(),
        extract_dir="multilayer-f2",
    )
    return _path_cache() / "multilayer-f2"


def multilayer_AlSc() -> optika.materials.MultilayerMirror:
    r"""
    Load a proposed coating design for ESIS II which uses :math:`\text{Al/Sc}` layers.

    Examples
    --------
    Plot the efficiency of the coating across the EUV range

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika
        import esis

        # Define an array of wavelengths with which to sample the efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incident rays from the wavelength array
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        )

        # Initialize the multilayer material
        material = esis.flights.f2.optics.materials.multilayer_AlSc()

        # Compute the reflectivity of the multilayer material
        reflectivity = material.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the reflectivity vs wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(wavelength, reflectivity, ax=ax);
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("reflectivity");
    """
    d = 257 * u.AA
    gamma = 0.5
    return optika.materials.MultilayerMirror(
        num_interpolation=esis.optics.num_interpolation,
        layers=[
            optika.materials.Layer(
                # The table for crystalline alumina starts at 2066 Angstroms,
                # so use the amorphous table, which covers the EUV.
                chemical=optika.chemicals.Chemical(
                    formula="Al2O3",
                    is_amorphous=True,
                ),
                thickness=1 * u.nm,
                interface=optika.materials.profiles.ErfInterfaceProfile(
                    width=7 * u.AA,
                ),
                kwargs_plot=dict(
                    color="tab:blue",
                    alpha=0.3,
                ),
            ),
            optika.materials.PeriodicLayerSequence(
                [
                    optika.materials.Layer(
                        chemical="Al",
                        thickness=gamma * d,
                        interface=optika.materials.profiles.ErfInterfaceProfile(
                            width=7 * u.AA,
                        ),
                        kwargs_plot=dict(
                            color="tab:blue",
                            alpha=0.5,
                        ),
                    ),
                    optika.materials.Layer(
                        chemical="Sc",
                        thickness=(1 - gamma) * d,
                        interface=optika.materials.profiles.ErfInterfaceProfile(
                            width=7 * u.AA,
                        ),
                        kwargs_plot=dict(
                            color="tab:orange",
                            alpha=0.5,
                        ),
                    ),
                ],
                num_periods=10,
            ),
        ],
        substrate=optika.materials.Layer(
            chemical=optika.chemicals.Chemical(
                formula="SiO2",
                is_amorphous=True,
            ),
            thickness=30 * u.mm,
            interface=optika.materials.profiles.ErfInterfaceProfile(
                7 * u.AA,
            ),
            kwargs_plot=dict(
                color="gray",
                alpha=0.5,
            ),
        ),
    )


def _multilayer_SiSc(
    period: u.Quantity,
    ratio: float,
    num_periods: int,
    width_Si: u.Quantity,
    width_Sc: u.Quantity,
    substrate: optika.materials.Layer,
    thickness_top: None | u.Quantity = None,
) -> optika.materials.MultilayerMirror:
    """
    Make a Si/Sc multilayer coating, with a Si layer on top under its native oxide.

    Parameters
    ----------
    period
        The thickness of a period, a layer of Si and a layer of Sc.
    ratio
        The thickness of the Sc layer of a period divided by the period.
    num_periods
        The number of periods.
    width_Si
        The width of the erf interface on top of each Si layer.
    width_Sc
        The width of the erf interface on top of each Sc layer.
    substrate
        The substrate the coating is deposited on.
    thickness_top
        The thickness of the top Si layer before it oxidized.
        If :obj:`None` (the default), it is as thick as the other Si layers.
        Growing the oxide consumes :data:`_ratio_oxide` of its thickness of
        the top layer.
    """
    chemical_Si = optika.chemicals.Chemical("Si", is_amorphous=True)
    thickness_Si = (1 - ratio) * period
    if thickness_top is None:
        thickness_top = thickness_Si

    def _Si(thickness: u.Quantity) -> optika.materials.Layer:
        return optika.materials.Layer(
            chemical=chemical_Si,
            thickness=thickness,
            interface=optika.materials.profiles.ErfInterfaceProfile(width_Si),
            kwargs_plot=dict(
                color="tab:blue",
                alpha=0.5,
            ),
        )

    def _Sc() -> optika.materials.Layer:
        return optika.materials.Layer(
            chemical="Sc",
            thickness=ratio * period,
            interface=optika.materials.profiles.ErfInterfaceProfile(width_Sc),
            kwargs_plot=dict(
                color="tab:orange",
                alpha=0.5,
            ),
        )

    layers = [
        optika.materials.Layer(
            chemical=optika.chemicals.Chemical("SiO2", is_amorphous=True),
            thickness=_thickness_oxide,
            interface=optika.materials.profiles.ErfInterfaceProfile(7 * u.AA),
            kwargs_plot=dict(
                color="tab:blue",
                alpha=0.3,
            ),
        ),
        _Si(thickness_top - _ratio_oxide * _thickness_oxide),
        _Sc(),
    ]
    if num_periods > 1:
        layers.append(
            optika.materials.PeriodicLayerSequence(
                [_Si(thickness_Si), _Sc()],
                num_periods=num_periods - 1,
            )
        )

    return optika.materials.MultilayerMirror(
        num_interpolation=esis.optics.num_interpolation,
        layers=layers,
        substrate=substrate,
    )


def multilayer_SiSc() -> optika.materials.MultilayerMirror:
    r"""
    Load the coating design for ESIS II, which uses :math:`\text{Si/Sc}` layers.

    The design has 5 periods of 26 nm, each half Si and half Sc, with a Si
    layer on top, on a glass substrate.
    Ten periods would reflect about 15% more at each of the two lines, but
    five were chosen to keep the coating thin, in case a thicker one is more
    likely to fail.

    The rest of the model follows the sample of
    :func:`multilayer_SiSc_sample_measured`:

    - the Si is amorphous;
    - the top Si layer is under 1.5 nm of its native oxide, which consumed
      0.66 nm of it;
    - the widths of the erf interfaces on top of the Si layers (Sc on Si) and
      on top of the Sc layers (Si on Sc) are those
      :func:`multilayer_SiSc_sample_fit` finds for the sample.

    Examples
    --------
    Plot the efficiency of the coating across the EUV range

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika
        import esis

        # Define an array of wavelengths with which to sample the efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incident rays from the wavelength array
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        )

        # Initialize the multilayer material
        material = esis.flights.f2.optics.materials.multilayer_SiSc()

        # Compute the reflectivity of the multilayer
        reflectivity = material.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the reflectivity vs wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(wavelength, reflectivity, ax=ax);
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("reflectivity");
    """
    Si, Sc = multilayer_SiSc_sample_fit().layers[~0].layers
    return _multilayer_SiSc(
        period=26 * u.nm,
        ratio=0.5,
        num_periods=5,
        width_Si=Si.interface.width,
        width_Sc=Sc.interface.width,
        substrate=optika.materials.Layer(
            chemical=optika.chemicals.Chemical("SiO2", is_amorphous=True),
            thickness=30 * u.mm,
            interface=optika.materials.profiles.ErfInterfaceProfile(7 * u.AA),
            kwargs_plot=dict(
                color="gray",
                alpha=0.5,
            ),
        ),
    )


def multilayer_SiSc_sample_measured() -> optika.materials.MeasuredMirror:
    r"""
    Load the measured reflectivity of a test sample of the :math:`\text{Si/Sc}` coating.

    Eric Gullikson of Lawrence Berkeley National Laboratory measured sample
    ``CX250916B`` on beamline 6.3.2 of the Advanced Light Source, from 25.1 to
    50.0 nm, 5 degrees from the normal, with a beam 0.98 linearly polarized
    (s), and sent the measurement on November 5, 2026.
    It is downloaded from the
    `esis-data-f2 <https://github.com/esis-mission/esis-data-f2/releases/tag/multilayer-v1.0>`_
    repository the first time it is needed, as the file which his software
    saved.

    The sample has 10 periods of Si and Sc, with Sc 0.4 of each period, on a
    Si substrate.
    Its top layer is about 16 nm of Si, under about 1.5 nm of native oxide.
    It was coated on :data:`time_coating_sample` and measured about two months
    later, and only once, so how it ages is not known.

    Examples
    --------
    Load the measurement and plot it as a function of wavelength

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.visualization
        import named_arrays as na
        import esis

        # Load the measurement of the sample
        multilayer = esis.flights.f2.optics.materials.multilayer_SiSc_sample_measured()
        measurement = multilayer.efficiency_measured

        # Plot the measurement as a function of wavelength
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(
                measurement.inputs.wavelength,
                measurement.outputs,
                ax=ax,
                label=measurement.inputs.direction,
            )
            ax.set_xlabel(f"wavelength ({ax.get_xlabel()})");
            ax.set_ylabel("reflectivity");
            ax.legend();
    """
    wavelength, reflectivity = np.loadtxt(
        fname=_path_directory() / "multilayer/mul080572.abs",
        skiprows=1,
        unpack=True,
    )
    wavelength = na.ScalarArray(wavelength << u.nm, axes="wavelength")
    reflectivity = na.ScalarArray(reflectivity, axes="wavelength")

    return optika.materials.MeasuredMirror(
        efficiency_measured=na.FunctionArray(
            inputs=na.SpectralDirectionalVectorArray(
                wavelength=wavelength,
                direction=5 * u.deg,
            ),
            outputs=reflectivity,
        ),
        substrate=optika.materials.Layer(
            chemical="Si",
        ),
    )


def multilayer_SiSc_sample_fit() -> optika.materials.MultilayerMirror:
    r"""
    Fit a model of the sample to :func:`multilayer_SiSc_sample_measured`.

    The model is the stack Eric Gullikson described: 10 periods of amorphous
    Si and Sc, with Sc 0.4 of each period, on a Si substrate, and a top Si
    layer 16 nm thick before 1.5 nm of native oxide grew on it.
    The fit finds the period and the widths of the erf interfaces on top of
    the Si layers and on top of the Sc layers.
    It minimizes the RMS of the logarithm of the ratio of the measured
    reflectivity to that of the model, for the polarization of the beam,
    at the angle of the measurement.

    The last point, at 50.0 nm, is left out.
    It is 29% below the point before it, a drop much steeper than that of
    any of these models.

    Examples
    --------
    Plot the reflectivity of the fit against the measurement.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.visualization
        import named_arrays as na
        import optika
        import esis

        materials = esis.flights.f2.optics.materials

        # Load the measured reflectivity of the sample
        measurement = materials.multilayer_SiSc_sample_measured().efficiency_measured
        angle = measurement.inputs.direction

        # Fit the model of the sample to the measurement
        multilayer = materials.multilayer_SiSc_sample_fit()

        # Compute the reflectivity of the fit, unpolarized
        rays = optika.rays.RayVectorArray(
            wavelength=measurement.inputs.wavelength,
            direction=na.Cartesian3dVectorArray(np.sin(angle), 0, np.cos(angle)),
        )
        reflectivity = multilayer.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the fit against the measurement
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.scatter(
                measurement.inputs.wavelength,
                measurement.outputs,
                ax=ax,
                label="measured",
            );
            na.plt.plot(
                rays.wavelength,
                reflectivity,
                ax=ax,
                label="fit",
                color="tab:orange",
            );
            ax.set_xlabel(f"wavelength ({ax.get_xlabel()})");
            ax.set_ylabel("reflectivity");
            ax.legend();

        # Print the fitted multilayer stack
        multilayer
    """
    return _multilayer_SiSc_sample_fit(versions=_caching.versions(*_packages))


def _multilayer_SiSc_sample(
    period: u.Quantity,
    width_Si: u.Quantity,
    width_Sc: u.Quantity,
) -> optika.materials.MultilayerMirror:
    """
    Make the model of the sample of :func:`multilayer_SiSc_sample_measured`.

    Parameters
    ----------
    period
        The thickness of a period.
    width_Si
        The width of the erf interface on top of each Si layer.
    width_Sc
        The width of the erf interface on top of each Sc layer.
    """
    return _multilayer_SiSc(
        period=period,
        ratio=0.4,
        num_periods=10,
        width_Si=width_Si,
        width_Sc=width_Sc,
        substrate=optika.materials.Layer(
            chemical="Si",
            interface=optika.materials.profiles.ErfInterfaceProfile(7 * u.AA),
            kwargs_plot=dict(
                color="gray",
                alpha=0.5,
            ),
        ),
        thickness_top=16 * u.nm,
    )


@esis.memory.cache
def _multilayer_SiSc_sample_fit(
    versions: dict[str, None | str],
) -> optika.materials.MultilayerMirror:
    """
    Compute :func:`multilayer_SiSc_sample_fit`, and remember it.

    ``versions`` is not read here. It is an argument so that it is part of
    the key the cache is looked up by, so that a new release of any of
    :data:`_packages` fits the measurement again rather than returning the
    fit made with the old release.

    Parameters
    ----------
    versions
        The versions of :data:`_packages`.

    Raises
    ------
    RuntimeError
        If the fit does not converge.
    """
    measurement = multilayer_SiSc_sample_measured().efficiency_measured
    index = dict(wavelength=slice(None, -1))
    wavelength = measurement.inputs.wavelength[index]
    reflectivity = measurement.outputs[index]
    direction = np.cos(measurement.inputs.direction)
    unit = u.nm

    def _multilayer(x: np.ndarray) -> optika.materials.MultilayerMirror:
        period, width_Si, width_Sc = x
        return _multilayer_SiSc_sample(
            period=period * unit,
            width_Si=width_Si * unit,
            width_Sc=width_Sc * unit,
        )

    def _residual(x: np.ndarray) -> np.ndarray:
        multilayer = _multilayer(x)
        r, _ = optika.materials.multilayer_efficiency(
            wavelength=wavelength,
            direction=direction,
            layers=multilayer.layers,
            substrate=multilayer.substrate,
            num_interpolation=multilayer.num_interpolation,
        )
        r = _fraction_s * r.s + (1 - _fraction_s) * r.p
        return np.log(r / reflectivity).ndarray

    fit = scipy.optimize.least_squares(
        fun=_residual,
        x0=[26, 1, 1],
        bounds=([24, 0, 0], [28, 4, 4]),
        diff_step=1e-3,
    )
    if not fit.success:  # pragma: no cover
        raise RuntimeError(f"The fit did not converge: {fit.message}")

    return _multilayer(fit.x)
