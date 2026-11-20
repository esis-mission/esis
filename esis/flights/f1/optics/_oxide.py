"""
The growth of the oxide on the silicon carbide on top of the coatings.

The primary mirror and the gratings are both topped with sputtered SiC, which
keeps oxidizing in air after it is coated. Their witness samples were measured
weeks to months before the flight, so the coatings flew with more SiO2 on top
than the measurements saw.

At room temperature, a native oxide grows roughly logarithmically in time,
adding about the same thickness in each decade of time since the coating.
:cite:t:`Schwarcz1994` measured the reflectivity of ion-beam-sputtered SiC at
92.0 nm fall by 25% over 2.5 years, 85% of that in the first three months.
The remaining 15% fell in the one decade of time from three months to
2.5 years. :func:`rate_growth` converts that into the thickness of SiO2 on
SiC, with the optical constants of sputtered SiC measured by
:cite:t:`Kortright1988`, which gives about 0.18 nm per decade.

This is uncertain to about a factor of two. If the first measurement of
:cite:t:`Schwarcz1994` came a day after deposition rather than minutes, the
oxide grew faster in the first months than this law, and the growth after a
measurement made the day after coating could be up to twice as thick.
Part of their loss could also be contamination rather than oxide, which
would make it thinner.
"""

import functools
import numpy as np
import scipy.optimize
import astropy.units as u
import astropy.time
import named_arrays as na
import optika

__all__ = [
    "time_flight",
    "rate_growth",
    "thickness_growth",
]

time_flight = astropy.time.Time("2019-09-30")
"""The day of the flight, after which the coatings stopped oxidizing."""

_wavelength = 920 * u.AA
"""The wavelength at which :cite:t:`Schwarcz1994` measured the aging of SiC."""

_loss = 0.25
"""The relative loss of reflectivity of the SiC after :data:`_timedelta_total`."""

_timedelta_total = 2.5 * u.yr
"""How long :cite:t:`Schwarcz1994` followed the aging of the SiC."""

_fraction_early = 0.85
"""The fraction of :data:`_loss` which happened within :data:`_timedelta_early`."""

_timedelta_early = 0.25 * u.yr
"""The first three months of the aging of the SiC."""

_thickness_native = 1 * u.nm
"""
The thickness of the native oxide on the SiC when it was first measured.

It is the oxide the designs of the coatings put on top of their SiC.
The rate of growth hardly depends on it: from 0.5 to 1.5 nm, the rate changes
by about 5%.
"""


@functools.cache
def rate_growth() -> u.Quantity:
    """
    Compute the thickness of SiO2 grown on sputtered SiC per decade of time.

    The decades are of the time since the SiC was coated. The result is the
    thickness which, added between three months and 2.5 years after the
    coating on top of what grew before, lowers the reflectivity of bulk SiC at
    normal incidence by as much as :cite:t:`Schwarcz1994` measured in that
    time. That interval is one decade.
    """
    silicon_carbide = optika.chemicals.Chemical(
        formula="SiC",
        is_amorphous=True,
        table="kortright",
    )

    rays = optika.rays.RayVectorArray(
        wavelength=_wavelength,
        direction=na.Cartesian3dVectorArray(0, 0, 1),
    )

    normal = na.Cartesian3dVectorArray(0, 0, -1)

    unit = u.nm

    def reflectivity(thickness: float) -> float:
        """
        Compute the reflectivity of bulk SiC under a layer of SiO2.

        Parameters
        ----------
        thickness
            The thickness of the SiO2 in nanometers.
        """
        mirror = optika.materials.MultilayerMirror(
            layers=[
                optika.materials.Layer(
                    chemical="SiO2",
                    thickness=thickness * unit,
                ),
            ],
            substrate=optika.materials.Layer(
                chemical=silicon_carbide,
            ),
        )
        result = mirror.efficiency(rays=rays, normal=normal)
        return float(na.as_named_array(result).ndarray)

    thickness_native = _thickness_native.to_value(unit)
    reflectivity_native = reflectivity(thickness_native)

    def thickness(ratio: float) -> float:
        """
        Find the SiO2 under which the SiC reflects a fraction of its first value.

        Parameters
        ----------
        ratio
            The fraction of the reflectivity when the SiC was first measured.
        """

        def residual(t: float) -> float:
            """
            Compute the difference from the target fraction.

            Parameters
            ----------
            t
                A thickness of SiO2 in nanometers.
            """
            return reflectivity(t) / reflectivity_native - ratio

        return scipy.optimize.brentq(
            f=residual,
            a=thickness_native,
            b=thickness_native + 10,
        )

    thickness_early = thickness(1 - _fraction_early * _loss)
    thickness_total = thickness(1 - _loss)

    decades = np.log10((_timedelta_total / _timedelta_early).to_value(u.one))

    return (thickness_total - thickness_early) * unit / decades


def thickness_growth(
    time_coating: astropy.time.Time,
    time_start: astropy.time.Time,
    time_stop: astropy.time.Time,
) -> u.Quantity:
    """
    Compute the thickness of SiO2 grown on sputtered SiC between two times.

    It is :func:`rate_growth` times the number of decades between the times
    since the coating.

    Parameters
    ----------
    time_coating
        When the SiC was coated.
    time_start
        When the interval starts, such as when the coating was measured.
        It must be after `time_coating`.
    time_stop
        When the interval stops, such as :data:`time_flight`.
        It must not be before `time_start`, since an oxide cannot be removed
        from a coating that was measured with it.

    Raises
    ------
    ValueError
        If `time_start` is not after `time_coating`, or if `time_stop` is
        before `time_start`.
    """
    if time_start <= time_coating:
        raise ValueError(
            f"The interval must start after the coating, on {time_coating.iso}, "
            f"but it starts on {time_start.iso}."
        )
    if time_stop < time_start:
        raise ValueError(
            f"The interval must stop after it starts, on {time_start.iso}, "
            f"but it stops on {time_stop.iso}."
        )
    ratio = (time_stop - time_coating) / (time_start - time_coating)
    return rate_growth() * np.log10(ratio.to_value(u.one))
