import astropy.units as u
import esis

__all__ = [
    "requirements",
]


def requirements() -> esis.optics.Requirements:
    """
    Load the performance of the ESIS optical system required for mission success.

    Examples
    --------
    Load and print the requirements.

    .. jupyter-execute::

        import esis

        requirements = esis.flights.f1.optics.requirements()
        print(requirements)
    """
    return esis.optics.Requirements(
        resolution_spatial=1.5 * u.Mm,
        resolution_spectral=18 * u.km / u.s,
        fov=10 * u.arcmin,
        snr=17.3 * u.dimensionless_unscaled,
        cadence=15 * u.s,
        length_observation=150 * u.s,
    )
