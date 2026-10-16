import astropy.units as u
import esis

__all__ = [
    "sparcs",
]


def sparcs() -> esis.nsroc.Sparcs:
    """
    Load the specified performance of SPARCS for this flight.

    The jitter of the pointing is the specification of SPARCS VII, and its
    drift is the rate NSROC gave for ESIS in 2016.
    The NASA Sounding Rocket Program Handbook (810-HB-SRP, 2015, Section
    5.12.1.1) gives no drift, and says only that the pointing is stable to
    tenths of an arcsecond.

    In fine-pointing mode, SPARCS controls the roll with a ring-laser
    gyroscope (:attr:`esis.nsroc.Timeline.timedelta_sparcs_rlg_enable`),
    whose stability and jitter are those given for it by the same section
    of the handbook.

    Examples
    --------
    Load and print this specification.

    .. jupyter-execute::

        import esis

        esis.flights.f1.nsroc.sparcs()
    """
    return esis.nsroc.Sparcs(
        jitter_pointing=0.5 * u.arcsec,
        drift_pointing=0.014 * u.arcsec / u.s,
        jitter_roll=0.01 * u.deg,
        drift_roll=0.04 * u.deg / u.hr,
    )
