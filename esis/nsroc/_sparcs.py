import dataclasses
import astropy.units as u
import optika

__all__ = [
    "Sparcs",
]


@dataclasses.dataclass(repr=False)
class Sparcs(
    optika.mixins.Printable,
    optika.mixins.Replaceable,
):
    """
    A Python representation of the specified performance of SPARCS.

    The Solar Pointing Attitude Rocket Control System (SPARCS) is the attitude
    control system NSROC provides to point a payload at the Sun.
    Pitch and yaw move the line of sight across the Sun, and roll turns the
    payload about it.
    """

    jitter_pointing: None | u.Quantity = None
    """The peak-to-peak jitter of the pointing in pitch and yaw."""

    drift_pointing: None | u.Quantity = None
    """The rate at which the pointing drifts in pitch and yaw."""

    jitter_roll: None | u.Quantity = None
    """
    The jitter of the roll angle.

    The NASA Sounding Rocket Program Handbook gives the jitter of the
    ring-laser gyroscope which controls the roll only as an upper bound,
    without saying whether it is peak to peak or RMS.
    """

    drift_roll: None | u.Quantity = None
    """The rate at which the roll angle drifts."""
