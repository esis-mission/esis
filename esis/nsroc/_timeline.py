import dataclasses
import astropy.units as u
import optika

__all__ = [
    "Timeline",
]


@dataclasses.dataclass(repr=False)
class Timeline(
    optika.mixins.Printable,
):
    """
    A Python representation of the NSROC timeline for the ESIS mission.

    All times are relative to the start time of the mission.
    """

    timedelta_esis_start: None | u.Quantity = None
    """The mission time at the start of the ESIS exposure sequence."""

    timedelta_rail_release: None | u.Quantity = None
    """The mission time when the vehicle clears the launch rail."""

    timedelta_terrier_burnout: None | u.Quantity = None
    """The mission time of the Terrier first stage burnout."""

    timedelta_blackbrant_ignition: None | u.Quantity = None
    """The mission time of the Black Brant second stage ignition."""

    timedelta_canard_decouple: None | u.Quantity = None
    """The mission time when the S-19 guidance system releases the canards."""

    timedelta_blackbrant_burnout: None | u.Quantity = None
    """The mission time of the Black Brant second stage burnout."""

    timedelta_despin: None | u.Quantity = None
    """The mission time when the despin sequence is initiated."""

    timedelta_payload_separation: None | u.Quantity = None
    """The mission time when the payload separates from the rest of the vehicle."""

    timedelta_sparcs_enable: None | u.Quantity = None
    """The mission time when the SPARCS pointing system is enabled."""

    timedelta_shutter_open: None | u.Quantity = None
    """The mission time when the payload shutter door is opened."""

    timedelta_nosecone_eject: None | u.Quantity = None
    """The mission time when the nosecone is ejected from the top of the payload."""

    timedelta_sparcs_finemode: None | u.Quantity = None
    """The mission time when SPARCS is predicted to have acquired fine-pointing mode."""

    timedelta_sparcs_rlg_enable: None | u.Quantity = None
    """The mission time when the ring-laser gyroscope is enabled."""

    timedelta_sparcs_rlg_disable: None | u.Quantity = None
    """The mission time when the ring-laser gyroscope is disabled."""

    timedelta_shutter_close: None | u.Quantity = None
    """The mission time when the payload shutter door is closed."""

    timedelta_sparcs_spinup: None | u.Quantity = None
    """The mission time when SPARCS spins up the payload for re-entry."""

    timedelta_sparcs_vent: None | u.Quantity = None
    """The mission time when SPARCS vents leftover propellant."""

    timedelta_ballistic_impact: None | u.Quantity = None
    """The predicted mission time of ballistic impact of the payload."""

    timedelta_sparcs_disable: None | u.Quantity = None
    """The mission time when SPARCS is powered off."""

    timedelta_parachute_deploy: None | u.Quantity = None
    """The mission time when the parachute is deployed."""

    timedelta_payload_impact: None | u.Quantity = None
    """The mission time when the payload impacts the Earth on its parachute."""
