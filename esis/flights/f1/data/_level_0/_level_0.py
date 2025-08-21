import esis
from ... import nsroc
from .. import path_fits

__all__ = [
    "level_0",
]


def level_0(
    axis_time: str = "time",
    axis_channel: str = "channel",
    axis_x: str = "detector_x",
    axis_y: str = "detector_y",
) -> esis.data.Level_0:
    """
    All the raw images captured during this flight.

    Parameters
    ----------
    axis_time
        The name of the logical axis representing time.
    axis_channel
        The name of the logical axis representing the different ESIS channels.
    axis_x
        The name of the logical axis representing the detector's long axis.
    axis_y
        The name of the logical axis representing the detector's short axis.
    """

    path = path_fits(
        axis_time=axis_time,
        axis_channel=axis_channel,
    )

    return esis.data.Level_0.from_fits(
        path=path,
        camera=esis.optics.Camera(),
        axis_x=axis_x,
        axis_y=axis_y,
        timeline=nsroc.timeline(),
    )
