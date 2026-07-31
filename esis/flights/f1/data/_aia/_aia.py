import numpy as np
import astropy.units as u
import named_arrays as na
import esis
from .. import level_1

__all__ = [
    "aia_context",
]


@esis.memory.cache
def aia_context(
    wavelength: u.Quantity = [304, 171, 193] * u.AA,
    halfwidth: u.Quantity = 700 * u.arcsec,
    margin: u.Quantity = 30 * u.s,
    axis_time: str = "time",
    axis_x: str = "detector_x",
    axis_y: str = "detector_y",
    limit: None | int = None,
) -> dict[str, na.FunctionArray]:
    """
    Load co-temporal AIA context images of the ESIS field of view.

    For each requested channel, the AIA filtergrams captured during the
    flight are downloaded from the JSOC (cached to disk, so the queries are
    only performed on the first call), cropped to a box around the center of
    the ESIS field of view, and repackaged on fixed helioprojective
    coordinates: the WCS of the first frame is frozen into one-dimensional
    vertex arrays (the AIA pointing drift and roll over the five-minute
    flight are far below the 0.6 arcsec plate scale).

    The result is suitable as the ``context`` argument of
    :meth:`esis.data.Level_4.animate_event`, since the distortion fit
    registers the Level-4 scene coordinates to the AIA helioprojective
    frame.

    Parameters
    ----------
    wavelength
        The AIA channels to load.
    halfwidth
        The half-width of the box around the ESIS field-of-view center to
        keep.
    margin
        How far beyond the first and last ESIS exposures to search for AIA
        images.
    axis_time
        The logical axis corresponding to changes in time.
    axis_x
        The logical axis corresponding to changes in the horizontal
        coordinate.
    axis_y
        The logical axis corresponding to changes in the vertical
        coordinate.
    limit
        The maximum number of files to download per channel.
    """
    import sdo

    l1 = level_1()

    time_start = l1.inputs.time_start[{l1.axis_time: 0}].ndarray.mean() - margin
    time_stop = l1.inputs.time_end[{l1.axis_time: ~0}].ndarray.mean() + margin

    # the center of the ESIS field of view in the AIA helioprojective frame,
    # from the distortion fit the Level-4 scene coordinates inherit
    center_x = 17 * u.arcsec
    center_y = -21 * u.arcsec

    result = {}
    for w in wavelength:
        obs = sdo.aia.open(
            time_start=time_start,
            time_stop=time_stop,
            wavelength=w,
            axis_time=axis_time,
            axis_detector_x=axis_x,
            axis_detector_y=axis_y,
            limit=limit,
        )
        obs = obs[dict(wavelength=0)]

        position = obs.inputs[{axis_time: 0}].position
        x = position.x[{axis_y: 0}].ndarray.to(u.arcsec)
        y = position.y[{axis_x: 0}].ndarray.to(u.arcsec)

        slice_x = slice(
            int(np.searchsorted(x, center_x - halfwidth)),
            int(np.searchsorted(x, center_x + halfwidth)) + 1,
        )
        slice_y = slice(
            int(np.searchsorted(y, center_y - halfwidth)),
            int(np.searchsorted(y, center_y + halfwidth)) + 1,
        )

        outputs = obs.outputs[
            {
                axis_x: slice(slice_x.start, slice_x.stop - 1),
                axis_y: slice(slice_y.start, slice_y.stop - 1),
            }
        ]

        label = f"AIA {w.to_value(u.AA):.0f}"
        result[label] = na.FunctionArray(
            inputs=na.TemporalPositionalVectorArray(
                time=obs.inputs.time,
                position=na.Cartesian2dVectorArray(
                    x=na.ScalarArray(x[slice_x], axes=(axis_x,)),
                    y=na.ScalarArray(y[slice_y], axes=(axis_y,)),
                ),
            ),
            outputs=na.ScalarArray(
                ndarray=np.asarray(outputs.ndarray, dtype=np.float32),
                axes=outputs.axes,
            ),
        )

    return result
