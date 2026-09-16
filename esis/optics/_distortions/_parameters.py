"""The degrees of freedom of the distortion fit, and their serialization."""

from __future__ import annotations
from typing import Any
import copy
import dataclasses
import pathlib
import numpy as np
import astropy.units as u
import astropy.table
import named_arrays as na
import optika
import esis

__all__ = [
    "DistortionParameters",
    "KAPPA_FOCUS",
]

KAPPA_FOCUS = 17.0
"""
The sensor travel per unit of grating travel along the beam that keeps the
image in focus.

For a magnification :math:`m` the longitudinal magnification is :math:`m^2`,
about 16 for ESIS; measured on the as-built model it is 16.9 on every
channel (the O V spot size changes by 58--59 μm rms per mm of grating
travel and by 3.4--3.5 μm per mm of sensor travel, in opposite senses).
:attr:`DistortionParameters.z_sensor` moves the grating by this fraction of
the sensor move, so that it changes the magnification of a channel without
defocusing it.
"""


@dataclasses.dataclass(eq=False, repr=False)
class DistortionParameters(
    optika.mixins.Printable,
):
    """
    The degrees of freedom adjusted when fitting an instrument model to images.

    An instance of this class serves both as a point in parameter space and as
    the prototype that defines the units and structure of the flat vector seen
    by :mod:`scipy.optimize` routines: :func:`named_arrays.pack` flattens it
    into a dimensionless vector and :func:`named_arrays.unpack` rebuilds it.

    Examples
    --------
    Gather the distortion parameters of the ESIS flight-1 design and flatten
    them into a vector suitable for :mod:`scipy.optimize`.

    .. jupyter-execute::

        import named_arrays as na
        import esis

        instrument = esis.flights.f1.optics.design(num_distribution=0)
        parameters = esis.optics.DistortionParameters.from_instrument(instrument)
        na.pack(parameters)
    """

    yaw_grating: u.Quantity | na.AbstractScalar
    """The yaw angle of the diffraction grating."""

    pitch_grating: u.Quantity | na.AbstractScalar
    """The pitch angle of the diffraction grating."""

    roll_grating: u.Quantity | na.AbstractScalar
    """The roll angle of the diffraction grating."""

    roll_field_stop: u.Quantity | na.AbstractScalar
    """The roll angle of the field stop."""

    spacing_rulings: u.Quantity | na.AbstractScalar
    """The constant coefficient of the grating ruling spacing polynomial."""

    displacement_primary: u.Quantity | na.AbstractScalar
    """
    The displacement of the primary mirror along the optic axis relative to
    its nominal focal length.

    A displacement :math:`d` simultaneously sets the focal length to
    :math:`f_\\mathrm{nominal} + d` and the translation of the primary mirror
    to :math:`-d`, so that the primary moves while the focal plane stays put.
    """

    pitch: u.Quantity | na.AbstractScalar
    """The pitch angle of the entire instrument."""

    yaw: u.Quantity | na.AbstractScalar
    """The yaw angle of the entire instrument."""

    roll: u.Quantity | na.AbstractScalar
    """
    The roll angle of the entire instrument.

    In the flight-1 model this is defined per channel as the offset from that
    channel's grating azimuth, so it is a grating placement term rather than
    a quantity shared by the channels.
    """

    z_sensor: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The displacement of the sensor along its normal, with the grating
    following along the beam by :obj:`KAPPA_FOCUS` times less so that the
    image stays in focus: a change of the magnification of one channel
    without a change of its focus.

    Together with the sensor rotations and in-plane translations below this
    lets a channel's mapping differ from another's by a scale, an
    anisotropy and a rotation, which is what the inter-channel alignment
    measures and which none of the parameters above can produce.
    """

    roll_sensor: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The roll of the sensor about its normal (a rotation of the image)."""

    pitch_sensor: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The pitch of the sensor (a foreshortening of the image along one axis)."""

    yaw_sensor: u.Quantity | na.AbstractScalar = 0 * u.deg
    """The yaw of the sensor (a foreshortening of the image along the other axis)."""

    x_sensor: u.Quantity | na.AbstractScalar = 0 * u.mm
    """The in-plane translation of the sensor along its :math:`x` axis."""

    y_sensor: u.Quantity | na.AbstractScalar = 0 * u.mm
    """The in-plane translation of the sensor along its :math:`y` axis."""

    @classmethod
    def from_instrument(
        cls,
        instrument: esis.optics.abc.AbstractInstrument,
    ) -> DistortionParameters:
        """
        Gather the current distortion parameters of the given instrument.

        The parameters are converted to a canonical set of units
        (:obj:`~astropy.units.arcmin` for the grating angles,
        :obj:`~astropy.units.arcsec` for the instrument pointing, etc.)
        so that the components of the packed vector are of order unity
        and bounds built from different instances are consistent.

        Parameters
        ----------
        instrument
            The instrument model to gather the parameters from.
        """
        primary_mirror = instrument.primary_mirror
        sensor = instrument.camera.sensor
        return cls(
            yaw_grating=instrument.grating.yaw.to(u.arcmin),
            pitch_grating=instrument.grating.pitch.to(u.arcmin),
            roll_grating=instrument.grating.roll.to(u.deg),
            roll_field_stop=instrument.field_stop.roll.to(u.deg),
            spacing_rulings=instrument.grating.rulings.spacing.coefficients[0].to(u.um),
            displacement_primary=-primary_mirror.translation.z.to(u.mm),
            pitch=instrument.pitch.to(u.arcsec),
            yaw=instrument.yaw.to(u.arcsec),
            roll=instrument.roll.to(u.deg),
            z_sensor=(sensor.translation.z - _design(instrument, "z_sensor")).to(u.mm),
            roll_sensor=sensor.roll.to(u.deg),
            pitch_sensor=sensor.pitch.to(u.deg),
            yaw_sensor=sensor.yaw.to(u.deg),
            x_sensor=sensor.translation.x.to(u.mm),
            y_sensor=sensor.translation.y.to(u.mm),
        )

    def to_instrument(
        self,
        instrument: esis.optics.abc.AbstractInstrument,
    ) -> esis.optics.abc.AbstractInstrument:
        """
        Apply these parameters to a copy of the given instrument.

        The given instrument is left unmodified, and any cached optical
        system on the result is discarded so that it is rebuilt with the
        new parameters.

        The nominal focal length of the primary mirror is recovered from the
        invariant ``focal_length + translation.z``, which is unchanged by
        applying a :attr:`displacement_primary`, so this method may be applied
        repeatedly to the results of previous applications.

        Parameters
        ----------
        instrument
            The instrument model to apply the parameters to.
        """
        result = copy.copy(instrument)

        # discard the cached system before the deep copy so that the
        # (potentially large) raytrace results are not copied
        result.__dict__.pop("system", None)

        result = copy.deepcopy(result)

        # copy the parameter values too, so that the result does not alias
        # this object's arrays (mutating one must not silently change the
        # other)
        p = copy.deepcopy(self)

        primary_mirror = result.primary_mirror
        focal_length_nominal = (
            primary_mirror.sag.focal_length + primary_mirror.translation.z
        )

        result.grating.yaw = p.yaw_grating
        result.grating.pitch = p.pitch_grating
        result.grating.roll = p.roll_grating
        result.field_stop.roll = p.roll_field_stop
        result.grating.rulings.spacing.coefficients[0] = p.spacing_rulings
        primary_mirror.sag.focal_length = focal_length_nominal + p.displacement_primary
        primary_mirror.translation.z = -p.displacement_primary
        result.pitch = p.pitch
        result.yaw = p.yaw
        result.roll = p.roll

        # the sensor terms are offsets from the placement the instrument was
        # built with, so that repeated applications do not accumulate; the
        # compound focus-preserving move shifts the grating along with it
        sensor = result.camera.sensor
        sensor.translation.z = _design(instrument, "z_sensor") + p.z_sensor
        result.grating.translation.z = (
            _design(instrument, "z_grating") + p.z_sensor / KAPPA_FOCUS
        )
        sensor.roll = p.roll_sensor
        sensor.pitch = p.pitch_sensor
        sensor.yaw = p.yaw_sensor
        sensor.translation.x = p.x_sensor
        sensor.translation.y = p.y_sensor
        for name in ("z_sensor", "z_grating"):
            result.__dict__[f"_design_{name}"] = _design(instrument, name)

        return result

    def to_file(
        self,
        path: str | pathlib.Path,
        metadata: None | dict[str, Any] = None,
    ) -> None:
        """
        Save these parameters as a plain-text ECSV table.

        The fields become the columns of the table (with their units), and
        the elements along the (single) logical axis of the parameters become
        its rows, so that fit results can be committed to version control and
        reviewed as text. Scalar fields are broadcast along the axis.

        Parameters
        ----------
        path
            The path of the file to write.
        metadata
            Additional provenance recorded in the table header, for example
            the date and configuration of the fit that produced the
            parameters.

        Raises
        ------
        ValueError
            If the parameters have more than one logical axis, or if
            `metadata` contains the reserved key ``"axis"``, which records
            the name of the logical axis for :meth:`from_file`.

        See Also
        --------
        from_file : The inverse of this method.
        """
        shape = na.shape(self)
        if len(shape) > 1:
            raise ValueError(
                f"only parameters with at most one axis can be saved, " f"got {shape=}"
            )
        axis = next(iter(shape), None)
        num = shape.get(axis, 1)

        columns = {}
        for field in dataclasses.fields(self):
            value = na.as_named_array(getattr(self, field.name))
            unit = na.unit(value)
            data = na.value(value).ndarray
            if unit is not None:
                data = data * unit
            columns[field.name] = np.broadcast_to(data, (num,), subok=True)

        table = astropy.table.QTable(columns)
        if metadata is not None and "axis" in metadata:
            raise ValueError(
                f"the metadata key 'axis' is reserved for the name of the "
                f"logical axis, got {metadata['axis']=}"
            )
        table.meta["axis"] = axis
        if metadata is not None:
            table.meta.update(metadata)

        table.write(path, format="ascii.ecsv", overwrite=True)

    @classmethod
    def from_file(
        cls,
        path: str | pathlib.Path,
        axis: None | str = None,
    ) -> DistortionParameters:
        """
        Load parameters saved by :meth:`to_file`.

        The logical axis of the parameters is recovered from the ``axis``
        entry of the table header.

        Parameters
        ----------
        path
            The path of the file to read.
        axis
            The name to use for the logical axis of the parameters,
            overriding the name recorded in the file.
        """
        table = astropy.table.QTable.read(path, format="ascii.ecsv")
        if axis is None:
            axis = table.meta["axis"]

        fields = {}
        for field in dataclasses.fields(cls):
            if field.name not in table.colnames:
                # written before this field existed: the default applies
                continue
            column = table[field.name]
            if axis is None:
                fields[field.name] = column[0]
            else:
                fields[field.name] = (
                    na.ScalarArray(np.asarray(column.value), axes=axis) * column.unit
                )

        return cls(**fields)


def _design(instrument: esis.optics.abc.AbstractInstrument, name: str):
    """
    Return the placement the instrument was built with, before any sensor term.

    An instrument returned by :meth:`DistortionParameters.to_instrument`
    remembers the values it was built from, so that applying parameters to
    it again measures the sensor terms from the same origin.

    Raises
    ------
    ValueError
        If `name` is not a placement this function knows.
    """
    if f"_design_{name}" in instrument.__dict__:
        return instrument.__dict__[f"_design_{name}"]
    if name == "z_sensor":
        return instrument.camera.sensor.translation.z
    if name == "z_grating":
        return instrument.grating.translation.z
    raise ValueError(name)  # pragma: nocover
