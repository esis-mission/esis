from __future__ import annotations
from typing import Any, Sequence
import copy
import dataclasses
import datetime
import itertools
import json
import pathlib
import time
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import optika
import esis

__all__ = [
    "DistortionParameters",
    "DistortionObjective",
    "DistortionResidual",
    "ConvergenceLogger",
    "fit_distortion",
    "fit_distortion_scan",
    "fit_distortion_series",
]


@dataclasses.dataclass(repr=False)
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
    """The roll angle of the entire instrument."""

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

        primary_mirror = result.primary_mirror
        focal_length_nominal = (
            primary_mirror.sag.focal_length + primary_mirror.translation.z
        )

        result.grating.yaw = self.yaw_grating
        result.grating.pitch = self.pitch_grating
        result.grating.roll = self.roll_grating
        result.field_stop.roll = self.roll_field_stop
        result.grating.rulings.spacing.coefficients[0] = self.spacing_rulings
        primary_mirror.sag.focal_length = (
            focal_length_nominal + self.displacement_primary
        )
        primary_mirror.translation.z = -self.displacement_primary
        result.pitch = self.pitch
        result.yaw = self.yaw
        result.roll = self.roll

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
            If the parameters have more than one logical axis.

        See Also
        --------
        from_file : The inverse of this method.
        """
        import astropy.table

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
        import astropy.table

        table = astropy.table.QTable.read(path, format="ascii.ecsv")
        if axis is None:
            axis = table.meta["axis"]

        fields = {}
        for field in dataclasses.fields(cls):
            column = table[field.name]
            if axis is None:
                fields[field.name] = column[0]
            else:
                fields[field.name] = (
                    na.ScalarArray(np.asarray(column.value), axes=axis) * column.unit
                )

        return cls(**fields)


@dataclasses.dataclass(repr=False)
class DistortionObjective(
    optika.mixins.Printable,
):
    """
    An objective function which compares modeled images to an observation.

    Instances of this class are intended to be passed directly to
    :mod:`scipy.optimize` routines such as
    :func:`scipy.optimize.differential_evolution`: they are callables of a
    flat, dimensionless parameter vector, and they are picklable so that the
    optimizer can evaluate them in parallel worker processes.

    Each evaluation unpacks the vector into a :class:`DistortionParameters`
    using :attr:`parameters` as the prototype, applies it to a copy of
    :attr:`instrument`, images :attr:`scene` through the resulting optical
    system, and compares the result to :attr:`observation` using the Pearson
    correlation coefficient. A penalty proportional to the squared distance
    between the mean ray position and the origin of the sensor discourages
    solutions that walk the image off the detector.

    Note that the imaging model samples photons randomly, so evaluations of
    this objective are stochastic even for identical parameter vectors, and
    it should only be used with optimizers that tolerate a noisy objective,
    such as :func:`scipy.optimize.differential_evolution`.
    """

    instrument: esis.optics.abc.AbstractInstrument
    """The instrument model being fit to the observation."""

    parameters: DistortionParameters
    """
    The prototype which defines the units and structure of the flat
    parameter vector, usually the initial guess of the fit.
    """

    scene: na.FunctionArray
    """
    The spectral radiance of the scene as a function of wavelength and field
    position, imaged through the instrument on every evaluation.
    """

    observation: na.AbstractScalar
    """
    The observed image that the modeled image is compared against.

    Any axes of the modeled image which are not present in this array are
    summed over before comparing.
    """

    pupil: None | na.AbstractCartesian2dVectorArray = None
    """The vertices of the pupil grid used to image the scene."""

    axis_wavelength: None | str = None
    """The logical axis of the scene corresponding to changing wavelength."""

    axis_field: None | tuple[str, str] = None
    """The logical axes of the scene corresponding to changing field position."""

    axis_channel: None | str = None
    """
    The logical axis of the observation corresponding to changing camera
    channel.

    If given, the correlation and the off-target penalty are computed
    independently for each channel and then averaged, so that brightness
    differences between the channels (e.g. from differing gains) do not
    suppress the correlation. This is the appropriate setting when fitting
    several channels at once with shared parameters.
    """

    weight_correlation: float = 1000
    """The weight of the correlation term relative to the distance penalty."""

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate the objective for a flat, dimensionless parameter vector.

        Parameters
        ----------
        x
            The flat parameter vector, interpreted in the units and structure
            of :attr:`parameters`.
        """
        parameters = na.unpack(x, self.parameters)
        instrument = parameters.to_instrument(self.instrument)
        system = instrument.system

        image = system.image(
            scene=self.scene,
            pupil=self.pupil,
            axis_wavelength=self.axis_wavelength,
            axis_field=self.axis_field,
            noise=False,
        )

        image = na.value(image.outputs)
        observation = na.value(self.observation)

        axis_extra = tuple(set(na.shape(image)) - set(na.shape(observation)))
        if axis_extra:
            image = image.sum(axis=axis_extra)

        axis_image = tuple(set(na.shape(observation)) - {self.axis_channel})
        correlation = _correlation(image, observation, axis=axis_image).mean()

        position = system.rayfunction_default.outputs.position.to(u.mm)
        axis_position = tuple(set(na.shape(position)) - {self.axis_channel})
        distance = np.square(position.mean(axis_position).length).mean()

        result = -self.weight_correlation * correlation + distance.value

        return float(na.value(result).ndarray)


@dataclasses.dataclass(repr=False)
class DistortionResidual(
    optika.mixins.Printable,
):
    r"""
    A least-squares residual comparing modeled images to an observation.

    This is the residual-vector counterpart of :class:`DistortionObjective`,
    suitable for :func:`scipy.optimize.least_squares`: each evaluation returns
    a flat vector of per-pixel residuals rather than a single scalar, which
    lets a Gauss-Newton optimizer exploit the structure of the problem.

    The residual is built from the same comparison as
    :class:`DistortionObjective`. For images standardized to zero mean and unit
    deviation, the sum of squared residuals is a monotonic function of the
    Pearson correlation (:math:`\lVert\hat a - \hat b\rVert^2 = 2N(1 -
    \mathrm{corr})`), so minimizing it maximizes the correlation. The squared
    off-target penalty of :class:`DistortionObjective` is appended as extra
    residual rows.

    Two features make this objective usable with a derivative-based optimizer,
    where :class:`DistortionObjective` is not:

    * **Determinism.** The imaging model jitters each ray within its scene cell
      using the global :mod:`numpy` random state, so it is stochastic by
      default. Re-seeding that state with :attr:`seed` before every evaluation
      freezes the jitter realization, making the residual a deterministic
      function of the parameters that finite-difference Jacobians can read.
      Note that this mutates the global :mod:`numpy` random state as a side
      effect; it is intended for serial use.
    * **Smoothing.** The images are sparse, so a sub-pixel shift can leave the
      residual unchanged until a feature crosses a pixel boundary. Convolving
      the modeled image with a :attr:`sigma_psf`-wide Gaussian point-spread
      function widens the features so that small parameter changes produce a
      readable gradient, while also modeling the blur that the real optics
      imprint on the observation. A :attr:`smoothing`-wide box filter applied
      to both images is retained as a purely numerical alternative.

    Notes
    -----
    In practice, :func:`scipy.optimize.least_squares` was found to
    under-converge on this residual for realistic scenes even with the
    features above: the smoothed correlation surface remains locally flat
    enough that the optimizer terminates after moving a small fraction of the
    distance to the true peak. :func:`fit_distortion_scan`, which reads the
    shape of the merit basin directly instead of differentiating it, is the
    recommended driver.
    """

    instrument: esis.optics.abc.AbstractInstrument
    """The instrument model being fit to the observation."""

    parameters: DistortionParameters
    """
    The prototype which defines the units and structure of the flat
    parameter vector, usually the initial guess of the fit.
    """

    scene: na.FunctionArray
    """
    The spectral radiance of the scene as a function of wavelength and field
    position, imaged through the instrument on every evaluation.
    """

    observation: na.AbstractScalar
    """
    The observed image that the modeled image is compared against.

    Any axes of the modeled image which are not present in this array are
    summed over before comparing.
    """

    pupil: None | na.AbstractCartesian2dVectorArray = None
    """The vertices of the pupil grid used to image the scene."""

    axis_wavelength: None | str = None
    """The logical axis of the scene corresponding to changing wavelength."""

    axis_field: None | tuple[str, str] = None
    """The logical axes of the scene corresponding to changing field position."""

    axis_channel: None | str = None
    """
    The logical axis of the observation corresponding to changing camera
    channel.

    If given, each channel is standardized independently before the residual
    is formed, so that brightness differences between the channels do not
    dominate the fit. See :attr:`DistortionObjective.axis_channel`.
    """

    weight_correlation: float = 1000
    """The weight of the correlation residual relative to the off-target penalty."""

    weight_distance: float = 1.0
    """The weight of the off-target penalty residual."""

    smoothing: None | int = 1
    """
    The width, in detector pixels, of the box filter applied to both images
    before comparing. A value of :obj:`None` or :math:`\\leq 1` disables
    smoothing.
    """

    sigma_psf: None | float = None
    """
    The standard deviation, in detector pixels, of a Gaussian point-spread
    function convolved with the modeled image only.

    The observation already carries the blur of the real optics, so applying
    the point-spread function to the modeled image makes the two directly
    comparable while also letting a sparsely-sampled raytraced image produce
    a smooth, differentiable residual. A value of :obj:`None` disables the
    convolution.
    """

    seed: int = 0
    """
    The seed used to freeze the imaging model's random ray jitter, so that the
    residual is a deterministic function of the parameters.
    """

    def _smooth(
        self,
        image: na.AbstractScalar,
        axis_image: tuple[str, ...],
    ) -> na.AbstractScalar:
        return _smooth_box(image, self.smoothing, axis_image)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the residual vector for a flat, dimensionless parameter vector.

        Parameters
        ----------
        x
            The flat parameter vector, interpreted in the units and structure
            of :attr:`parameters`.
        """
        parameters = na.unpack(x, self.parameters)
        instrument = parameters.to_instrument(self.instrument)
        system = instrument.system

        # freeze the per-cell ray jitter so the residual is deterministic
        np.random.seed(self.seed)
        image = system.image(
            scene=self.scene,
            pupil=self.pupil,
            axis_wavelength=self.axis_wavelength,
            axis_field=self.axis_field,
            noise=False,
        )

        image = na.value(image.outputs)
        observation = na.value(self.observation)

        axis_extra = tuple(set(na.shape(image)) - set(na.shape(observation)))
        if axis_extra:
            image = image.sum(axis=axis_extra)

        axis_image = tuple(set(na.shape(observation)) - {self.axis_channel})

        if self.sigma_psf is not None:
            kernel = _kernel_gaussian(self.sigma_psf, axis_image)
            image = na.convolve(image, kernel, axis=axis_image)

        model = _standardize(self._smooth(image, axis_image), axis_image)
        data = _standardize(self._smooth(observation, axis_image), axis_image)

        num_pixel = 1
        for ax in axis_image:
            num_pixel *= na.shape(observation)[ax]
        residual_correlation = (model - data) * np.sqrt(
            self.weight_correlation / num_pixel
        )

        # off-target penalty: the components of the mean ray position, so that
        # the sum of their squares reproduces the squared-distance penalty.
        position = system.rayfunction_default.outputs.position.to(u.mm)
        axis_position = tuple(set(na.shape(position)) - {self.axis_channel})
        mean_position = position.mean(axis_position)
        num_channel = na.shape(position).get(self.axis_channel, 1)
        residual_distance = na.stack(
            arrays=[mean_position.x, mean_position.y, mean_position.z],
            axis="_component",
        )
        residual_distance = na.value(residual_distance) * np.sqrt(
            self.weight_distance / num_channel
        )

        return np.concatenate(
            [
                na.value(residual_correlation).ndarray.reshape(-1),
                na.value(residual_distance).ndarray.reshape(-1),
            ]
        )


def _kernel_gaussian(
    sigma: float,
    axis: tuple[str, ...],
) -> na.ScalarArray:
    """
    Build a normalized Gaussian convolution kernel.

    Parameters
    ----------
    sigma
        The standard deviation of the Gaussian in pixels.
    axis
        The logical axes of the kernel, one per dimension.
    """
    radius = max(2, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    profile = np.exp(-np.square(x / sigma) / 2)
    kernel = profile
    for _ in range(len(axis) - 1):
        kernel = np.multiply.outer(kernel, profile)
    kernel = kernel / kernel.sum()
    return na.ScalarArray(ndarray=kernel, axes=axis)


def _standardize(
    a: na.AbstractScalar,
    axis: None | tuple[str, ...] = None,
) -> na.AbstractScalar:
    """
    Shift and scale an array to zero mean and unit standard deviation.

    A constant array is left at zero, so that it contributes nothing to a
    correlation or a normalized residual.

    Parameters
    ----------
    a
        The array to standardize.
    axis
        The logical axes along which to compute the mean and deviation.
        If :obj:`None`, every axis is used.
    """
    a = a - a.mean(axis)
    deviation = a.std(axis)
    return a / np.where(deviation == 0, 1, deviation)


def _correlation(
    a: na.AbstractScalar,
    b: na.AbstractScalar,
    axis: None | tuple[str, ...] = None,
) -> na.AbstractScalar:
    """
    Compute the Pearson correlation coefficient of two arrays.

    Parameters
    ----------
    a
        The first array, standardized before comparing.
        If it is constant, the correlation is zero.
    b
        The second array, standardized before comparing.
        If it is constant, the correlation is zero.
    axis
        The logical axes along which to compute the correlation.
        If :obj:`None`, the correlation is computed over every axis.
    """
    return (_standardize(a, axis) * _standardize(b, axis)).mean(axis)


def _smooth_box(
    a: na.AbstractScalar,
    size: None | int,
    axis: tuple[str, ...],
) -> na.AbstractScalar:
    r"""
    Smooth an array with a normalized box filter.

    Parameters
    ----------
    a
        The array to smooth.
    size
        The width of the box in pixels. A value of :obj:`None` or
        :math:`\leq 1` disables smoothing.
    axis
        The logical axes along which to smooth.
    """
    if size is None or size <= 1:
        return a
    size = int(size)
    kernel = na.ScalarArray(
        ndarray=np.ones(len(axis) * (size,)) / size ** len(axis),
        axes=axis,
    )
    return na.convolve(a, kernel, axis=axis)


def _peak_parabola(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Locate the peak of a sampled curve with a parabola refinement.

    A parabola is fit through the best sample and its two neighbors, and the
    abscissa of its vertex is returned, clipped to the sampled interval.
    If the parabola is not concave, the abscissa of the best sample is
    returned instead.

    Parameters
    ----------
    x
        The sample positions.
    y
        The sampled curve values.
    """
    i = int(np.argmax(y))
    i = min(max(i, 1), len(x) - 2)
    c = np.polyfit(x[i - 1 : i + 2], y[i - 1 : i + 2], 2)
    if c[0] >= 0:
        return float(x[int(np.argmax(y))])
    return float(np.clip(-c[1] / (2 * c[0]), x.min(), x.max()))


def _correlation_model(
    instrument: esis.optics.abc.AbstractInstrument,
    parameters: DistortionParameters,
    scene: na.FunctionArray,
    observation: na.AbstractScalar,
    pupil: None | na.AbstractCartesian2dVectorArray,
    axis_wavelength: None | str,
    axis_field: None | tuple[str, str],
    axis_channel: None | str,
    smoothing: None | int,
    sigma_psf: None | float,
    seed: int,
) -> na.AbstractScalar:
    """
    Compute the correlation between the modeled image and the observation.

    Applies `parameters` to `instrument`, images `scene` deterministically,
    optionally convolves the modeled image with a Gaussian point-spread
    function, and computes the Pearson correlation against `observation`
    (per channel, if `axis_channel` is given).

    Parameters
    ----------
    instrument
        The instrument model being fit to the observation.
    parameters
        The distortion parameters applied to the instrument.
    scene
        The spectral radiance of the scene imaged through the instrument.
    observation
        The observed image that the modeled image is compared against.
    pupil
        The vertices of the pupil grid used to image the scene.
    axis_wavelength
        The logical axis of the scene corresponding to changing wavelength.
    axis_field
        The logical axes of the scene corresponding to changing field position.
    axis_channel
        The logical axis of the observation corresponding to changing camera
        channel.
    smoothing
        The width, in detector pixels, of a box filter applied to both images.
    sigma_psf
        The standard deviation, in detector pixels, of a Gaussian point-spread
        function convolved with the modeled image only.
    seed
        The seed used to freeze the imaging model's random ray jitter.
    """
    instrument = parameters.to_instrument(instrument)

    # freeze the per-cell ray jitter so the correlation is deterministic
    np.random.seed(seed)
    image = instrument.system.image(
        scene=scene,
        pupil=pupil,
        axis_wavelength=axis_wavelength,
        axis_field=axis_field,
        noise=False,
    )

    image = na.value(image.outputs)
    observation = na.value(observation)

    axis_extra = tuple(set(na.shape(image)) - set(na.shape(observation)))
    if axis_extra:
        image = image.sum(axis=axis_extra)

    axis_image = tuple(set(na.shape(observation)) - {axis_channel})

    if sigma_psf is not None:
        kernel = _kernel_gaussian(sigma_psf, axis_image)
        image = na.convolve(image, kernel, axis=axis_image)

    image = _smooth_box(image, smoothing, axis_image)
    observation = _smooth_box(observation, smoothing, axis_image)

    return _correlation(image, observation, axis_image)


def fit_distortion(
    instrument: esis.optics.abc.AbstractInstrument,
    scene: na.FunctionArray,
    observation: na.AbstractScalar,
    bounds: tuple[DistortionParameters, DistortionParameters],
    parameters: None | DistortionParameters = None,
    pupil: None | na.AbstractCartesian2dVectorArray = None,
    axis_wavelength: None | str = None,
    axis_field: None | tuple[str, str] = None,
    axis_channel: None | str = None,
    weight_correlation: float = 1000,
    directory: None | pathlib.Path = None,
    kwargs_optimizer: None | dict[str, Any] = None,
) -> DistortionParameters:
    """
    Fit the distortion parameters of an instrument to an observed image.

    Wraps :func:`scipy.optimize.differential_evolution` around a
    :class:`DistortionObjective`: the given `parameters` are flattened with
    :func:`named_arrays.pack` to seed the optimizer, the `bounds` are
    flattened the same way, and the best solution found is unpacked back
    into a :class:`DistortionParameters`, which can be applied to the
    instrument with :meth:`DistortionParameters.to_instrument`.

    Parameters
    ----------
    instrument
        The instrument model to fit, usually reduced to a single channel.
    scene
        The spectral radiance of the scene as a function of wavelength and
        field position, imaged through the instrument on every evaluation.
    observation
        The observed image that the modeled images are compared against.
        Any axes of the modeled image which are not present in this array
        are summed over before comparing.
    bounds
        The lower and upper bounds of the fit, expressed in the same units
        as `parameters`.
    parameters
        The initial guess of the fit, which also defines the units and
        structure of the parameter vector seen by the optimizer.
        If :obj:`None`, the current parameters of `instrument` are used.
    pupil
        The vertices of the pupil grid used to image the scene.
    axis_wavelength
        The logical axis of the scene corresponding to changing wavelength.
    axis_field
        The logical axes of the scene corresponding to changing field position.
    axis_channel
        The logical axis of the observation corresponding to changing camera
        channel, used to compare each channel independently when fitting
        several channels at once with shared parameters.
        See :attr:`DistortionObjective.axis_channel`.
    weight_correlation
        The weight of the correlation term of the objective relative to its
        off-target distance penalty.
    directory
        A directory where the convergence history is logged using a
        :class:`ConvergenceLogger`.
        If :obj:`None`, the fit is not logged.
    kwargs_optimizer
        Additional keyword arguments passed to
        :func:`scipy.optimize.differential_evolution`, for example
        ``dict(workers=-1, popsize=40)``.

    Examples
    --------
    Fit each channel of the ESIS flight-1 design to the Level-1 frame that
    :func:`esis.flights.f1.optics.distortion_fit` was optimized against.

    .. code-block:: python

        import pathlib
        import named_arrays as na
        import esis

        obs = esis.flights.f1.data.level_1()[dict(time=15)]
        scene = esis.flights.f1.data.synth.scene_aia()[dict(time=15)]

        instrument = esis.flights.f1.optics.design(num_distribution=0)

        for i in range(obs.shape[instrument.axis_channel]):
            channel = instrument[dict(channel=i)]
            parameters = esis.optics.DistortionParameters.from_instrument(channel)

            # the scene's wavelength coordinate varies within each spectral
            # line along "velocity"; its per-line axis ("wavelength") is
            # absent from the observation and therefore summed over
            fitted = esis.optics.fit_distortion(
                instrument=channel,
                scene=scene,
                observation=obs.outputs[dict(channel=i)].value,
                bounds=esis.flights.f1.optics.distortion_fit_bounds(parameters),
                parameters=parameters,
                axis_wavelength="velocity",
                axis_field=("detector_x", "detector_y"),
                directory=pathlib.Path(f"distortion_fit/channel_{i}"),
                kwargs_optimizer=dict(workers=-1, popsize=40),
            )

            model = fitted.to_instrument(channel)
    """
    if parameters is None:
        parameters = DistortionParameters.from_instrument(instrument)

    if kwargs_optimizer is None:
        kwargs_optimizer = dict()

    objective = DistortionObjective(
        instrument=instrument,
        parameters=parameters,
        scene=scene,
        observation=observation,
        pupil=pupil,
        axis_wavelength=axis_wavelength,
        axis_field=axis_field,
        axis_channel=axis_channel,
        weight_correlation=weight_correlation,
    )

    lower, upper = bounds

    callback = None
    if directory is not None:
        callback = ConvergenceLogger(
            directory=directory,
            offset_energy=weight_correlation,
        )

    time_start = time.perf_counter()

    result = scipy.optimize.differential_evolution(
        objective,
        bounds=scipy.optimize.Bounds(
            lb=na.pack(lower).ndarray,
            ub=na.pack(upper).ndarray,
        ),
        x0=na.pack(parameters).ndarray,
        callback=callback,
        **kwargs_optimizer,
    )

    result_parameters = na.unpack(result.x, parameters)

    if callback is not None:
        elapsed = time.perf_counter() - time_start
        callback.log(f"{result}")
        callback.log(f"{result_parameters}")
        callback.log(f"elapsed time: {datetime.timedelta(seconds=elapsed)}")

    return result_parameters


def fit_distortion_scan(
    instrument: esis.optics.abc.AbstractInstrument,
    scene: na.FunctionArray,
    observation: na.AbstractScalar,
    grids: Sequence[dict[str, u.Quantity]],
    parameters: None | DistortionParameters = None,
    pupil: None | na.AbstractCartesian2dVectorArray = None,
    axis_wavelength: None | str = None,
    axis_field: None | tuple[str, str] = None,
    axis_channel: None | str = None,
    smoothing: None | int = None,
    sigma_psf: None | float = 1.0,
    seed: int = 0,
    tolerance: None | float = None,
    num_repeat: int = 8,
    directory: None | pathlib.Path = None,
) -> DistortionParameters:
    r"""
    Fit the distortion parameters of an instrument by scanning the merit.

    A derivative-free coordinate search: for each round in `grids`, every
    listed parameter is scanned along the given offsets while the others are
    held at their current best, and the parameter is moved to the peak of the
    sampled correlation curve, refined with a three-point parabola. Rounds
    are typically coarse-to-fine, so early rounds capture the solution and
    later rounds polish it.

    If `axis_channel` is given, the correlation of each channel is recorded
    separately during every scan, and each channel's peak is read off its own
    curve. Because a channel's correlation depends only on that channel's
    parameters, one scan pass fits all channels simultaneously, and the
    fitted parameters gain an `axis_channel` axis.

    This is the production fitting engine: unlike
    :func:`scipy.optimize.least_squares` driving a
    :class:`DistortionResidual`, which was found to under-converge on the
    locally-flat correlation surface of realistic scenes, the scans read the
    shape of the merit basin directly and are robust as long as the true
    solution lies within the coarsest scan range (measured to be at least
    :math:`\pm 10''` in pointing for the ESIS-I flight data).

    Parameters
    ----------
    instrument
        The instrument model to fit.
    scene
        The spectral radiance of the scene as a function of wavelength and
        field position, imaged through the instrument on every evaluation.
    observation
        The observed image that the modeled images are compared against.
        Any axes of the modeled image which are not present in this array
        are summed over before comparing.
    grids
        The scan schedule: a sequence of rounds, each a mapping from a field
        name of :class:`DistortionParameters` to a one-dimensional
        :class:`~astropy.units.Quantity` of offsets (relative to the current
        best) to scan, for example
        ``[dict(pitch=np.linspace(-10, 10, 21) * u.arcsec)]``.
        Strongly-coupled fields can be scanned jointly by using a tuple of
        field names as the key and a matching tuple of offset grids as the
        value, for example ``{("pitch_grating", "pitch"): (grid_a, grid_b)}``,
        which scans the full outer product and reads the joint peak —
        independent scans of such pairs converge to a compensating local
        optimum instead of the true solution.
    parameters
        The starting point of the fit.
        If :obj:`None`, the current parameters of `instrument` are used.
    pupil
        The vertices of the pupil grid used to image the scene.
    axis_wavelength
        The logical axis of the scene corresponding to changing wavelength.
    axis_field
        The logical axes of the scene corresponding to changing field position.
    axis_channel
        The logical axis of the observation corresponding to changing camera
        channel. If given, each channel's peak is read from its own
        correlation curve.
    smoothing
        The width, in detector pixels, of a box filter applied to both images
        before comparing.
    sigma_psf
        The standard deviation, in detector pixels, of a Gaussian point-spread
        function convolved with the modeled image only.
        See :attr:`DistortionResidual.sigma_psf`.
    seed
        The seed used to make each evaluation deterministic.
        See :attr:`DistortionResidual.seed`.
    tolerance
        If given, the last round of `grids` is repeated until the mean
        correlation improves by less than this amount, at most `num_repeat`
        extra rounds.
    num_repeat
        The maximum number of extra polish rounds appended when `tolerance`
        is given.
    directory
        A directory where the scan curves and convergence history are written
        as ``scan.json`` and ``scan.log``. If :obj:`None`, the fit is not
        logged.

    Raises
    ------
    ValueError
        If a joint entry of `grids` has a different number of fields and
        offset grids.

    Examples
    --------
    Fit the per-frame payload pointing of the ESIS flight-1 model to a
    Level-1 frame.

    .. code-block:: python

        import numpy as np
        import astropy.units as u
        import esis

        obs = esis.flights.f1.data.level_1()[dict(time=0)]
        scene = ...  # an AIA scene resampled to the frame's timestamp

        instrument = esis.flights.f1.optics.distortion_fit(num_distribution=0)

        fitted = esis.optics.fit_distortion_scan(
            instrument=instrument,
            scene=scene,
            observation=obs.outputs.value,
            grids=[
                dict(
                    pitch=np.linspace(-10, 10, 21) * u.arcsec,
                    yaw=np.linspace(-10, 10, 21) * u.arcsec,
                    roll=np.linspace(-0.4, 0.4, 11) * u.deg,
                ),
                dict(
                    pitch=np.linspace(-1, 1, 11) * u.arcsec,
                    yaw=np.linspace(-1, 1, 11) * u.arcsec,
                    roll=np.linspace(-0.05, 0.05, 11) * u.deg,
                ),
            ],
            axis_wavelength="velocity",
            axis_field=("detector_x", "detector_y"),
            sigma_psf=1.0,
        )
    """
    if parameters is None:
        parameters = DistortionParameters.from_instrument(instrument)

    parameters = copy.deepcopy(parameters)

    path_log = None
    if directory is not None:
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path_log = directory / "scan.log"
        path_log.write_text(f"--- scan fit start: {datetime.datetime.now()} ---\n")

    def log(message: str) -> None:
        if path_log is not None:
            with path_log.open("a") as f:
                f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | {message}\n")

    def evaluate(p: DistortionParameters) -> na.AbstractScalar:
        return _correlation_model(
            instrument=instrument,
            parameters=p,
            scene=scene,
            observation=observation,
            pupil=pupil,
            axis_wavelength=axis_wavelength,
            axis_field=axis_field,
            axis_channel=axis_channel,
            smoothing=smoothing,
            sigma_psf=sigma_psf,
            seed=seed,
        )

    def merit(corr: na.AbstractScalar) -> float:
        if axis_channel is not None and axis_channel in na.shape(corr):
            corr = corr.mean(axis_channel)
        return float(na.value(corr).ndarray)

    time_start = time.perf_counter()
    nfev = 0
    summary_rounds = []

    corr = evaluate(parameters)
    nfev += 1
    merit_current = merit(corr)
    log(f"start | correlation {merit_current:.6f}")

    schedule = list(grids)
    if tolerance is not None and len(schedule) > 0:
        schedule = schedule + [schedule[-1]] * num_repeat

    for index, grids_round in enumerate(schedule):
        curves = {}
        for key, offsets in grids_round.items():
            fields = key if isinstance(key, tuple) else (key,)
            if not isinstance(offsets, tuple):
                offsets = (offsets,)
            offsets = tuple(u.Quantity(o) for o in offsets)
            if len(fields) != len(offsets):
                raise ValueError(
                    f"the number of fields {fields} does not match the "
                    f"number of offset grids for round {index}"
                )

            shape_grid = tuple(len(o) for o in offsets)
            values = []
            for deltas in itertools.product(*offsets):
                trial = copy.copy(parameters)
                for field, delta in zip(fields, deltas):
                    setattr(trial, field, getattr(trial, field) + delta)
                values.append(evaluate(trial))
                nfev += 1

            # one correlation curve per channel, or a single coherent curve
            values = np.stack(
                [np.atleast_1d(na.value(v).ndarray) for v in values],
            )
            num_curve = values.shape[~0]
            values = values.reshape(shape_grid + (num_curve,))

            # locate the joint maximum of each curve, then refine every field
            # with a parabola along its own axis through the maximum
            peak = {field: [] for field in fields}
            for i in range(num_curve):
                v = values[..., i]
                index_max = np.unravel_index(int(np.argmax(v)), v.shape)
                for a, field in enumerate(fields):
                    section = v[
                        tuple(
                            slice(None) if b == a else index_max[b]
                            for b in range(len(fields))
                        )
                    ]
                    peak[field].append(_peak_parabola(offsets[a].value, section))

            for a, field in enumerate(fields):
                if axis_channel is not None and num_curve > 1:
                    delta = na.ScalarArray(np.array(peak[field]), axes=axis_channel)
                else:
                    delta = peak[field][0]
                setattr(
                    parameters,
                    field,
                    getattr(parameters, field) + delta * offsets[a].unit,
                )

            name = " x ".join(fields)
            curves[name] = dict(
                fields=list(fields),
                offsets=[[float(x) for x in o.value] for o in offsets],
                unit=[str(o.unit) for o in offsets],
                values=values.tolist(),
                peak={f: [float(p) for p in peak[f]] for f in fields},
            )
            peak_round = {f: np.round(peak[f], 6) for f in fields}
            log(f"round {index} | {name} | peak {peak_round}")

        corr = evaluate(parameters)
        nfev += 1
        gain = merit(corr) - merit_current
        merit_current = merit(corr)
        log(
            f"round {index} | correlation {merit_current:.6f} "
            f"({gain:+.6f}) | nfev {nfev}"
        )

        summary_rounds.append(
            dict(
                index=index,
                correlation=[float(x) for x in np.atleast_1d(na.value(corr).ndarray)],
                gain=gain,
                curves=curves,
            )
        )

        if directory is not None:
            summary = dict(
                rounds=summary_rounds,
                nfev=nfev,
                seconds=time.perf_counter() - time_start,
            )
            with (directory / "scan.json").open("w") as f:
                json.dump(summary, f, indent=2)

        if tolerance is not None and index >= len(grids) and gain < tolerance:
            log(f"converged | round gain {gain:+.6f} < {tolerance}")
            break

    return parameters


def fit_distortion_series(
    instrument: esis.optics.abc.AbstractInstrument,
    scenes: Sequence[na.FunctionArray],
    observations: Sequence[na.AbstractScalar],
    grids: Sequence[dict[str, u.Quantity]],
    parameters: None | DistortionParameters = None,
    pupil: None | na.AbstractCartesian2dVectorArray = None,
    axis_wavelength: None | str = None,
    axis_field: None | tuple[str, str] = None,
    axis_channel: None | str = None,
    smoothing: None | int = None,
    sigma_psf: None | float = 1.0,
    seed: int = 0,
    tolerance: None | float = None,
    num_repeat: int = 8,
    directory: None | pathlib.Path = None,
    workers: int = 1,
) -> list[DistortionParameters]:
    """
    Fit a time series of frames with :func:`fit_distortion_scan`.

    Every frame is fit independently, starting from the same `parameters`
    (for example the best fit of a reference frame) rather than warm-starting
    from the previous frame: independent fits cannot accumulate drift from
    one bad frame, make the per-frame results directly comparable, and
    parallelize perfectly.

    Parameters
    ----------
    instrument
        The instrument model to fit.
    scenes
        The per-frame scenes, one for each observation.
    observations
        The per-frame observed images, in time order.
    grids
        The scan schedule applied to every frame.
        See :func:`fit_distortion_scan`.
    parameters
        The starting point shared by every frame. If :obj:`None`, the current
        parameters of `instrument` are used.
    pupil
        The vertices of the pupil grid used to image the scenes.
    axis_wavelength
        The logical axis of the scenes corresponding to changing wavelength.
    axis_field
        The logical axes of the scenes corresponding to changing field position.
    axis_channel
        The logical axis of the observations corresponding to changing camera
        channel. See :func:`fit_distortion_scan`.
    smoothing
        The width, in detector pixels, of a box filter applied to both images
        before comparing.
    sigma_psf
        The standard deviation, in detector pixels, of a Gaussian point-spread
        function convolved with the modeled image only.
        See :attr:`DistortionResidual.sigma_psf`.
    seed
        The seed used to make each evaluation deterministic.
    tolerance
        If given, the last round of `grids` is repeated for each frame until
        the mean correlation improves by less than this amount.
        See :func:`fit_distortion_scan`.
    num_repeat
        The maximum number of extra polish rounds appended when `tolerance`
        is given.
    directory
        A directory under which each frame's fit is logged in a ``frame_NNN``
        subdirectory. If :obj:`None`, the fits are not logged.
    workers
        The number of processes used to fit frames concurrently.
        A value of 1 fits the frames serially in the current process.
    """
    if parameters is None:
        parameters = DistortionParameters.from_instrument(instrument)

    def kwargs_frame(i: int, scene, observation) -> dict[str, Any]:
        directory_frame = None
        if directory is not None:
            directory_frame = pathlib.Path(directory) / f"frame_{i:03d}"
        return dict(
            instrument=instrument,
            scene=scene,
            observation=observation,
            grids=grids,
            parameters=parameters,
            pupil=pupil,
            axis_wavelength=axis_wavelength,
            axis_field=axis_field,
            axis_channel=axis_channel,
            smoothing=smoothing,
            sigma_psf=sigma_psf,
            seed=seed,
            tolerance=tolerance,
            num_repeat=num_repeat,
            directory=directory_frame,
        )

    frames = list(zip(scenes, observations))

    if workers > 1:
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_fit_frame_scan, kwargs_frame(i, scene, observation))
                for i, (scene, observation) in enumerate(frames)
            ]
            return [future.result() for future in futures]

    return [
        _fit_frame_scan(kwargs_frame(i, scene, observation))
        for i, (scene, observation) in enumerate(frames)
    ]


def _fit_frame_scan(kwargs: dict[str, Any]) -> DistortionParameters:
    """Fit a single frame; a picklable target for the process pool."""
    return fit_distortion_scan(**kwargs)


@dataclasses.dataclass(repr=False)
class ConvergenceLogger(
    optika.mixins.Printable,
):
    """
    A callback which logs the convergence of an optimization to disk.

    Intended for :func:`scipy.optimize.differential_evolution`.
    On every iteration, the best objective value and the standard deviation of
    the population energies are appended to a CSV file, the best parameter
    vector is appended to a text log, and a convergence plot is saved, so that
    long-running fits can be monitored and post-mortemed.
    """

    directory: pathlib.Path
    """The directory where the log files are saved."""

    offset_energy: float = 1000.0
    """
    An offset added to the objective values before plotting on a logarithmic
    scale, chosen to keep the (typically negative) correlation objective of
    :class:`DistortionObjective` positive.
    """

    def __post_init__(self):
        self.directory = pathlib.Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.iteration = 0
        self.history_energy = []
        self.history_deviation = []
        self.path_data.write_text("iteration,energy_best,deviation_population\n")
        self.path_log.write_text(
            f"--- optimization start: {datetime.datetime.now()} ---\n"
        )

    @property
    def path_data(self) -> pathlib.Path:
        """The path of the CSV file containing the convergence history."""
        return self.directory / "convergence_data.csv"

    @property
    def path_log(self) -> pathlib.Path:
        """The path of the text file mirroring the console output."""
        return self.directory / "full_output.log"

    @property
    def path_plot(self) -> pathlib.Path:
        """The path of the convergence plot."""
        return self.directory / "convergence_plot.png"

    def log(self, message: str) -> None:
        """
        Print a message to the console and append it to :attr:`path_log`.

        Parameters
        ----------
        message
            The message to log.
        """
        print(message)
        with self.path_log.open("a") as f:
            f.write(message + "\n")

    def __call__(self, intermediate_result) -> None:
        """
        Record the state of the optimizer after an iteration.

        Parameters
        ----------
        intermediate_result
            The :class:`scipy.optimize.OptimizeResult` passed by
            :func:`scipy.optimize.differential_evolution`, with at least the
            ``x``, ``fun``, and ``population_energies`` attributes.
        """
        self.iteration += 1
        energy = intermediate_result.fun
        deviation = np.std(intermediate_result.population_energies)

        self.history_energy.append(energy)
        self.history_deviation.append(deviation)

        with self.path_data.open("a") as f:
            f.write(f"{self.iteration},{energy:.8e},{deviation:.4e}\n")

        x = np.array2string(intermediate_result.x, precision=4, separator=", ")
        self.log(f"iteration {self.iteration:03d} | energy: {energy:.6e} | x: {x}")

        self.save_plot()

    def save_plot(self) -> None:
        """Save a plot of the convergence history to :attr:`path_plot`."""
        fig, axs = plt.subplots(ncols=2, figsize=(10, 4), constrained_layout=True)
        axs[0].plot(np.array(self.history_energy) + self.offset_energy)
        axs[0].set_title("best energy")
        axs[0].set_yscale("log")
        axs[1].plot(self.history_deviation)
        axs[1].set_title("population diversity (std. dev.)")
        axs[1].set_yscale("log")
        fig.savefig(self.path_plot)
        plt.close(fig)
