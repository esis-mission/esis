from __future__ import annotations
from typing import Any
import copy
import dataclasses
import datetime
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
    "ConvergenceLogger",
    "fit_distortion",
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

        correlation = _correlation(image, observation)

        position = system.rayfunction_default.outputs.position
        distance = np.square(position.to(u.mm).mean().length.value)

        result = -self.weight_correlation * correlation + distance

        return float(na.value(result).ndarray)


def _correlation(
    a: na.AbstractScalar,
    b: na.AbstractScalar,
) -> na.AbstractScalar:
    """
    Compute the Pearson correlation coefficient of two arrays.

    Parameters
    ----------
    a
        The first array, standardized before comparing.
        If it is constant, it is compared without standardizing.
    b
        The second array, standardized before comparing.
        If it is constant, it is compared without standardizing.
    """
    a = a - a.mean()
    deviation_a = a.std()
    if na.value(deviation_a).ndarray != 0:
        a = a / deviation_a

    b = b - b.mean()
    deviation_b = b.std()
    if na.value(deviation_b).ndarray != 0:
        b = b / deviation_b

    return (a * b).sum() / a.size


def fit_distortion(
    instrument: esis.optics.abc.AbstractInstrument,
    scene: na.FunctionArray,
    observation: na.AbstractScalar,
    bounds: tuple[DistortionParameters, DistortionParameters],
    parameters: None | DistortionParameters = None,
    pupil: None | na.AbstractCartesian2dVectorArray = None,
    axis_wavelength: None | str = None,
    axis_field: None | tuple[str, str] = None,
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
