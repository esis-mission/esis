"""The merit of a distortion fit: a linearized image compared with a frame."""

from __future__ import annotations
from typing import Callable
import numpy as np
import astropy.units as u
import scipy.ndimage
import named_arrays as na
import optika
import esis

__all__ = [
    "LinearMerit",
    "correlation",
    "least_squares",
]


def _standardize(
    a: na.AbstractScalar,
    axis: None | tuple[str, ...] = None,
) -> na.AbstractScalar:
    """
    Shift and scale an array to zero mean and unit standard deviation.

    A constant array is left at zero, so that it contributes nothing to a
    correlation.

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


def correlation(
    a: na.AbstractScalar,
    b: na.AbstractScalar,
    axis: None | tuple[str, ...] = None,
) -> na.AbstractScalar:
    """
    Compute the Pearson correlation coefficient of two arrays.

    Maximizing the correlation is the same as minimizing the squared
    difference after fitting a gain and an offset, so it is the least-squares
    merit with the absolute calibration of the frame left free.

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


def least_squares(
    a: na.AbstractScalar,
    b: na.AbstractScalar,
    axis: None | tuple[str, ...] = None,
) -> na.AbstractScalar:
    """
    Compute one minus the squared difference of two arrays relative to the second.

    Unlike :func:`correlation` this compares absolute intensities: the gain
    and the offset of `a` are not free, so a model image is only rewarded
    when it reproduces the level of the observation as well as its shape.
    It is one at a perfect match and falls with the residual, so that it can
    be maximized in place of the correlation.

    Parameters
    ----------
    a
        The model image.
    b
        The observation, whose power normalizes the residual.
    axis
        The logical axes along which to sum.
        If :obj:`None`, every axis is used.
    """
    power = (b * b).sum(axis)
    residual = ((a - b) * (a - b)).sum(axis)
    return 1 - residual / np.where(power == 0, 1, power)


def _scalar(a: float | u.Quantity | na.AbstractScalar) -> float:
    """Read a dimensionless scalar out of whatever array type carries it."""
    return float(np.asarray(na.value(na.as_named_array(a)).ndarray).ravel()[0])


def _host(a: na.AbstractScalar) -> na.ScalarArray:
    """Bring an array that may live on a device back to the host."""
    ndarray = a.ndarray
    if not isinstance(ndarray, np.ndarray):  # pragma: nocover
        return na.ScalarArray(np.asarray(ndarray.copy_to_host()), axes=a.axes)
    return a


class LinearMerit:
    """
    The merit of a set of :class:`DistortionParameters` for one channel.

    Each evaluation applies the parameters to the channel, linearizes the
    resulting optical system (:meth:`optika.systems.SequentialSystem.linearize`),
    clips the scene by the field stop in object space, images it with a
    conservative regrid onto the sensor, and returns the Pearson correlation
    with the observed frame.

    Compared with imaging the scene by a sparse ray trace, the linearized
    image has no sampling noise, so the merit is deterministic and smooth,
    and it needs no point-spread function: the exact footprint integral of
    the regrid is a physically sized smoothing on its own.  It is also about
    three times cheaper, and the regrid can run on a GPU.

    The merit is geometric: the materials of the instrument are idealized
    and the effective area of every line is pinned to a common value, so
    that the fit measures where light lands and not how much of it there is.

    Instances are callables of a flat parameter vector, picklable so that
    :func:`scipy.optimize.differential_evolution` can evaluate them in worker
    processes, and return the *negative* correlation for minimization.

    Parameters
    ----------
    instrument
        The channel to fit, usually the as-built model reduced to a single
        channel.
    parameters
        The prototype which defines the units and structure of the flat
        parameter vector.
    scene
        The spectral radiance of the scene as a function of wavelength and
        field position, on the cell *vertices* of the field grid.
    observation
        The observed frame of this channel.
    device
        The device on which to build and apply the regridding weights; see
        :meth:`optika.systems.AbstractLinearSystem.image`.
    degree
        The degree of the polynomial distortion model.
    merit
        Which comparison :meth:`__call__` maximizes: ``"correlation"``,
        the Pearson correlation of :meth:`correlation`, or
        ``"least_squares"``, the absolute comparison of :meth:`score`, which
        lets the channel's
        :attr:`~esis.optics.DistortionParameters.degradation` set the level
        of the image.
    axis_wavelength
        The logical axis of the scene along which the wavelength varies
        within each spectral line.
    axis_field
        The logical axes of the scene corresponding to field position.
    """

    def __init__(
        self,
        instrument: esis.optics.abc.AbstractInstrument,
        parameters: esis.optics.DistortionParameters,
        scene: na.FunctionArray,
        observation: na.AbstractScalar,
        device: None | str = None,
        degree: int = 2,
        merit: str = "correlation",
        axis_wavelength: str = "velocity",
        axis_field: tuple[str, str] = ("detector_x", "detector_y"),
    ):
        self.instrument = instrument
        self.parameters = parameters
        self.observation = na.value(observation)
        self.device = device
        self.degree = degree
        if merit not in ("correlation", "least_squares"):
            raise ValueError(f"unknown merit {merit!r}")
        self.merit = merit
        self.axis_wavelength = axis_wavelength
        self.axis_field = axis_field
        # the linear forward model wants spectral-positional coordinates
        self.scene = na.FunctionArray(
            inputs=na.SpectralPositionalVectorArray(
                wavelength=scene.inputs.wavelength,
                position=scene.inputs.position,
            ),
            outputs=scene.outputs,
        )
        self.field_centers = self.scene.inputs.position.cell_centers(axis=axis_field)
        self.num_calls = 0
        self.num_failed = 0
        # the level of the frame away from every window, which the image
        # model cannot produce, is removed before any comparison
        self.background = self._background(parameters)
        self.observation = self.observation - self.background

    def _background(
        self,
        parameters: esis.optics.DistortionParameters,
        margin: int = 12,
    ) -> float:
        """
        Estimate the level of the observation outside every window.

        The image of the scene at the given parameters says where the
        windows are; the median of the observation more than `margin`
        pixels from any of them is the background.

        Parameters
        ----------
        parameters
            The distortion parameters that place the windows.
        margin
            The distance from a window, in pixels, beyond which the
            observation is taken to be background.
        """
        axes = tuple(self.axis_field)
        image = np.asarray(self.image(parameters).ndarray_aligned(axes))
        observation = np.asarray(self.observation.ndarray_aligned(axes))
        if observation.shape != image.shape:
            # a placeholder observation, as a probe that only images has
            return 0.0
        lit = image > 1e-4 * image.max()
        outside = ~scipy.ndimage.binary_dilation(lit, iterations=margin)
        if not outside.any():
            return 0.0
        return float(np.median(observation[outside]))

    def estimate_degradation(
        self,
        parameters: esis.optics.DistortionParameters,
    ) -> u.Quantity:
        """
        Estimate the degradation that best scales the image onto the observation.

        The least-squares gain of the image at the given parameters, whose
        own degradation plays no part, which seeds a fit whose start knows
        nothing about the level of the frame.

        Parameters
        ----------
        parameters
            The distortion parameters to image the scene with.
        """
        image = self.image(parameters)
        gain = (image * self.observation).sum() / (image * image).sum()
        return float(na.value(gain).ndarray) * u.dimensionless_unscaled

    def linearize(self, instrument) -> optika.systems.LinearSystem:
        """
        Linearize the channel with the effective area pinned per line.

        Parameters
        ----------
        instrument
            The channel with the current parameters applied.
        """
        linear = instrument.system.linearize(
            wavelength=instrument.wavelength,
            degree=self.degree,
            field_stop=True,
        )
        # the fit is geometric: every line shares one effective area, which
        # also removes the sampling scatter of the fitted area model
        area = linear.area_effective
        linear.area_effective = optika.radiometry.InterpolatedEffectiveAreaModel(
            wavelength=area.wavelength,
            area=0 * area.area + area.area.mean(),
            axis_wavelength=area.axis_wavelength,
        )
        return linear

    def _check_on_sensor(
        self,
        linear: optika.systems.LinearSystem,
        minimum: int = 1000,
    ) -> None:
        """
        Refuse a trial that puts the scene off the sensor.

        Such a trial would have the regrid allocate an empty set of weights,
        which a device kernel cannot survive, and it is the worst possible
        image in any case.

        Raises
        ------
        ValueError
            If fewer than `minimum` cells of the scene land on the sensor.
        """
        coordinates = self.scene.inputs.cell_centers(self.axis_wavelength)
        position = linear.distortion.distort(coordinates).position
        sensor = linear.coordinates_sensor
        lo, hi = sensor.start, sensor.stop
        inside = (
            (position.x > lo.x)
            & (position.x < hi.x)
            & (position.y > lo.y)
            & (position.y < hi.y)
        )
        num = int(np.asarray(na.value(inside).ndarray).sum())
        if num < minimum:
            raise ValueError(f"only {num} scene cells land on the sensor")

    def image(
        self,
        parameters: esis.optics.DistortionParameters,
    ) -> na.ScalarArray:
        """
        Compute the linearized image of the scene for the given parameters.

        Parameters
        ----------
        parameters
            The distortion parameters to apply to the channel.
        """
        instrument = parameters.to_instrument(self.instrument)
        linear = self.linearize(instrument)
        scene = self.scene
        self._check_on_sensor(linear)
        kwargs_device = dict() if self.device is None else dict(device=self.device)
        weights = linear.weights(
            scene.inputs,
            axis_wavelength=self.axis_wavelength,
            axis_field=self.axis_field,
            **kwargs_device,
        )
        result = linear.image_from_weights(
            weights,
            scene,
            axis_wavelength=self.axis_wavelength,
            axis_field=self.axis_field,
            noise=False,
        )
        result = _host(na.value(result.outputs))
        extra = tuple(set(na.shape(result)) - set(self.axis_field))
        if extra:
            result = result.sum(axis=extra)
        # kept for callers that also need the mapping, e.g. the alignment
        self.linear = linear
        return result

    def correlation(
        self,
        parameters: esis.optics.DistortionParameters,
    ) -> float:
        """
        Compute the Pearson correlation of the linearized image with the observation.

        Parameters
        ----------
        parameters
            The distortion parameters to apply to the channel.
        """
        image = self.image(parameters)
        result = correlation(image, self.observation, self.axis_field)
        return float(na.value(result).ndarray)

    def score(
        self,
        parameters: esis.optics.DistortionParameters,
    ) -> float:
        """
        Compute the least-squares score of the linearized image against the observation.

        The image is scaled by the channel's
        :attr:`~esis.optics.DistortionParameters.degradation` before the
        comparison, see :func:`least_squares`.

        Parameters
        ----------
        parameters
            The distortion parameters to apply to the channel.
        """
        image = self.image(parameters) * _scalar(parameters.degradation)
        result = least_squares(image, self.observation, self.axis_field)
        return float(na.value(result).ndarray)

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate the negative merit for a flat parameter vector.

        A trial that cannot be imaged (rays off the stops, a singular
        polynomial fit, the scene off the sensor) is reported as the worst
        possible image rather than raising, so that a population-based
        optimizer can carry on.

        Parameters
        ----------
        x
            The flat parameter vector, interpreted in the units and structure
            of :attr:`parameters`.
        """
        parameters = na.unpack(np.asarray(x), self.parameters)
        self.num_calls += 1
        try:
            if self.merit == "least_squares":
                return -self.score(parameters)
            return -self.correlation(parameters)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            # a trial that cannot be imaged (off the sensor, a singular
            # linearization) is the worst possible; anything else, such as
            # a device that cannot run the regrid, is a fault and propagates
            self.num_failed += 1
            return 1.0


def correlation_raytraced(
    instrument: esis.optics.abc.AbstractInstrument,
    parameters: esis.optics.DistortionParameters,
    scene: na.FunctionArray,
    observation: na.AbstractScalar,
    pupil: None | na.AbstractCartesian2dVectorArray,
    sigma_psf: None | float,
    seed: int = 0,
    axis_wavelength: str = "velocity",
    axis_field: tuple[str, str] = ("detector_x", "detector_y"),
) -> float:
    """
    Compute the correlation of a ray-traced image with the observation.

    This is the merit the fit used before it was linearized, kept as an
    independent forward model to score a fit against.  The ray trace samples
    every scene cell with a few rays, so the image carries sampling noise and
    needs a point-spread function of a few pixels to compare with the data.

    Parameters
    ----------
    instrument
        The channel to image.
    parameters
        The distortion parameters to apply to the channel.
    scene
        The spectral radiance of the scene.
    observation
        The observed frame of this channel.
    pupil
        The vertices of the pupil grid used to image the scene.
    sigma_psf
        The standard deviation, in detector pixels, of the Gaussian
        point-spread function convolved with the modeled image.
    seed
        The seed used to freeze the imaging model's random ray jitter.
    axis_wavelength
        The logical axis of the scene along which the wavelength varies
        within each spectral line.
    axis_field
        The logical axes of the scene corresponding to field position.
    """
    instrument = parameters.to_instrument(instrument)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        image = instrument.system.image(
            scene=scene,
            pupil=pupil,
            axis_wavelength=axis_wavelength,
            axis_field=axis_field,
            noise=False,
        )
    finally:
        np.random.set_state(state)
    image = na.value(image.outputs)
    observation = na.value(observation)
    extra = tuple(set(na.shape(image)) - set(na.shape(observation)))
    if extra:
        image = image.sum(axis=extra)
    if sigma_psf is not None:
        radius = max(2, int(np.ceil(3 * sigma_psf)))
        x = np.arange(-radius, radius + 1)
        profile = np.exp(-np.square(x / sigma_psf) / 2)
        kernel = np.multiply.outer(profile, profile)
        kernel = na.ScalarArray(kernel / kernel.sum(), axes=axis_field)
        image = na.convolve(image, kernel, axis=axis_field)
    return float(na.value(correlation(image, observation, axis_field)).ndarray)


ObjectiveType = Callable[[np.ndarray], float]
