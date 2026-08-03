from typing import Self
import time as _time
import dataclasses
import scipy.ndimage
import numpy as np
import matplotlib.axes
import matplotlib.animation
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
import IPython.display
import astropy.constants
import astropy.units as u
import named_arrays as na
import esis
from .. import Level_1
from . import _caching

__all__ = [
    "Level_4",
]


@dataclasses.dataclass(eq=False, repr=False)
class Level_4(
    na.FunctionArray,
):
    """
    Time-dependent MART inversions of the Level-1 images.

    Where the lower data levels live on the sensor grid, this level lives on
    the scene grid: :attr:`~named_arrays.FunctionArray.outputs` is the
    reconstructed spectral radiance on a ``(time, wavelength, field_x,
    field_y)`` grid, where the wavelength axis is the concatenation of one
    Doppler window per spectral line.

    This class is intended to be created from an instance of
    :class:`~esis.data.Level_1` using the :meth:`from_level_1` method.
    """

    instrument: None | esis.optics.Instrument = None
    """A model of the optical system that the images were inverted with."""

    wavelength_center: None | na.AbstractScalarArray = None
    """The rest wavelength of each reconstructed spectral line."""

    label_line: None | list[str] = None
    """
    A human-readable label for each reconstructed spectral line,
    including any blended lines sharing its window.
    """

    members_line: None | list[list[tuple[u.Quantity, float]]] = None
    """
    The spectral lines tied into each reconstructed window.

    Each window is reconstructed as a single scene, so lines that share an
    upper level — and therefore a fixed photon branching ratio — can be
    tied together and solved for as one emitting ion.  This lists, for each
    window, the rest wavelength of every line contributing to it paired
    with its photon ratio relative to the first, whose rest wavelength is
    the :attr:`wavelength_center` the velocity axis is measured against.

    A window with a single member, ratio one, is an ordinary untied line;
    :obj:`None` means every window is untied.
    """

    num_velocity: None | int = None
    """The number of velocity bins in the Doppler window of each line."""

    mean_chi_squared: None | na.AbstractScalarArray = None
    r"""The final mean :math:`\chi^2` of each channel for each frame."""

    num_iteration: None | na.AbstractScalarArray = None
    """The number of MART iterations performed for each frame."""

    factor_norm: None | na.AbstractScalarArray = None
    """The channel-normalization factor applied to each image."""

    where_shadow: None | na.AbstractScalarArray = None
    """The mask of shaded detector pixels excluded from the inversion."""

    drift_applied: None | u.Quantity = None
    """
    The scene offset already removed from each frame by :meth:`coregister`.

    :obj:`None` means the frames are as reconstructed, on a grid fixed in
    the coordinates of the instrument, so a feature fixed on the Sun
    wanders across them as the payload pointing does.  Once set, the
    frames share a common sky frame and the coordinates apply to all of
    them rather than only to the reference frame.
    """

    axis_time: str = dataclasses.field(default="time", kw_only=True)
    """The name of the logical axis corresponding to changing time."""

    axis_wavelength: str = dataclasses.field(default="wavelength", kw_only=True)
    """The name of the logical axis corresponding to changing wavelength."""

    axis_x: str = dataclasses.field(default="field_x", kw_only=True)
    """The name of the horizontal axis of the scene."""

    axis_y: str = dataclasses.field(default="field_y", kw_only=True)
    """The name of the vertical axis of the scene."""

    axis_line: str = dataclasses.field(default="line", kw_only=True)
    """The name of the logical axis corresponding to the different lines."""

    @classmethod
    def from_level_1(
        cls,
        a: Level_1,
        wavelength_center: na.AbstractScalarArray,
        width_doppler: na.AbstractScalarArray,
        label_line: None | list[str] = None,
        instrument: None | esis.optics.Instrument = None,
        limit_velocity: u.Quantity = 200 * u.km / u.s,
        num_velocity: None | int = None,
        pitch_scene: None | u.Quantity = None,
        factor_fov: float = 1.25,
        where_shadow: None | bool | na.AbstractScalarArray = True,
        degree: int = 2,
        index_time_reference: int = 0,
        index_channel_reference: int = 0,
        floor_guess: float = 0.01,
        gamma: float = 1,
        threshold_convergence: float = 1e-2,
        num_iteration: int = 100,
        verbose: bool = False,
        axis_wavelength: str = "wavelength",
        axis_x: str = "field_x",
        axis_y: str = "field_y",
        axis_line: str = "line",
    ) -> Self:
        r"""
        Invert every frame of a :class:`~esis.data.Level_1` observation.

        The forward model is the fitted optical system linearized on the
        wavelength grid (:func:`optika.systems.SequentialSystem.linearize`),
        adapted to the CTIS instrument interface by
        :class:`ctis.instruments.OptikaInstrument` and inverted with
        :class:`ctis.inverters.MartInverter`.
        The linearization and the regridding weights are cached in the ESIS
        cache, so rerunning with an unchanged instrument model, scene grid,
        and code loads the stored results instead of recomputing them.

        Every frame is inverted with the weights of the reference frame:
        rebuilding the weights per frame has been checked to be
        :math:`\chi^2`-equivalent, since the payload pointing drift moves the
        scene but barely changes the field-stop-to-detector mapping.
        The reference frame is inverted first, from a spatially-uniform
        Gaussian spectral seed; every other frame warm-starts from the
        solution of its neighbor, rescaled to its own total signal.

        Before inversion, negative pixels are clipped (MART reconstructs a
        non-negative scene) and each channel is normalized to the mean of the
        reference channel, a first-order correction for the uncalibrated
        per-channel effective areas.

        .. note::

            This method requires the :mod:`ctis` library, currently on its
            unreleased ``feature/electron-based-instruments`` branch
            (`ctis PR #18 <https://github.com/sun-data/ctis/pull/18>`_),
            along with :mod:`optika` on ``feature/interpolated-system``
            (`optika PR #152 <https://github.com/sun-data/optika/pull/152>`_).

        Parameters
        ----------
        a
            The Level-1 observation to invert.
        wavelength_center
            The rest wavelength of each spectral line to reconstruct,
            arranged along `axis_line`.
        width_doppler
            The Doppler width of each spectral line, arranged along
            `axis_line`, used as the width of the Gaussian initial guess.
        label_line
            A human-readable label for each spectral line, including any
            blended lines sharing its window.
        instrument
            A model of the ESIS instrument to invert with.
            If :obj:`None` (the default), the instrument associated with `a`
            is used.
        limit_velocity
            The half-width of the Doppler window reconstructed around the
            rest wavelength of each line.
        num_velocity
            The number of velocity bins in the Doppler window of each line.
            If :obj:`None` (the default), the number of bins that matches the
            spectral plate scale of the instrument, measured from the
            linearized optical system.
        pitch_scene
            The spatial pitch of the scene grid.
            If :obj:`None` (the default), the spatial plate scale of the
            instrument, measured from the linearized optical system.
        factor_fov
            The factor by which the scene grid is padded beyond the extent of
            the default raytrace grid, so the corners of the octagonal field
            stop are not truncated.
        where_shadow
            A boolean mask of detector pixels shaded by the misaligned
            frame-transfer storage-region mask, :obj:`True` where shaded.
            If :obj:`True` (the default), the mask is measured from the data
            using :meth:`~esis.data.Level_1.where_shadow`; if :obj:`None` or
            :obj:`False`, no pixels are masked.
            Shaded pixels are removed from the forward and transpose weights,
            so the model can place no flux on them and they contribute
            nothing to the multiplicative correction, and their data is
            zeroed so they cannot bias the channel normalization or the
            :math:`\chi^2`.
            Without this mask the near-zero shaded pixels sit inside the
            model's He I footprint and MART carves non-physical dark lanes
            across the He I maps to satisfy them.
        degree
            The degree of the polynomial distortion and vignetting models
            fitted by the linearization.
        index_time_reference
            The index of the frame inverted first, from the Gaussian seed,
            and whose weights are shared by every frame.
            Choose the frame the instrument model was fitted against.
        index_channel_reference
            The index of the channel whose mean the other channels are
            normalized to.
        floor_guess
            The floor of the initial guess, as a fraction of the peak of the
            Gaussian profile.
            A cell seeded at exactly zero would stay zero forever under
            MART's multiplicative updates.
        gamma
            The learning rate of the MART iteration.
        threshold_convergence
            The minimum improvement in the mean :math:`\chi^2` per iteration;
            below it the iteration stops.
        num_iteration
            The maximum number of MART iterations per frame.
        verbose
            Whether to print the convergence statistics of each frame.
        axis_wavelength
            The name of the logical axis of the scene corresponding to
            changing wavelength.
        axis_x
            The name of the horizontal axis of the scene.
        axis_y
            The name of the vertical axis of the scene.
        axis_line
            The logical axis of `wavelength_center` and `width_doppler`
            corresponding to the different lines.
        """
        import ctis

        if instrument is None:
            instrument = a.instrument

        axis_time = a.axis_time
        axis_channel = a.axis_channel

        system = instrument.system
        key = _caching.key_system(system)
        code = _caching.code_state()

        if (pitch_scene is None) or (num_velocity is None):
            pitch_nominal, dispersion_nominal = _plate_scale(
                system=system,
                key=key,
                code=code,
                wavelength_center=wavelength_center,
                limit_velocity=limit_velocity,
                degree=degree,
                axis_wavelength=axis_wavelength,
                axis_line=axis_line,
            )
            if pitch_scene is None:
                pitch_scene = pitch_nominal
            if num_velocity is None:
                ratio = (2 * limit_velocity / dispersion_nominal).to_value(
                    u.dimensionless_unscaled
                )
                num_velocity = int(np.ceil(ratio))

        velocity = na.linspace(
            start=-limit_velocity,
            stop=limit_velocity,
            axis=axis_wavelength,
            num=num_velocity + 1,
        )
        wavelength_vertices = wavelength_center * (1 + velocity / astropy.constants.c)
        wavelength_vertices = wavelength_vertices.to(u.AA).combine_axes(
            axes=(axis_line, axis_wavelength),
            axis_new=axis_wavelength,
        )

        field = system.rayfunction_default.inputs.field
        center = (field.max() + field.min()) / 2
        halfwidth = factor_fov * (field.max() - field.min()) / 2
        start = center - halfwidth
        stop = center + halfwidth

        extent = stop - start
        ratio = (np.maximum(extent.x, extent.y) / pitch_scene).ndarray
        num_field = int(np.ceil(ratio.to_value(u.dimensionless_unscaled)))

        position = na.Cartesian2dVectorLinearSpace(
            start=start,
            stop=stop,
            axis=na.Cartesian2dVectorArray(axis_x, axis_y),
            num=num_field + 1,
        )

        coordinates_scene = na.SpectralPositionalVectorArray(
            wavelength=wavelength_vertices,
            position=position,
        )

        linear = _caching.linear_system(
            system,
            key=key,
            wavelength=wavelength_vertices,
            degree=degree,
            code=code,
        )

        frame_reference = a[{axis_time: index_time_reference}]

        instrument_mart = ctis.instruments.OptikaInstrument(
            system=linear,
            coordinates_scene=coordinates_scene,
            channel=frame_reference.channel,
            axis_channel=axis_channel,
            axis_wavelength=axis_wavelength,
            axis_scene_xy=(axis_x, axis_y),
        )

        instrument_mart.weights = _caching.weights(
            system,
            key=key,
            wavelength=wavelength_vertices,
            degree=degree,
            coordinates_scene=coordinates_scene,
            axis_wavelength=axis_wavelength,
            axis_field=(axis_x, axis_y),
            code=code,
        )
        instrument_mart.weights_transpose = _caching.weights_transpose(
            system,
            key=key,
            wavelength=wavelength_vertices,
            degree=degree,
            coordinates_scene=coordinates_scene,
            axis_wavelength=axis_wavelength,
            axis_field=(axis_x, axis_y),
            code=code,
        )

        if where_shadow is True:
            where_shadow = a.where_shadow()
        if where_shadow is False:
            where_shadow = None

        if where_shadow is not None:
            shape_detector = {
                axis_channel: a.shape[axis_channel],
                a.axis_x: a.shape[a.axis_x],
                a.axis_y: a.shape[a.axis_y],
            }
            keep = (~where_shadow).broadcast_to(shape_detector)
            src = [keep.axes.index(ax) for ax in shape_detector]
            keep_flat = np.moveaxis(keep.ndarray, src, range(3)).reshape(
                shape_detector[axis_channel], -1
            )
            instrument_mart.weights = _filter_weights_shadow(
                weights=instrument_mart.weights,
                keep_flat=keep_flat,
                axis_channel=axis_channel,
                index_detector=1,
            )
            instrument_mart.weights_transpose = _filter_weights_shadow(
                weights=instrument_mart.weights_transpose,
                keep_flat=keep_flat,
                axis_channel=axis_channel,
                index_detector=0,
            )

        electrons = np.maximum(a.outputs, 0)
        if where_shadow is not None:
            electrons = electrons * ~where_shadow
        mean_channel = electrons.mean((a.axis_x, a.axis_y))
        mean_reference = mean_channel[{axis_channel: index_channel_reference}]
        factor_norm = mean_reference / mean_channel
        electrons = electrons * factor_norm

        total = electrons.sum((axis_channel, a.axis_x, a.axis_y))

        guess_reference = _guess(
            instrument_mart=instrument_mart,
            electrons_reference=electrons[{axis_time: index_time_reference}],
            width_doppler=width_doppler,
            velocity=velocity,
            num_velocity=num_velocity,
            floor_guess=floor_guess,
            axis_wavelength=axis_wavelength,
            axis_line=axis_line,
        )

        mart = ctis.inverters.MartInverter(
            instrument=instrument_mart,
            gamma=gamma,
            threshold_convergence=threshold_convergence,
            num_iteration=num_iteration,
        )
        axis_iteration = mart.axis_iteration

        num_time = a.shape[axis_time]
        indices = list(range(index_time_reference, num_time))
        indices += list(range(index_time_reference - 1, -1, -1))

        solutions = [None] * num_time
        chi_squared = [None] * num_time
        iterations = [None] * num_time

        for index in indices:
            if index == index_time_reference:
                guess = guess_reference
            else:
                if index > index_time_reference:
                    index_previous = index - 1
                else:
                    index_previous = index + 1
                scale = total[{axis_time: index}] / total[{axis_time: index_previous}]
                guess = solutions[index_previous] * scale

            images = na.FunctionArray(
                inputs=instrument_mart.coordinates_sensor,
                outputs=electrons[{axis_time: index}],
            )

            t_start = _time.perf_counter()
            result = mart(images, guess=guess)
            t_elapsed = _time.perf_counter() - t_start

            solutions[index] = result.solutions[{axis_iteration: ~0}].outputs
            chi_squared[index] = result.mean_chi_squared[{axis_iteration: ~0}]
            iterations[index] = result.num_iteration

            if verbose:
                chi2 = np.asarray(chi_squared[index].ndarray)
                print(
                    f"frame {index:3d}: {result.num_iteration:3d} iterations,"
                    f" chi2 {np.array2string(chi2, precision=2)},"
                    f" {t_elapsed:.0f} s",
                    flush=True,
                )

        time = a.inputs.time
        if axis_channel in time.shape:
            time = time.mean(axis_channel)

        return cls(
            inputs=na.TemporalSpectralPositionalVectorArray(
                time=time,
                wavelength=wavelength_vertices,
                position=position,
            ),
            outputs=na.stack(solutions, axis=axis_time),
            instrument=instrument,
            wavelength_center=wavelength_center,
            label_line=label_line,
            num_velocity=num_velocity,
            mean_chi_squared=na.stack(chi_squared, axis=axis_time),
            num_iteration=na.ScalarArray(np.array(iterations), axes=(axis_time,)),
            factor_norm=factor_norm,
            where_shadow=where_shadow,
            axis_time=axis_time,
            axis_wavelength=axis_wavelength,
            axis_x=axis_x,
            axis_y=axis_y,
            axis_line=axis_line,
        )

    @property
    def num_line(self) -> int:
        """The number of reconstructed spectral lines."""
        return self.wavelength_center.shape[self.axis_line]

    def label(self, index_line: int) -> str:
        """
        Look up the label of the given line.

        Falls back to the rest wavelength if :attr:`label_line` is not set.

        Parameters
        ----------
        index_line
            The index of the spectral line along :attr:`axis_line`.
        """
        if self.label_line is not None:
            return self.label_line[index_line]
        wavelength = self.wavelength_center[{self.axis_line: index_line}]
        return f"{wavelength.ndarray.to_value(u.AA):.0f} Å"

    def window(self, index_line: int) -> dict[str, slice]:
        """
        Select the wavelength cells belonging to the given line.

        The concatenated wavelength axis has ``num_velocity + 1`` vertices
        per line, so consecutive line windows are separated by one spurious
        "gap" cell, which this slice excludes.

        Parameters
        ----------
        index_line
            The index of the spectral line along :attr:`axis_line`.
        """
        num = self.num_velocity + 1
        return {
            self.axis_wavelength: slice(index_line * num, (index_line + 1) * num - 1),
        }

    @property
    def velocity(self) -> na.AbstractScalarArray:
        """The vertices of the Doppler-velocity grid of each line window."""
        num = self.num_velocity + 1
        wavelength = self.inputs.wavelength[{self.axis_wavelength: slice(0, num)}]
        center = self.wavelength_center[{self.axis_line: 0}]
        result = (wavelength / center - 1) * astropy.constants.c
        return result.to(u.km / u.s)

    @property
    def velocity_center(self) -> na.AbstractScalarArray:
        """The centers of the Doppler-velocity grid of each line window."""
        velocity = self.velocity
        axis = self.axis_wavelength
        lower = velocity[{axis: slice(None, ~0)}]
        upper = velocity[{axis: slice(1, None)}]
        return (lower + upper) / 2

    @property
    def intensity(self) -> na.AbstractScalarArray:
        """The zeroth moment (total radiance) of each line window."""
        result = [
            self.outputs[self.window(i)].sum(self.axis_wavelength)
            for i in range(self.num_line)
        ]
        return na.stack(result, axis=self.axis_line)

    @property
    def velocity_mean(self) -> na.AbstractScalarArray:
        """The first moment (intensity-weighted mean Doppler velocity) of each line."""
        velocity = self.velocity_center
        result = []
        for i in range(self.num_line):
            radiance = self.outputs[self.window(i)]
            moment_0 = radiance.sum(self.axis_wavelength)
            moment_1 = (radiance * velocity).sum(self.axis_wavelength)
            result.append(moment_1 / moment_0)
        return na.stack(result, axis=self.axis_line)

    @property
    def velocity_width(self) -> na.AbstractScalarArray:
        """The second moment (Doppler width) of each line window."""
        velocity = self.velocity_center
        mean = self.velocity_mean
        result = []
        for i in range(self.num_line):
            radiance = self.outputs[self.window(i)]
            mean_i = mean[{self.axis_line: i}]
            moment_0 = radiance.sum(self.axis_wavelength)
            moment_2 = (radiance * np.square(velocity - mean_i)).sum(
                self.axis_wavelength
            )
            result.append(np.sqrt(moment_2 / moment_0))
        return na.stack(result, axis=self.axis_line)

    @property
    def _transmission(self) -> na.AbstractScalarArray:
        """
        The relative atmospheric transmission of each frame.

        The total reconstructed radiance of each frame, normalized to the
        maximum over the flight.  The 2019 flight data fade toward the ends
        of the flight because the payload observes through more atmosphere,
        not because the Sun dims.
        """
        total = self.outputs.sum((self.axis_wavelength, self.axis_x, self.axis_y))
        return total / total.max()

    def drift(
        self,
        index_line: None | int = None,
        index_time_reference: None | int = None,
    ) -> u.Quantity:
        """
        Measure how far the scene has slid across the grid in each frame.

        Every frame is inverted with the weights of one reference pointing,
        onto a grid fixed in the coordinates of the instrument.  The payload
        pointing wanders over the flight, so a fixed solar feature slides
        across that grid: about seven arcseconds end to end for the 2019
        flight, which is several scene cells and is plainly visible in a
        movie of a single event.

        The offset of each frame is measured against the reference frame by
        phase correlation of the line intensity, which needs no pointing
        model and so cannot get its sign backwards.  Compare it with the
        fitted pointing (:func:`esis.flights.f1.optics.distortion_fit` with
        ``axis_time``) to check the two agree.

        Parameters
        ----------
        index_line
            The index of the spectral line to correlate along
            :attr:`axis_line`.
            If :obj:`None`, the brightest line.
        index_time_reference
            The frame every other frame is measured against.
            If :obj:`None`, the middle frame.
        """
        intensity = self.intensity
        if index_line is None:
            total = [
                np.nansum(np.asarray(intensity[{self.axis_line: i}].ndarray))
                for i in range(self.num_line)
            ]
            index_line = int(np.argmax(total))
        intensity = intensity[{self.axis_line: index_line}]

        num_time = self.shape[self.axis_time]
        if index_time_reference is None:
            index_time_reference = num_time // 2

        x = self.inputs.position.x.ndarray.to_value(u.arcsec)
        y = self.inputs.position.y.ndarray.to_value(u.arcsec)
        pitch = np.array([x[1] - x[0], y[1] - y[0]])

        reference = self._index_xy(intensity[{self.axis_time: index_time_reference}])
        reference = np.nan_to_num(reference)
        spectrum_reference = np.fft.rfft2(reference - reference.mean())

        result = np.zeros((num_time, 2))
        for t in range(num_time):
            frame = np.nan_to_num(self._index_xy(intensity[{self.axis_time: t}]))
            spectrum = np.fft.rfft2(frame - frame.mean())
            cross = spectrum * np.conj(spectrum_reference)
            cross = cross / np.maximum(np.abs(cross), 1e-30)
            correlation = np.fft.irfft2(cross, s=reference.shape)
            result[t] = _peak_subpixel(correlation)

        return result * pitch * u.arcsec

    def coregister(
        self,
        drift: None | u.Quantity = None,
        order: int = 1,
        **kwargs,
    ) -> Self:
        """
        Put every frame on a common sky frame.

        The reconstruction lives on a grid fixed in the coordinates of the
        instrument, so the payload's wander carries the scene across it.
        This resamples each frame by the measured offset, after which a
        feature fixed on the Sun stays at fixed grid coordinates and the
        stored coordinates describe every frame rather than only the
        reference.  The offset removed is recorded in
        :attr:`drift_applied`.

        The interpolation is linear by default: the reconstruction is
        non-negative, and a higher-order kernel would overshoot into
        negative values around the sharp edges of the field stop.

        Parameters
        ----------
        drift
            The offset of each frame, from :meth:`drift`.
            If :obj:`None`, it is measured.
        order
            The order of the interpolation.
        kwargs
            Additional arguments for :meth:`drift`.

        Raises
        ------
        ValueError
            If this product has already been coregistered, since the
            offsets would compound.
        """
        if self.drift_applied is not None:
            raise ValueError("this product has already been coregistered")

        if drift is None:
            drift = self.drift(**kwargs)

        x = self.inputs.position.x.ndarray.to_value(u.arcsec)
        y = self.inputs.position.y.ndarray.to_value(u.arcsec)
        pitch = np.array([x[1] - x[0], y[1] - y[0]])
        offset = drift.to_value(u.arcsec) / pitch

        axes = self.outputs.axes
        order_axes = (self.axis_time, self.axis_x, self.axis_y)
        source = [axes.index(ax) for ax in order_axes]
        values = np.moveaxis(np.asarray(self.outputs.ndarray.value), source, (0, 1, 2))

        result = np.empty_like(values)
        for t in range(values.shape[0]):
            if not np.any(offset[t]):
                result[t] = values[t]
                continue
            for k in range(values.shape[3]):
                # the vacated margin lies outside the field stop, which is
                # already empty, so filling it with zero loses nothing
                result[t, ..., k] = scipy.ndimage.shift(
                    values[t, ..., k],
                    -offset[t],
                    order=order,
                    mode="constant",
                    cval=0,
                )
        result = np.moveaxis(result, (0, 1, 2), source)

        return dataclasses.replace(
            self,
            outputs=na.ScalarArray(
                result * na.unit(self.outputs),
                axes=axes,
            ),
            drift_applied=drift,
        )

    def _index_xy(self, a: na.AbstractScalarArray) -> np.ndarray:
        """
        Extract the ndarray of `a` with ``(axis_x, axis_y)`` leading.

        Parameters
        ----------
        a
            The array to extract.
        """
        source = (a.axes.index(self.axis_x), a.axes.index(self.axis_y))
        return np.moveaxis(np.asarray(a.ndarray), source, (0, 1))

    def _coregistered(
        self,
        frames: list[np.ndarray],
        drift: None | u.Quantity,
    ) -> list[np.ndarray]:
        """
        Undo the measured scene drift, frame by frame.

        Parameters
        ----------
        frames
            The maps to shift, one per exposure, with ``(x, y)`` leading.
        drift
            The offset of each frame in arcsec, or :obj:`None` to leave
            the frames alone.
        """
        if drift is None:
            return frames

        x = self.inputs.position.x.ndarray.to_value(u.arcsec)
        y = self.inputs.position.y.ndarray.to_value(u.arcsec)
        pitch = np.array([x[1] - x[0], y[1] - y[0]])
        offset = drift.to_value(u.arcsec) / pitch

        return [
            scipy.ndimage.shift(
                frame,
                -offset[t],
                order=1,
                mode="constant",
                cval=np.nan,
            )
            for t, frame in enumerate(frames)
        ]

    def animate_intensity(
        self,
        index_line: int,
        ax: None | matplotlib.axes.Axes = None,
        percentile_max: float = 99.5,
        drift: None | u.Quantity = None,
        interval: int = 200,
    ) -> matplotlib.animation.FuncAnimation:
        """
        Animate the total intensity of the given line over the flight.

        Parameters
        ----------
        index_line
            The index of the spectral line along :attr:`axis_line`.
        ax
            The :class:`~matplotlib.axes.Axes` instance to use.
            If :obj:`None`, a new set of axes will be created.
        percentile_max
            The percentile of the intensity mapped to the top of the
            colormap.
        drift
            The per-frame scene offset to undo, from :meth:`drift`.
            If :obj:`None` (the default), the frames are shown as
            reconstructed, and a feature fixed on the Sun will wander
            across the field as the payload pointing does.
        interval
            The delay between frames in milliseconds.
        """
        intensity = self.intensity[{self.axis_line: index_line}]

        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)
        else:
            fig = ax.figure

        x = self.inputs.position.x.ndarray.to_value(u.arcsec)
        y = self.inputs.position.y.ndarray.to_value(u.arcsec)
        extent = (x[0], x[~0], y[0], y[~0])

        ax.set_title(self.label(index_line))

        frames = [
            self._index_xy(intensity[{self.axis_time: t}])
            for t in range(self.shape[self.axis_time])
        ]
        frames = self._coregistered(frames, drift)
        vmax = np.nanpercentile(np.stack(frames), percentile_max)

        img = ax.imshow(
            frames[0].T,
            extent=extent,
            origin="lower",
            vmin=0,
            vmax=vmax,
        )
        ax.set_xlabel("field $x$ (arcsec)")
        ax.set_ylabel("field $y$ (arcsec)")
        ax.set_aspect("equal")
        fig.colorbar(
            img,
            ax=ax,
            label=f"intensity ({na.unit(intensity):latex_inline})",
        )

        def update(t: int) -> tuple:
            """
            Draw the frame with the given time index.

            Parameters
            ----------
            t
                The index of the frame to draw.
            """
            img.set_data(frames[t].T)
            return (img,)

        return matplotlib.animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=len(frames),
            interval=interval,
        )

    def animate_doppler(
        self,
        index_line: int,
        ax: None | matplotlib.axes.Axes = None,
        limit_velocity: None | u.Quantity = None,
        percentile_alpha: float = 99,
        correct_transmission: bool = True,
        drift: None | u.Quantity = None,
        interval: int = 200,
    ) -> matplotlib.animation.FuncAnimation:
        """
        Animate a Doppler map of the given line over the flight.

        The map encodes the intensity-weighted mean Doppler velocity of the
        line as color — blue toward the observer, red away — and the line
        intensity as opacity, so only regions with significant emission show
        color.

        Parameters
        ----------
        index_line
            The index of the spectral line along :attr:`axis_line`.
        ax
            The :class:`~matplotlib.axes.Axes` instance to use.
            If :obj:`None`, a new set of axes will be created.
        limit_velocity
            The Doppler velocity mapped to the ends of the colormap.
            If :obj:`None` (the default), a limit measured from the
            velocities that are actually visible; see
            :func:`esis.data._level_4._level_4._limit_velocity`.
        percentile_alpha
            The percentile of the intensity mapped to fully opaque.
        correct_transmission
            Whether to divide the intensity by the relative atmospheric
            transmission of each frame, so the opacity does not fade with
            the total signal toward the ends of the flight.
        drift
            The per-frame scene offset to undo, from :meth:`drift`.
        interval
            The delay between frames in milliseconds.
        """
        intensity = self.intensity[{self.axis_line: index_line}]
        velocity = self.velocity_mean[{self.axis_line: index_line}]

        if correct_transmission:
            intensity = intensity / self._transmission

        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)
        else:
            fig = ax.figure

        x = self.inputs.position.x.ndarray.to_value(u.arcsec)
        y = self.inputs.position.y.ndarray.to_value(u.arcsec)
        extent = (x[0], x[~0], y[0], y[~0])

        ax.set_title(self.label(index_line))

        cmap = matplotlib.colormaps["RdBu_r"]

        num_time = self.shape[self.axis_time]
        intensity_frames = [
            self._index_xy(intensity[{self.axis_time: t}]) for t in range(num_time)
        ]
        velocity_frames = [
            self._index_xy(velocity[{self.axis_time: t}].to(u.km / u.s))
            for t in range(num_time)
        ]
        intensity_frames = self._coregistered(intensity_frames, drift)
        velocity_frames = self._coregistered(velocity_frames, drift)
        alpha_reference = np.nanpercentile(np.stack(intensity_frames), percentile_alpha)

        limit = _limit_velocity(
            limit_velocity=limit_velocity,
            intensity_frames=intensity_frames,
            velocity_frames=velocity_frames,
            alpha_reference=alpha_reference,
        )
        norm = matplotlib.colors.Normalize(vmin=-limit, vmax=limit)

        def rgba(t: int) -> np.ndarray:
            """
            Build the RGBA image of the frame with the given time index.

            Parameters
            ----------
            t
                The index of the frame to draw.
            """
            result = cmap(norm(velocity_frames[t].T))
            alpha = intensity_frames[t].T / alpha_reference
            result[..., 3] = np.clip(np.nan_to_num(alpha), 0, 1)
            return result

        img = ax.imshow(
            rgba(0),
            extent=extent,
            origin="lower",
        )
        ax.set_xlabel("field $x$ (arcsec)")
        ax.set_ylabel("field $y$ (arcsec)")
        ax.set_aspect("equal")
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            label="mean Doppler velocity (km/s)",
        )

        def update(t: int) -> tuple:
            """
            Draw the frame with the given time index.

            Parameters
            ----------
            t
                The index of the frame to draw.
            """
            img.set_data(rgba(t))
            return (img,)

        return matplotlib.animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=num_time,
            interval=interval,
        )

    def to_jshtml(
        self,
        animation: matplotlib.animation.FuncAnimation,
        fps: None | float = None,
    ) -> IPython.display.HTML:
        """
        Convert an animation to Javascript ready to display in a notebook.

        Parameters
        ----------
        animation
            An animation created by :meth:`animate_intensity` or
            :meth:`animate_doppler`.
        fps
            The frames per second of the animation.
        """
        result = animation.to_jshtml(fps=fps)
        result = IPython.display.HTML(result)
        plt.close(animation._fig)
        return result

    def to_fits(
        self,
        path,
        overwrite: bool = False,
        split: bool = False,
    ) -> None:
        """
        Write the product to a self-describing FITS file.

        Each spectral line becomes one image extension holding a
        ``(time, velocity, y, x)`` cube with a full world coordinate
        system — helioprojective longitude and latitude in arcseconds,
        Doppler velocity in km/s about the rest wavelength of the line,
        and time in seconds from a reference epoch — so the file can be
        analyzed without this package.  The exact coordinate vertices and
        the per-frame diagnostics are stored alongside, so
        :meth:`from_fits` restores the product exactly;
        see :func:`esis.data._level_4._fits.to_fits`.

        Parameters
        ----------
        path
            The path of the file to write, or of the directory to fill if
            `split`.
        overwrite
            Whether to overwrite an existing file.
        split
            Whether to write one file per spectral line rather than one
            file holding every line.  Each file is a complete single-line
            product, so a reader interested in one line need not fetch
            the others.
        """
        from . import _fits

        return _fits.to_fits(self, path=path, overwrite=overwrite, split=split)

    @classmethod
    def from_fits(
        cls,
        path,
        instrument: None | esis.optics.Instrument = None,
    ) -> Self:
        """
        Read a product written by :meth:`to_fits`.

        The optical model is not stored in the file, so :attr:`instrument`
        is :obj:`None` unless supplied;
        see :func:`esis.data._level_4._fits.from_fits`.

        Parameters
        ----------
        path
            The path of the file to read.
        instrument
            A model of the optical system to attach to the result.
        """
        from . import _fits

        return _fits.from_fits(cls, path=path, instrument=instrument)

    def locate_event(
        self,
        index_line: None | int = None,
        center: None | u.Quantity = None,
        radius: u.Quantity = 330 * u.arcsec,
        sigma: float = 2,
    ) -> u.Quantity:
        """
        Locate the strongest compact Doppler event in the reconstruction.

        The event is found as the maximum over time of the smoothed product
        of the line intensity and the magnitude of the intensity-weighted
        mean Doppler velocity, restricted to the interior of the field of
        view; see :func:`esis.data._level_4._movies.locate_event`.

        Parameters
        ----------
        index_line
            The index of the spectral line to search along
            :attr:`axis_line`.
            If :obj:`None`, the last line (O V 630 for the baseline
            product).
        center
            The center of the field of view.
            If :obj:`None`, the center of the scene grid.
        radius
            The radius around `center` to search; the default excludes the
            region outside the octagonal field stop of ESIS-I.
        sigma
            The width of the Gaussian smoothing, in scene cells.
        """
        from . import _movies

        return _movies.locate_event(
            self,
            index_line=index_line,
            center=center,
            radius=radius,
            sigma=sigma,
        )

    def animate_event(
        self,
        position: u.Quantity,
        halfwidth: u.Quantity = 40 * u.arcsec,
        context: None | dict[str, na.FunctionArray] = None,
        cmaps_context: None | dict[str, str] = None,
        labels: None | list[str] = None,
        limit_velocity: u.Quantity = 80 * u.km / u.s,
        percentile_max: float = 99.5,
        percentile_alpha: float = 99,
        correct_transmission: bool = True,
        drift: None | u.Quantity = None,
        interval: int = 200,
    ) -> matplotlib.animation.FuncAnimation:
        """
        Animate an event: intensity and Doppler maps of every line.

        The figure has one column per spectral line and three rows: the
        total intensity of each line, the Doppler map of each line, and —
        if `context` is given — co-temporal context images (e.g. AIA
        channels), each shown at the frame nearest in time to the current
        Level-4 frame; see :func:`esis.data._level_4._movies.animate_event`.

        Parameters
        ----------
        position
            The center of the event, in scene coordinates.
        halfwidth
            The half-width of the field of view of the movie.
        context
            A mapping from label to a context-image function of
            ``(time, x, y)``, on the same coordinate frame as the scene,
            whose time, horizontal, and vertical coordinates are
            one-dimensional vertex arrays along their own logical axes.
        cmaps_context
            An optional mapping from context label to colormap name.
        labels
            The label of each spectral line.
            If :obj:`None`, the rest wavelength of each line is used.
        limit_velocity
            The Doppler velocity mapped to the ends of the colormap.
        percentile_max
            The percentile of the in-frame intensity mapped to the top of
            the intensity colormap.
        percentile_alpha
            The percentile of the in-frame intensity mapped to fully opaque
            in the Doppler maps.
        correct_transmission
            Whether to divide the intensity by the relative atmospheric
            transmission of each frame.
        drift
            The per-frame scene offset to undo, from :meth:`drift`.
        interval
            The delay between frames in milliseconds.
        """
        from . import _movies

        return _movies.animate_event(
            self,
            position=position,
            halfwidth=halfwidth,
            context=context,
            cmaps_context=cmaps_context,
            labels=labels,
            limit_velocity=limit_velocity,
            percentile_max=percentile_max,
            percentile_alpha=percentile_alpha,
            correct_transmission=correct_transmission,
            drift=drift,
            interval=interval,
        )


def _peak_subpixel(correlation: np.ndarray) -> np.ndarray:
    """
    Locate the peak of a cyclic correlation to sub-cell precision.

    The integer peak is refined by fitting a parabola to its immediate
    neighbours along each axis, and the result is wrapped into the
    half-open interval centred on zero, since the correlation of a
    periodic transform places negative offsets at the far end.

    Parameters
    ----------
    correlation
        The correlation image, with zero offset at the origin.
    """
    shape = np.array(correlation.shape)
    peak = np.array(np.unravel_index(np.argmax(correlation), correlation.shape))

    offset = peak.astype(float)
    for axis in range(2):
        index_low = peak.copy()
        index_high = peak.copy()
        index_low[axis] = (peak[axis] - 1) % shape[axis]
        index_high[axis] = (peak[axis] + 1) % shape[axis]
        low = correlation[tuple(index_low)]
        middle = correlation[tuple(peak)]
        high = correlation[tuple(index_high)]
        denominator = low - 2 * middle + high
        if denominator != 0:
            offset[axis] += 0.5 * (low - high) / denominator

    return np.where(offset > shape / 2, offset - shape, offset)


def _limit_velocity(
    limit_velocity: None | u.Quantity,
    intensity_frames: list[np.ndarray],
    velocity_frames: list[np.ndarray],
    alpha_reference: float,
    fraction_visible: float = 0.2,
    percentile: float = 99,
    headroom: float = 2,
    minimum: u.Quantity = 20 * u.km / u.s,
) -> float:
    """
    Choose the Doppler velocity mapped to the ends of the colormap.

    A fixed limit has to be guessed, and guessing low silently flattens
    the fastest flows into a single saturated color while guessing high
    wastes the colormap on an empty range.  The mean velocity is a ratio,
    so faint pixels carry wild values that say nothing about the plasma;
    those pixels are also nearly transparent in the map.  The limit is
    therefore measured only over the pixels bright enough to be visible.

    Parameters
    ----------
    limit_velocity
        An explicit limit, returned as-is; :obj:`None` to measure one.
    intensity_frames
        The line intensity of each frame.
    velocity_frames
        The mean Doppler velocity of each frame, in km/s.
    alpha_reference
        The intensity mapped to fully opaque.
    fraction_visible
        The opacity below which a pixel is too faint to read, and so is
        excluded from the measurement.
    percentile
        The percentile of the visible speeds taken as the typical fast
        flow.
    headroom
        How far past that speed to carry the colormap.  Mapping the
        percentile itself to the end saturates every feature at or above
        it, which is exactly the structure worth looking at; a factor of
        two keeps the common flows in the middle of the map and still
        resolves the fast tail.
    minimum
        The smallest limit to return, so a quiet line still gets a
        sensible range.
    """
    if limit_velocity is not None:
        return limit_velocity.to_value(u.km / u.s)

    intensity = np.stack(intensity_frames)
    velocity = np.abs(np.stack(velocity_frames))
    visible = (intensity / alpha_reference) > fraction_visible
    visible &= np.isfinite(velocity)

    if not visible.any():
        return minimum.to_value(u.km / u.s)

    result = headroom * np.percentile(velocity[visible], percentile)
    return float(max(result, minimum.to_value(u.km / u.s)))


def _filter_weights_shadow(
    weights: tuple,
    keep_flat: np.ndarray,
    axis_channel: str,
    index_detector: int,
) -> tuple:
    """
    Remove weight triples that reference shaded detector pixels.

    Each element of the weights table is a ``(indices_input, indices_output,
    values)`` triple; the triples whose detector index falls on a shaded
    pixel are deleted outright, so the forward model can place no flux on
    those pixels and the backprojection never reads them.

    Parameters
    ----------
    weights
        The ``(table, shape_input, shape_output)`` weights tuple, where the
        table is an array of triples over the channel and wavelength axes.
    keep_flat
        A boolean array of shape ``(num_channel, num_detector_pixel)``,
        :obj:`True` for the detector pixels to keep, with the detector axes
        flattened in the same C order as the weight indices.
    axis_channel
        The name of the logical axis of the table corresponding to changing
        channel.
    index_detector
        The member of each triple holding detector indices: ``1`` (the
        output indices) for the forward weights, ``0`` (the input indices)
        for the transpose weights.
    """
    table, shape_input, shape_output = weights

    axes = table.axes
    index_channel = axes.index(axis_channel)
    view = np.moveaxis(table.ndarray, index_channel, 0)

    result = np.empty_like(view)
    for c in range(view.shape[0]):
        keep_c = keep_flat[c]
        for j in range(view.shape[1]):
            idx_input, idx_output, values = view[c, j]
            keep = keep_c[(idx_input, idx_output)[index_detector]]
            result[c, j] = (
                np.ascontiguousarray(idx_input[keep]),
                np.ascontiguousarray(idx_output[keep]),
                np.ascontiguousarray(values[keep]),
            )
    result = np.moveaxis(result, 0, index_channel)

    return na.ScalarArray(result, axes=axes), shape_input, shape_output


def _plate_scale(
    system,
    key: str,
    code: str,
    wavelength_center: na.AbstractScalarArray,
    limit_velocity: u.Quantity,
    degree: int,
    axis_wavelength: str,
    axis_line: str,
) -> tuple[u.Quantity, u.Quantity]:
    """
    Measure the nominal plate scale and dispersion of the given system.

    The system is linearized on a coarse wavelength grid (the edges and
    centers of each line window, cached like the production linearization)
    and the returned distortion model is finite-differenced at the center of
    the field of view.  The minimum over channels, lines, and axes is
    returned, so the scene grid samples no coarser than the finest scale the
    instrument delivers anywhere.

    Parameters
    ----------
    system
        The sequential system to measure.
    key
        The fingerprint of `system` computed by
        :func:`esis.data._level_4._caching.key_system`.
    code
        The result of :func:`esis.data._level_4._caching.code_state`.
    wavelength_center
        The rest wavelength of each spectral line, arranged along
        `axis_line`.
    limit_velocity
        The half-width of the Doppler window of each line.
    degree
        The degree of the polynomial distortion model.
    axis_wavelength
        The name of the logical axis corresponding to changing wavelength.
    axis_line
        The logical axis of `wavelength_center` corresponding to the
        different lines.
    """
    velocity = na.linspace(
        start=-limit_velocity,
        stop=limit_velocity,
        axis=axis_wavelength,
        num=3,
    )
    wavelength = wavelength_center * (1 + velocity / astropy.constants.c)
    wavelength = wavelength.to(u.AA).combine_axes(
        axes=(axis_line, axis_wavelength),
        axis_new=axis_wavelength,
    )

    linear = _caching.linear_system(
        system,
        key=key,
        wavelength=wavelength,
        degree=degree,
        code=code,
    )

    field = system.rayfunction_default.inputs.field
    center = (field.max() + field.min()) / 2

    d_angle = 1 * u.arcsec
    d_velocity = 10 * u.km / u.s
    d_wavelength = wavelength_center * (d_velocity / astropy.constants.c).to_value(
        u.dimensionless_unscaled
    )

    def distort(
        wavelength: na.AbstractScalarArray,
        position: na.AbstractCartesian2dVectorArray,
    ) -> na.AbstractCartesian2dVectorArray:
        """
        Map the given scene coordinates onto the detector.

        Parameters
        ----------
        wavelength
            The wavelength coordinate of the scene.
        position
            The field coordinate of the scene.
        """
        coordinates = na.SpectralPositionalVectorArray(
            wavelength=wavelength,
            position=position,
        )
        return linear.distortion.distort(coordinates).position

    step_x = na.Cartesian2dVectorArray(d_angle, 0 * u.arcsec)
    step_y = na.Cartesian2dVectorArray(0 * u.arcsec, d_angle)

    p0 = distort(wavelength_center, center)
    px = distort(wavelength_center, center + step_x)
    py = distort(wavelength_center, center + step_y)
    pw = distort(wavelength_center + d_wavelength, center)

    pitch_x = d_angle / (px - p0).length
    pitch_y = d_angle / (py - p0).length
    dispersion = d_velocity / (pw - p0).length

    pitch = np.minimum(pitch_x, pitch_y).min()
    pitch = pitch.ndarray.to(u.arcsec / u.pix) * u.pix

    dispersion = dispersion.min()
    dispersion = dispersion.ndarray.to(u.km / u.s / u.pix) * u.pix

    return pitch, dispersion


def _guess(
    instrument_mart,
    electrons_reference: na.AbstractScalarArray,
    width_doppler: na.AbstractScalarArray,
    velocity: na.AbstractScalarArray,
    num_velocity: int,
    floor_guess: float,
    axis_wavelength: str,
    axis_line: str,
) -> na.AbstractScalarArray:
    """
    Build the initial guess for the reference frame.

    The guess is spatially uniform with a Gaussian spectral profile in each
    line window, centered on the rest wavelength with the measured Doppler
    width of the line, floored at `floor_guess` of the peak, and scaled so
    the total forward-modeled signal matches the total observed signal.

    Parameters
    ----------
    instrument_mart
        The :class:`ctis.instruments.OptikaInstrument` the guess is for.
    electrons_reference
        The observed (clipped and normalized) images of the reference frame.
    width_doppler
        The Doppler width of each spectral line, arranged along `axis_line`.
    velocity
        The vertices of the Doppler-velocity grid of each line window.
    num_velocity
        The number of velocity bins in the Doppler window of each line.
    floor_guess
        The floor of the profile, as a fraction of the peak.
    axis_wavelength
        The name of the logical axis corresponding to changing wavelength.
    axis_line
        The logical axis of `width_doppler` corresponding to the different
        lines.
    """
    velocity_center = (
        velocity[{axis_wavelength: slice(None, ~0)}]
        + velocity[{axis_wavelength: slice(1, None)}]
    ) / 2

    gaussian = np.exp(-np.square(velocity_center / width_doppler) / 2)

    shape_scene = {
        axis: num
        for axis, num in instrument_mart.weights[1].items()
        if axis != instrument_mart.axis_channel
    }

    profile = np.full(shape_scene[axis_wavelength], floor_guess)
    for i in range(gaussian.shape[axis_line]):
        j = slice(i * (num_velocity + 1), (i + 1) * (num_velocity + 1) - 1)
        profile[j] = np.maximum(
            np.asarray(gaussian[{axis_line: i}].ndarray),
            floor_guess,
        )
    profile = na.ScalarArray(profile, axes=(axis_wavelength,))

    seed = (
        na.ScalarArray(
            ndarray=np.ones(tuple(shape_scene.values())),
            axes=tuple(shape_scene),
        )
        * profile
    )

    images = na.FunctionArray(
        inputs=instrument_mart.coordinates_sensor,
        outputs=electrons_reference,
    )

    backprojected = instrument_mart.backproject(images.outputs)
    unit_scene = na.unit(backprojected.outputs)

    image_seed = instrument_mart.image(seed * unit_scene, noise=False)
    scale = images.outputs.sum() / image_seed.outputs.sum()

    return seed * scale * unit_scene
