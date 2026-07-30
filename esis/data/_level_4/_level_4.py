from typing import Self
import time as _time
import dataclasses
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

    num_velocity: None | int = None
    """The number of velocity bins in the Doppler window of each line."""

    mean_chi_squared: None | na.AbstractScalarArray = None
    r"""The final mean :math:`\chi^2` of each channel for each frame."""

    num_iteration: None | na.AbstractScalarArray = None
    """The number of MART iterations performed for each frame."""

    factor_norm: None | na.AbstractScalarArray = None
    """The channel-normalization factor applied to each image."""

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
        instrument: None | esis.optics.Instrument = None,
        limit_velocity: u.Quantity = 200 * u.km / u.s,
        num_velocity: None | int = None,
        pitch_scene: None | u.Quantity = None,
        factor_fov: float = 1.25,
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

        electrons = np.maximum(a.outputs, 0)
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
            num_velocity=num_velocity,
            mean_chi_squared=na.stack(chi_squared, axis=axis_time),
            num_iteration=na.ScalarArray(np.array(iterations), axes=(axis_time,)),
            factor_norm=factor_norm,
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

    def animate_intensity(
        self,
        index_line: int,
        ax: None | matplotlib.axes.Axes = None,
        percentile_max: float = 99.5,
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

        frames = [
            self._index_xy(intensity[{self.axis_time: t}])
            for t in range(self.shape[self.axis_time])
        ]
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
        limit_velocity: u.Quantity = 80 * u.km / u.s,
        percentile_alpha: float = 99,
        correct_transmission: bool = True,
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
        percentile_alpha
            The percentile of the intensity mapped to fully opaque.
        correct_transmission
            Whether to divide the intensity by the relative atmospheric
            transmission of each frame, so the opacity does not fade with
            the total signal toward the ends of the flight.
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

        limit = limit_velocity.to_value(u.km / u.s)
        norm = matplotlib.colors.Normalize(vmin=-limit, vmax=limit)
        cmap = matplotlib.colormaps["RdBu_r"]

        num_time = self.shape[self.axis_time]
        intensity_frames = [
            self._index_xy(intensity[{self.axis_time: t}]) for t in range(num_time)
        ]
        velocity_frames = [
            self._index_xy(velocity[{self.axis_time: t}].to(u.km / u.s))
            for t in range(num_time)
        ]
        alpha_reference = np.nanpercentile(np.stack(intensity_frames), percentile_alpha)

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
