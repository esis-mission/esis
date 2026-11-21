#!/usr/bin/env python3
"""
Register the channels of a finished fit with one another, empirically.

A check of the fit rather than a stage of it, kept as the study it was.  The
per-sector focus (:func:`esis.flights.f1.optics.fit_defocus_history`) takes
out the motion of the channels' skies against one another; this measures
what is left, frame by frame, and fits it as an offset of each channel's own
pointing: a polynomial in the frame index, zero at the reference frame, in
each channel's pitch and yaw, twelve numbers in all.  On the committed fit the
offsets are at most 0.15 arcsec (0.20 px) and take the registration from 0.11
to 0.09 px at He I and from 0.08 to 0.06 px at O V, with no mechanism behind
them, so the fit does not apply them; the paper quotes the shifts before and
after.  From this directory::

    python coregistration.py <pass directory>

writes ``coregistration.ecsv`` into the directory of the pass, with the
options of ``reproduce.py`` (``--seed``, ``--num-scene``) read from the same
environment variables.
"""

from __future__ import annotations
import copy
import os
import pathlib
import sys

import numpy as np
import astropy.units as u
import astropy.table
import esis
from esis.optics._distortions import _alignment
from esis.flights.f1.optics._fits._fits import (
    _base,
    _fit_through_reference,
    _frame,
    _frames_by_axis,
    _image_jacobians,
    _logger,
    _pixels_per_arcsec,
    _read_table,
    _stamp,
    _wavelengths_alignment,
    frame_parameters,
)
import esis.flights.f1.optics._fits._fits as _fits


def _pointing_pattern(
    instrument: esis.optics.Instrument,
    parameters: list[esis.optics.DistortionParameters],
    wavelengths: dict[str, u.Quantity],
    anchor: int = 1,
    step: u.Quantity = 1 * u.arcsec,
    seed: int = 0,
) -> np.ndarray:
    """
    Measure how a pointing offset of one channel moves that channel's sky.

    The same finite difference as :func:`_defocus_pattern`, in the channel's
    own pitch and yaw: where a point of the sky lands on the sensor before
    and after, taken back to the sky and expressed in the anchor's pixels.

    Parameters
    ----------
    instrument
        The instrument model.
    parameters
        The parameters of every channel.
    wavelengths
        The lines to measure at; the pattern is their mean.
    anchor
        The channel whose pixels the shifts are expressed in.
    step
        The offset of the finite difference.
    seed
        The seed of the samples every channel is linearized on, see
        :class:`esis.optics.LinearMerit`.

    Returns
    -------
    An array indexed ``[channel, axis, term]``: the shift on the sky in
    detector pixels, along each axis of the sky, of each channel per
    arcsecond of its own pitch and of its own yaw.
    """

    def where(c, name, d):
        p = copy.copy(parameters[c])
        setattr(p, name, getattr(p, name) + d)
        return _image_jacobians(p, instrument[dict(channel=c)], wavelengths, seed)

    num_channel = len(parameters)
    result = np.zeros((num_channel, 2, 2))
    scale = None
    for c in range(num_channel):
        before = where(c, "pitch", 0 * step)
        if c == anchor:
            scale = _pixels_per_arcsec(before)
        for k, name in enumerate(("pitch", "yaw")):
            after = where(c, name, step)
            result[c, :, k] = np.mean(
                [
                    np.linalg.solve(j, s1 - s0)
                    for (s0, j), (s1, _) in zip(before, after)
                ],
                axis=0,
            )
    return result * scale[None, :, None] / step.to_value(u.arcsec)


def fit_coregistration(
    reference: str | pathlib.Path | esis.optics.DistortionParameters,
    pointing: str | pathlib.Path | astropy.table.QTable,
    num_scene: int = 401,
    instrument: None | esis.optics.Instrument = None,
    frames: None | tuple[int, ...] = None,
    frame_reference: int = 15,
    degree: int = 2,
    anchor: int = 1,
    num_pass: int = 2,
    path: None | str | pathlib.Path = None,
    directory: None | str | pathlib.Path = None,
    seed: int = 0,
) -> astropy.table.QTable:  # pragma: nocover
    """
    Register the channels with one another empirically through the flight.

    A check on the chain rather than a stage of it: the per-sector focus of
    :func:`fit_defocus_history` takes out the motion of the channels' skies
    against one another, and this measures whatever is left as a pointing
    offset of each channel.  In every frame the channels' skies are measured
    against the anchor's (:func:`esis.optics.measure_channel_shifts`), the
    shift of each channel is converted to the offset of its own pitch and
    yaw that removes it (:func:`_pointing_pattern`), and a polynomial in the
    frame index that vanishes at the reference frame is fit through the
    flight (:func:`_fit_through_reference`).  The offsets are taken about
    their mean over the channels, so that the payload's pointing is left
    where the scene put it.  A second pass measures the channels again with
    the first fit applied and corrects it.

    Parameters
    ----------
    reference
        The reference parameters, or the path of their table.
    pointing
        The pointing table, or its path, see :func:`fit_distortion_pointing`.
    num_scene
        The number of samples along each axis of the resampled AIA scene,
        whose extent sets the sky grid.
    instrument
        The base model.  If :obj:`None`, the idealized as-built model.
    frames
        The frames to measure.  If :obj:`None`, every frame.
    frame_reference
        The frame the reference was fit to, where the offsets are zero.
    degree
        The degree of the polynomial in the frame index.
    anchor
        The channel the others are measured against.
    num_pass
        The number of times the channels are measured.
    path
        If given, the history is written here as an ECSV table.
    directory
        A directory where the progress is logged.
    seed
        The seed of the samples every channel is linearized on, see
        :class:`esis.optics.LinearMerit`.

    Returns
    -------
    One row per frame: the shift of every channel against the anchor as
    first measured and as measured in the last pass, the offsets that
    remove them, and ``pitch_channel`` and ``yaw_channel``, the fitted
    offsets of each channel's own pointing.
    """
    if not isinstance(reference, esis.optics.DistortionParameters):
        reference = esis.optics.DistortionParameters.from_file(reference)
    pointing = _read_table(pointing)
    instrument = _base(instrument)
    num_channel = instrument.camera.channel.shape["channel"]
    if frames is None:
        frames = tuple(range(esis.flights.f1.data.level_1().shape["time"]))
    frames = tuple(sorted(frames))
    log = _logger(directory, "coregistration")
    wavelengths = _wavelengths_alignment()
    others = [c for c in range(num_channel) if c != anchor]

    pattern = _pointing_pattern(
        instrument,
        [
            frame_parameters(reference, pointing, frame_reference, c)
            for c in range(num_channel)
        ],
        wavelengths,
        anchor,
        seed=seed,
    )
    log(
        "sky shift per arcsecond of a channel's own pitch; yaw [px]: "
        + " | ".join(
            f"{p[0, 0]:+.2f}, {p[1, 0]:+.2f}; {p[0, 1]:+.2f}, {p[1, 1]:+.2f}"
            for p in pattern
        )
    )

    data = {}
    for t in frames:
        scene, observation = _frame(t, num_scene)
        # copies, so that the Level-1 cube each frame was cut from is freed
        data[t] = (
            [np.array(f) for f in _frames_by_axis(observation, num_channel)],
            _alignment.sky_grid(scene.inputs.position, 1600),
        )
        del scene, observation

    num = len(frames)
    # the motion each channel's image needs against the anchor's, in pixels
    # on the sky: as measured in each frame, and its fit
    motion = np.zeros((num, num_channel, 2))
    relative = np.zeros((num, num_channel, 2))
    offsets = np.zeros((num, num_channel, 2))  # arcseconds of pitch and yaw
    used = np.ones((num, num_channel, 2), dtype=bool)
    scatter = np.zeros((num_channel, 2))
    shifts_first = np.zeros((num, num_channel, 2))
    shifts_last = np.zeros((num, num_channel, 2))
    for k in range(num_pass):
        for i, t in enumerate(frames):
            frames_data, sky = data[t]
            linears = []
            for c in range(num_channel):
                p = frame_parameters(reference, pointing, t, c)
                p.pitch = p.pitch + offsets[i, c, 0] * u.arcsec
                p.yaw = p.yaw + offsets[i, c, 1] * u.arcsec
                model = p.to_instrument(instrument[dict(channel=c)])
                linears.append(
                    model.system.linearize(
                        wavelength=model.wavelength,
                        degree=2,
                        field_stop=True,
                        seed_field=seed,
                        seed_pupil=seed,
                    )
                )
            fields, scales = esis.optics.measure_channel_shifts(
                linears, frames_data, sky, wavelengths, anchor=anchor
            )
            medians = esis.optics.median_shifts(fields, scales)
            for c in others:
                shifts_last[i, c] = medians[c]
                if k == 0:
                    shifts_first[i, c] = medians[c]
                # a measured shift s means the anchor shows at r + s what the
                # channel shows at r (see shift_fft): the channel's image sits
                # at -s, on top of the motion already applied
                motion[i, c] = relative[i, c] - medians[c]
            log(
                f"pass {k}, frame {t}: shifts vs channel {anchor} "
                + "; ".join(
                    f"{medians[c][0]:+.2f}, {medians[c][1]:+.2f}" for c in others
                )
                + " px"
            )
        rms = float(np.sqrt(np.nanmean(shifts_last[:, others] ** 2)))
        log(
            f"pass {k}: rms shift of the channels against channel {anchor} {rms:.3f} px"
        )
        for c in others:
            for j in range(2):
                relative[:, c, j], used[:, c, j], scatter[c, j] = (
                    _fit_through_reference(
                        np.array(frames),
                        motion[:, c, j],
                        frame_reference,
                        degree,
                        floor=0.05,
                    )
                )
        # about the mean of the channels, so the payload's pointing stays put
        centered = relative - relative.mean(axis=1, keepdims=True)
        for c in range(num_channel):
            offsets[:, c] = np.linalg.solve(pattern[c], centered[:, c].T).T
        log(
            f"pass {k}: offsets at the first and the last frame, pitch; yaw [arcsec]: "
            + " | ".join(
                f"ch{c} {offsets[0, c, 0]:+.2f}, {offsets[-1, c, 0]:+.2f}; "
                f"{offsets[0, c, 1]:+.2f}, {offsets[-1, c, 1]:+.2f}"
                for c in range(num_channel)
            )
            + "; scatter of the measured motion about the fit [px]: "
            + ", ".join(f"{np.hypot(*scatter[c]):.3f}" for c in others)
        )

    table = astropy.table.QTable(
        dict(
            frame=np.array(frames),
            shift_x=shifts_first[..., 0] * u.pix,
            shift_y=shifts_first[..., 1] * u.pix,
            residual_x=shifts_last[..., 0] * u.pix,
            residual_y=shifts_last[..., 1] * u.pix,
            motion_x=motion[..., 0] * u.pix,
            motion_y=motion[..., 1] * u.pix,
            used=used.all(axis=-1),
            pitch_channel=offsets[..., 0] * u.arcsec,
            yaw_channel=offsets[..., 1] * u.arcsec,
        )
    )
    table.meta.update(
        dict(
            description=(
                "The registration of the ESIS-I channels with one another "
                "through the flight.  shift is the median tile shift of every "
                "channel's sky against the anchor's with the pointing, the "
                "window drift and the focus of each sector applied, in detector "
                "pixels, and "
                "residual the same in the last pass, measured with the "
                "previous pass's offsets applied.  motion is the motion on the "
                "sky, in detector pixels, that each channel's image needs "
                "against the anchor's in each frame; a polynomial in the frame "
                "index that vanishes at the reference frame is fit through "
                "it, taken about the mean of the channels, and expressed as "
                "pitch_channel and yaw_channel, the offset of each channel's "
                "own pointing.  used marks the frames the fit kept.  This is "
                "what the focus of each sector leaves, an empirical term the "
                "fit does not apply."
            ),
            provenance=(
                "checks/coregistration.py of the distortion report "
                f"(num_scene={num_scene}, "
                f"degree={degree}, anchor={anchor}, num_pass={num_pass})"
            ),
            frame_reference=int(frame_reference),
            anchor=int(anchor),
            pattern=[[[float(v) for v in row] for row in p] for p in pattern],
            scatter=[[float(v) for v in row] for row in scatter],
            rms_first=float(np.sqrt(np.nanmean(shifts_first[:, others] ** 2))),
            rms_last=float(np.sqrt(np.nanmean(shifts_last[:, others] ** 2))),
            **_stamp(seed),
        )
    )
    if path is not None:
        table.write(path, format="ascii.ecsv", overwrite=True)
    return table


if __name__ == "__main__":
    directory = pathlib.Path(sys.argv[1])
    stop = directory / "field_stop_input.ecsv"
    _fits.path_field_stop = stop if stop.exists() else None
    fit_coregistration(
        reference=directory / "distortion_reference.ecsv",
        pointing=directory / "distortion_pointing.ecsv",
        num_scene=int(os.environ.get("ESIS_NUM_SCENE", 401)),
        path=directory / "coregistration.ecsv",
        directory=directory,
        seed=int(os.environ.get("ESIS_SEED", 0)),
    )
