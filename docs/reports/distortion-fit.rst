Distortion fit
==============

How the distortion of ESIS-I is fit to the flight data, and how to reproduce
the fit committed to the package.

What the data determine
-----------------------

One Level-1 frame determines eight combinations of the parameters of a
channel: two translations, the dispersion, two scales, a rotation, a shear
and one more.  That is the result of a singular-value decomposition of the
Jacobian of the sky-to-sensor mapping with respect to the fifteen optical
parameters, sampled over the field at the three bright lines: eight singular
values above forty pixels per half-box, then a cliff to below three.  The
null directions are what one would guess.  The roll of the field stop moves
only the edges of the windows.  The displacement of the primary trades
against the focus-preserving distance of the sensor.  The in-plane position
of the sensor trades against the grating's yaw and pitch, and both trade
against the pointing: the mapping cannot tell a moved sky from a moved
grating.  A global search over a set that contains null directions lands
somewhere along them at random, and two seeds disagree by tens of pixels;
over the nine terms that span the eight combinations without a null
direction, two seeds agree to two hundredths in merit.

Two things break the degeneracies the frame leaves.  The edges of the
windows are the image of the field stop through the optics, and do not move
when the pointing does, so they say where the grating and sensor put the
image and leave the pointing to explain the rest.  Measured in every frame
of the flight, they move by less than half a pixel in :math:`x` while the
pointing sweeps seven arcseconds, and drift smoothly by two pixels in
:math:`y` end to end, top and bottom together, which is the optics moving
and is carried per frame.  And the channels compared with one another on
the sky resolve a tenth of a pixel where the proxy scene resolves one.

What the data do not determine is stated rather than fit.  The angle
between the windows and the sky, which a roll of the instrument or of the
field stop would set, changes the merit by less than three thousandths over
four degrees and is held at the design.  The instrument roll is one number
for the whole instrument and is held at zero; a per-channel value would be
the azimuth of that channel's arm, which turns the window and the sky
together and is the sensor's roll by another name.  The primary's
displacement is the one shared quantity the windows do measure, see below.

The merit
---------

The fit compares an image of the Sun seen by another instrument (the AIA
proxy scene, :func:`esis.flights.f1.data.synth.scene_aia`) with a Level-1
frame, after imaging the scene through the model of one channel.  The
image is made by :class:`esis.optics.LinearMerit`: the channel is
linearized (:meth:`optika.systems.SequentialSystem.linearize`) with its
field of view carried as the polygon its stop rays trace, so that every
scene cell outside the field stop is blocked before the lines are summed,
and the scene is regridded conservatively onto the sensor.  The merit is
the Pearson correlation with the frame, or the least-squares score of
:func:`esis.optics.least_squares`, which compares absolute intensities
after the level of the frame outside every window is subtracted and lets
one :attr:`~esis.optics.DistortionParameters.degradation` per channel set
the level of the image.  Both give the same geometry within the
acceptance test's noise; the least-squares score delivers the four
degradations, which reproduce the factor of 1.6 between the channel pairs
seen in the photon budget.

The edge pixels are 0.37 % of a window.  Moving only the edges by a pixel
changes the correlation by 0.0003 to 0.0014; moving the whole image by a
pixel changes it by 0.005.  The merit therefore cannot place the windows,
which is why the edges are measured directly.

The stages
----------

:func:`esis.flights.f1.optics.fit_distortion_reference` runs the stages
from the as-built model (:func:`esis.flights.f1.optics.as_built`) and the
data alone; the free set of every stage is recorded in the table it
writes.

1.  **Absolute.**  Each channel is fit on its own with
    :func:`esis.optics.fit_distortion`: a seeded differential evolution
    over the nine terms one frame determines (grating yaw and pitch, ruling
    spacing, pointing pitch and yaw, and the sensor's focus-preserving
    distance, roll, pitch and yaw), eighty generations of fifteen members
    per dimension, then a restarted Nelder-Mead polish.  Every other term
    stays at the as-built model.

2.  **Edges.**  :func:`esis.flights.f1.optics.measure_window_edges` fits an
    error-function step where each window's predicted outline crosses the
    rows and columns of every frame whose signal is at least half the
    flight's median, and takes the median over the frames per crossing,
    with the per-frame departures kept as the drift of the windows.  The
    middle window sits on its neighbours and is left out.

3.  **Outline.**  :func:`esis.optics.fit_distortion_outline` places each
    channel's windows on the median edges through the grating's yaw and
    pitch and the sensor's roll, which move the windows without moving the
    sky within them, then re-polishes the sky terms against the merit,
    twice.

4.  **Primary.**  Displacing the primary changes the scale of the sky on
    the sensor and not the size of the windows; the focus-preserving sensor
    move does the opposite.  With the sky's scale pinned by the merit, the
    width of the windows against the edges therefore fixes the one
    displacement the instrument has.  The scan tries a grid of
    displacements, re-polishes every channel's scale terms at each, and
    takes the one whose windows have the measured width.

5.  **Shared.**  The primary displacement, the field-stop roll and the
    payload pitch and yaw belong to the instrument.  The pointing is set to
    its mean over the channels, the others to what the stages above left,
    and each channel's grating and camera placement is polished again with
    them fixed, the channels side by side in worker processes.

6.  **Internal.**  :func:`esis.optics.align_channels` samples every
    channel's frame on a common sky grid through its own distortion at
    He I 584 and O V 630, measures the tile shifts against channel 1, and
    solves for the increments of the same terms whose change of the mapping
    reproduces the shifts.  Using two lines separates a geometric error
    from a dispersion error.

7.  **Pointing.**  :func:`esis.flights.f1.optics.fit_distortion_pointing`
    polishes the pitch, yaw and roll of every frame on the mean merit of
    the four channels, with each channel's grating offset by that frame's
    measured window drift first; both ride along in the pointing table for
    :func:`esis.flights.f1.optics.distortion_fit` to apply.

8.  **Acceptance.**  :func:`esis.flights.f1.optics.acceptance` scores the
    result on frames it was not fit to: the merit with each frame's
    pointing applied, the median tile shift of every channel against
    channel 1 at each aligned line, and the residual of the window outlines
    and widths against the measured edges.  No inversion is involved.

Reproducing the committed fit
-----------------------------

The scripts under ``docs/reports/distortion_fit`` run the stages on a
Slurm cluster, in this order::

    sbatch --job-name=dist-channel --array=0-3 reproduce.sbatch channel <dir>
    sbatch --job-name=dist-edges reproduce.sbatch edges <dir>
    sbatch --job-name=dist-combine reproduce.sbatch combine <dir>
    sbatch --job-name=dist-pointing --array=0-29 reproduce.sbatch pointing <dir>
    sbatch --job-name=dist-gather reproduce.sbatch gather <dir>
    sbatch --job-name=dist-accept reproduce.sbatch accept <dir>
    sbatch --job-name=dist-blink reproduce.sbatch blink <dir>

The first produces one ECSV per channel; ``edges`` the median edges and
the drift; ``combine`` runs the outline, primary, shared and internal
stages into the committed ``distortion_reference.ecsv``; ``pointing`` and
``gather`` the committed ``distortion_pointing.ecsv``; ``accept`` the
acceptance table; and ``blink`` renders every frame beside its model for
``blink.py page`` to bind into a page that blinks between them.  The
environment needs the ``optika`` that carries the field of view on a
linearized system and moves the optics rather than the object under a
roll, ``named_arrays`` and ``regridding`` with device support, and a
``numba`` that can see the CUDA driver; the exact versions the committed
tables were made with are in their headers.  Loading the Level-1 frames
peaks above 100 GB of memory, which the job script asks for.  End to end
the chain is about a working day on the cluster: three to five hours for
the capture, an hour for the edges, two to three for the combine, an hour
for the pointing.

Results
-------

The correlation of each channel with its frame after each stage of the
committed chain, run on the cluster on 2026-09-24 (the as-built row is the
starting model, the absolute row the capture of the four channel jobs):

=============  ======  ======  ======  ======
stage             ch0     ch1     ch2     ch3
=============  ======  ======  ======  ======
as-built        0.350   0.419   0.404   0.379
absolute        0.799   0.840   0.794   0.816
outline         0.799   0.839   0.794   0.815
edges [px]       2.43    2.18    1.90    2.35
shared          0.805   0.843   0.793   0.820
aligned [px]     0.16    0.00    0.20    0.24
=============  ======  ======  ======  ======

The merit is the correlation.  The primary displacement is held at nominal, the field-stop roll and the instrument roll at zero,
and the pointing is shared: pitch -21.8, yaw -17.5 arcsec
from the as-built model.  The free set of every stage, the seed and
schedule of the capture, the per-stage scores, the package commits and
where the windows land are in the table's header.

Acceptance
----------

The reference scored on frames it was not fit to, with each frame's
pointing applied: the correlation of every channel, and the median
tile shift of every channel against channel 1 at each aligned line in
pixels, He I first then O V.

=====  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======
frame  corr 0  corr 1  corr 2  corr 3  He I 0  He I 1  He I 2  He I 3   O V 0   O V 1   O V 2   O V 3
=====  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======
    9   0.788   0.842   0.792   0.816    0.27    0.00    0.31    0.41    0.20    0.00    0.20    0.32
   12   0.794   0.837   0.786   0.811    0.22    0.00    0.27    0.37    0.18    0.00    0.16    0.29
   15   0.801   0.843   0.789   0.813    0.23    0.00    0.32    0.36    0.17    0.00    0.14    0.26
   18   0.797   0.838   0.788   0.815    0.13    0.00    0.38    0.30    0.23    0.00    0.17    0.18
   21   0.797   0.837   0.788   0.812    0.20    0.00    0.41    0.41    0.22    0.00    0.27    0.25
   24   0.794   0.833   0.786   0.806    0.24    0.00    0.42    0.44    0.34    0.00    0.25    0.32
=====  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======

The residual of the window outlines against the flight-median edges is 2.5, 2.5, 2.4, 2.0 px on channels 0 to 3, and of the window widths 2.7, 3.0, 2.8, 3.0 px: the modelled windows are about two pixels too large on every
side, which the primary cannot fix, see below.

What remains open
-----------------

The primary scan did not discriminate on the flight data: the width
error was the same to a hundredth of a pixel at every displacement,
because two channels' widths sit beyond the residual's clip and the
re-polish of the sensor terms at each trial was too weak to restore the
scale of the sky.  The committed tables therefore hold the primary at
nominal, and the stage stays in the package with ``primary=False`` as
the driver's ``ESIS_PRIMARY=0``; a wider clip and a stronger inner
polish are the fix to try.  The modelled windows are two to three
pixels too large on every side against the measured edges, on every
channel, which is the same mismatch seen from the other direction and
is not yet explained: the field stop's clear width, the penumbra of an
f/90 beam, or the sensor's placement are the candidates.

The fit model traces ideal materials, and its photometry is about ten
times brighter than the V&R radiance and the effective area predict, so
the degradations are not yet a throughput; that is a units or integration
factor to find in the scene synthesis or the image operator before they
are read as physics.  The model has no knob for a scale along :math:`y`
alone, and the anisotropy of channels 0 and 3 is expressed through the
sensor's yaw and distance; the physical candidate is the per-channel
grating radius, which was measured and differs.  The reference is fit to
one frame and scored on the others; a polish over several frames at once
is the natural next step now that every stage takes the frame as a
parameter.
