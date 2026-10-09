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
of the flight, they move by up to a pixel and a half end to end while the
pointing sweeps seven arcseconds, or nine pixels, and the motion of all
four channels' windows is one rigid translation of the field stop of a few
microns, smooth through the flight, which is carried per frame.  And the
channels compared with one another on the sky resolve a tenth of a pixel
where the proxy scene resolves one.

One more motion is visible only in the channels against one another.
Each channel views the solar image through its own sector of the primary,
so if that image is not exactly at the field stop, each channel sees it
shifted along its own dispersion by the sector's offset times the
defocus, while the stop's edge, an aperture in the plane the grating
images, stays put: 1.6 pixels per 0.1 mm for ESIS-I.  A slow change of the
primary's focus through the flight therefore separates the channels'
skies without moving their windows, in a pattern no pointing or grating
motion can make, and the model carries it as one shared term per frame,
:attr:`esis.optics.DistortionParameters.z_primary`.

What the data do not determine is stated rather than fit.  The angle
between the windows and the sky, which a roll of the instrument or of the
field stop would set, changes the merit by less than three thousandths over
four degrees and is held at the design.  The instrument roll is one number
for the whole instrument and is held at zero, in the reference and in
every frame: the merit cannot measure a roll to an arcminute, and a polish
left free in it settles on its first step rather than on a value.  A
per-channel value would be the azimuth of that channel's arm, which turns
the window and the sky together and is the sensor's roll by another name.
The primary's displacement is the one shared quantity the windows do
measure, see below.

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
:class:`~esis.optics.LinearMerit` (``merit="least_squares"``), which compares absolute intensities
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
    He I 584 and O V 630, reading only the window each line illuminates
    (:func:`esis.optics.window_mask`; outside it the frame holds the
    neighbouring lines, which through this line's distortion land on the
    sky as structure that belongs elsewhere), measures the tile shifts
    against channel 1, and solves for the increments of the same terms
    whose change of the mapping reproduces the shifts.  Using two lines
    separates a geometric error from a dispersion error.

7.  **Focus.**  :func:`esis.flights.f1.optics.fit_defocus_history`
    measures every channel's sky against channel 1 in every frame, with
    the windows placed by the smoothed drift.  Each channel views the Sun
    through its own sector of the primary, so a defocus moves its image
    along its own dispersion while the windows stay put, and a focus that
    differs from sector to sector moves each image by its own amount.  The
    focus of every sector is solved from the shifts through the model's
    pattern, four unknowns from six measured components, and a quadratic
    in the frame index, zero at the reference frame, smooths each sector's
    history.  The mean over the sectors is the defocus of the primary as a
    whole.  The proxy scene takes no part: it cannot tell a focus from the
    internal alignment, but the channels can, to a few microns.

8.  **Pointing.**  :func:`esis.flights.f1.optics.fit_distortion_pointing`
    polishes the pitch and yaw of every frame on the mean merit of the four
    channels, with the roll held, each channel's grating offset by that
    frame's window drift and each sector of the primary focused by that
    frame's value first.  The drift of a single frame is measured to a
    few tenths of a pixel and the darkest frames have no edges at all, so
    a cubic in the frame index smooths it through the flight and holds its
    end value where nothing was measured; the offsets and the defocus ride
    along in the pointing table for
    :func:`esis.flights.f1.optics.distortion_fit` to apply.

9.  **Acceptance.**  :func:`esis.flights.f1.optics.acceptance` scores the
    result on frames across the flight, the reference frame among them:
    the merit with each frame's pointing applied on six of them, and in
    every frame the coalignment metric: the length of the median tile
    shift of each channel's sky against channel 1 at each aligned line,
    its component along the channel's dispersion, which is the part a
    velocity is charged for and is given in km/s as well as pixels, the
    scatter of the tiles about the median, and the decomposition of the
    tiles into the linear distortion modes; with the residual of the
    window outlines and widths against the measured edges.  No inversion
    is involved.

Reproducing the committed fit
-----------------------------

The scripts under ``docs/reports/distortion_fit`` run the stages on a
Slurm cluster, in this order::

    sbatch --job-name=dist-channel --array=0-3 reproduce.sbatch channel <dir>
    sbatch --job-name=dist-edges reproduce.sbatch edges <dir>
    sbatch --job-name=dist-combine reproduce.sbatch combine <dir>
    sbatch --job-name=dist-defocus reproduce.sbatch defocus <dir>
    sbatch --job-name=dist-pointing --array=0-29 reproduce.sbatch pointing <dir>
    sbatch --job-name=dist-gather reproduce.sbatch gather <dir>
    sbatch --job-name=dist-accept reproduce.sbatch accept <dir>
    sbatch --job-name=dist-blink reproduce.sbatch blink <dir>
    sbatch --job-name=dist-coalign reproduce.sbatch coalign <dir>

The first produces one ECSV per channel; ``edges`` the median edges and
the drift; ``combine`` runs the outline, primary, shared and internal
stages into the committed ``distortion_reference.ecsv``; ``pointing`` and
``gather`` the committed ``distortion_pointing.ecsv``; ``accept`` the
acceptance table; ``blink`` renders every frame beside its model for
``blink.py page`` to bind into a page that blinks between them; and
``coalign`` reads every channel's frame onto the alignment's sky grid
through its fitted distortion at He I and O V, inside the window each
line illuminates, for ``coalign.py page`` to bind into a page that blinks
any two channels and shows their difference, with the median tile shift
of every pair.  The blink page judges the model against the data; the
coalignment page judges the channels against one another, which is what
the internal stage sets, and it is the view in which a residual of a tenth
of a pixel is visible.  Two more commands repeat late stages on a saved
reference without the capture: ``polish`` reruns the shared polish and the
alignment from ``start_reference.ecsv`` in the directory, for instance at
a finer scene (``ESIS_NUM_SCENE``), and ``align`` reruns only the
alignment in place.  The
environment is the released libraries, ``optika`` 3.1 and later,
``named_arrays`` 2.14 and ``regridding`` 3.5, with ``torch`` and a
``numba`` that can see the CUDA driver for the GPU.  The committed tables
were made on the cluster's GPUs; the same chain on the host
(``ESIS_DEVICE=cpu``) reproduces them, the capture merits to seven
figures and the pointing to 0.03 arcsec, in about the same time.  Every
table's header records the whole environment it was made in:
the interpreter, the platform, the version of every installed package and
the commit of each checkout the fit imported from, so that a table names
exactly the code that produced it.  A further command, ``coregister``,
registers the channels empirically with an offset of each channel's own
pointing; it was the measurement that led to the focus of each sector, and
it stays in the package as a check, not as a stage of the chain.  Loading the Level-1 frames
peaks above 100 GB of memory, which the job script asks for.  End to end
the chain is six hours on the cluster: one to two hours for the capture
of each channel, ten minutes for the edges, two and a half hours for the
combine, three quarters of an hour for the focus, ten to fifteen minutes
for each frame's pointing, and half an hour each for the acceptance and
the blink renders.  On the host without a GPU the capture takes about the
same, and the stages from the focus on fit a workstation with 128 GB of
memory.

Results
-------

The correlation of each channel with its frame after each stage of the
committed chain, run on the cluster on 2026-10-09 (the as-built row is the
starting model, the absolute row the capture of the four channel jobs):

=============  ======  ======  ======  ======
stage             ch0     ch1     ch2     ch3
=============  ======  ======  ======  ======
as-built        0.350   0.419   0.404   0.379
absolute        0.803   0.841   0.803   0.827
outline         0.803   0.840   0.803   0.826
edges [px]       2.53    2.17    2.08    2.12
shared          0.809   0.851   0.804   0.828
aligned [px]     0.01    0.00    0.03    0.03
=============  ======  ======  ======  ======

The merit is the correlation.  The primary displacement is held at nominal, the field-stop roll and the instrument roll at zero,
and the pointing is shared: pitch -21.3, yaw -17.0 arcsec
from the as-built model; the aligned row is the length of the median tile shift of each channel
against channel 1 after the internal alignment.  The free set of every stage, the seed and
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
    0     nan     nan     nan     nan    0.21    0.00    0.21    0.17    0.14    0.00    0.09    0.24
    1     nan     nan     nan     nan    0.16    0.00    0.12    0.11    0.12    0.00    0.06    0.18
    2     nan     nan     nan     nan    0.20    0.00    0.16    0.08    0.10    0.00    0.07    0.15
    3     nan     nan     nan     nan    0.15    0.00    0.16    0.05    0.07    0.00    0.07    0.17
    4     nan     nan     nan     nan    0.14    0.00    0.13    0.10    0.07    0.00    0.09    0.16
    5     nan     nan     nan     nan    0.09    0.00    0.13    0.04    0.04    0.00    0.07    0.18
    6     nan     nan     nan     nan    0.10    0.00    0.11    0.06    0.05    0.00    0.09    0.16
    7     nan     nan     nan     nan    0.15    0.00    0.13    0.08    0.03    0.00    0.09    0.11
    8     nan     nan     nan     nan    0.14    0.00    0.07    0.12    0.05    0.00    0.08    0.10
    9   0.799   0.852   0.805   0.828    0.09    0.00    0.06    0.06    0.02    0.00    0.06    0.06
   10     nan     nan     nan     nan    0.08    0.00    0.05    0.07    0.05    0.00    0.06    0.04
   11     nan     nan     nan     nan    0.05    0.00    0.13    0.06    0.05    0.00    0.04    0.09
   12   0.805   0.846   0.799   0.823    0.09    0.00    0.06    0.06    0.02    0.00    0.06    0.13
   13     nan     nan     nan     nan    0.09    0.00    0.05    0.11    0.02    0.00    0.05    0.07
   14     nan     nan     nan     nan    0.10    0.00    0.07    0.09    0.03    0.00    0.04    0.07
   15   0.810   0.851   0.804   0.827    0.10    0.00    0.08    0.11    0.04    0.00    0.03    0.08
   16     nan     nan     nan     nan    0.10    0.00    0.08    0.06    0.02    0.00    0.02    0.05
   17     nan     nan     nan     nan    0.05    0.00    0.13    0.05    0.02    0.00    0.03    0.06
   18   0.807   0.849   0.801   0.825    0.08    0.00    0.09    0.04    0.03    0.00    0.03    0.09
   19     nan     nan     nan     nan    0.06    0.00    0.10    0.06    0.02    0.00    0.02    0.08
   20     nan     nan     nan     nan    0.04    0.00    0.10    0.08    0.02    0.00    0.02    0.13
   21   0.808   0.848   0.800   0.823    0.07    0.00    0.09    0.10    0.02    0.00    0.03    0.13
   22     nan     nan     nan     nan    0.11    0.00    0.15    0.10    0.03    0.00    0.03    0.11
   23     nan     nan     nan     nan    0.09    0.00    0.12    0.06    0.03    0.00    0.02    0.07
   24   0.804   0.844   0.798   0.816    0.17    0.00    0.12    0.16    0.04    0.00    0.03    0.08
   25     nan     nan     nan     nan    0.14    0.00    0.07    0.09    0.08    0.00    0.04    0.11
   26     nan     nan     nan     nan    0.14    0.00    0.08    0.07    0.11    0.00    0.06    0.08
   27     nan     nan     nan     nan    0.12    0.00    0.11    0.11    0.10    0.00    0.09    0.07
   28     nan     nan     nan     nan    0.14    0.00    0.18    0.13    0.15    0.00    0.16    0.12
   29     nan     nan     nan     nan    0.79    0.00    0.01    0.23    0.18    0.00    0.18    0.08
=====  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======

The coalignment metric over 30 frames of the flight, every channel but channel 1: the rms and the
largest of the whole-channel shifts against channel 1, and the mean scatter of the tiles about them,
in detector pixels (about 17 km/s per pixel along the dispersion):

=====  =======  =======  =======  ==============  ==============  ==============
line   rms      max      scatter  along disp.     rms velocity    max velocity
=====  =======  =======  =======  ==============  ==============  ==============
He I     0.138    0.791    0.348    0.120 px        2.3 km/s       14.9 km/s
O V      0.090    0.239    0.244    0.056 px        1.0 km/s        3.0 km/s
=====  =======  =======  =======  ==============  ==============  ==============

A pixel along the dispersion is 18.9 km/s at He I and 17.4 km/s at O V; only the component of a shift along a channel's dispersion is a velocity error.

The residual of the window outlines against the flight-median edges is 2.4, 2.1, 2.2, 1.9 px on channels 0 to 3, and of the window widths 2.7, 2.8, 2.7, 2.8 px: the modelled windows are about two pixels too large on every
side, which the primary cannot fix, see below.

The focus of each sector of the primary
---------------------------------------

With every frame's pointing, window drift and one defocus of the whole
primary applied, the channels still slid against one another through the
flight: by 0.20 px rms at He I and 0.17 px at O V over the frames bright
enough to measure, and by 0.6 px at the first.  It was a translation of
each channel's whole image, with a tenth of a pixel of magnification and
rotation at most on top; He I and O V agreed on it to a tenth of a pixel,
so it was not a matter of dispersion; and it was smooth in time.  Two
measurements on the Level-1 frames alone, with no model in them but each
channel's Jacobian, bounded what it could be
(``docs/reports/distortion_fit/diagnostics.py``), and a third test named
it.

*The image against the window.*  Against frame 15, on each detector: the
motion of the image, from correlating the interior of the He I and O V
windows with the same pixels at frame 15, thirty pixels in from the edges
so that the edge takes no part; and the motion of the window, from the edge
table.  Taken to the sky and differenced against channel 1, the pointing
drops out, as does any rigid motion of the field stop, and a grating or a
camera that moved would carry its image and its window together.  The
image did not ride with the window: channel 2's image stayed on
channel 1's to 0.02 px while its window moved 0.4 px, and channel 3's
image and window moved apart.  So the motion was not a grating or a
camera, and it was not a decenter of the field stop or of the primary,
which move every window, or every image, by one vector.

*The sectors.*  A focus that differs from sector to sector moves each
channel's image along its own dispersion by its own amount, which is a
three-parameter pattern per frame against six measured components.  It
fits: over the frames with edges the rms of the measured motion falls from
0.13 px to 0.04 px, the floor of the measurement, and what is left across
each channel's dispersion is at the noise.  The figure modes of the primary
tried first, astigmatism, coma and trefoil, had failed because they force
the four sectors into a fixed relation; the sectors want their own
histories, channel 0's sector holding within 8 µm of its focus at the
reference frame while channel 1's moves from −26 to +22 µm, channel 2's
from +2 to +24 and channel 3's from −17 to +13, and each is smooth in time
to the precision of a single frame, 2 to 4 µm.  Nothing after the field stop can do this, and
neither can any rigid motion of the mirror or the stop, so if it is real
it is the figure of the primary changing unevenly with temperature, about
100 nm of sag between sectors over the flight, the same scale as the common
drift of the focus.  The focus stage now solves one focus per sector, and
the pointing table carries four values per frame.

*The edges as fiducials, and the caveat.*  The conclusion rests on the
window edges marking a rigid outline, and measured side by side and line
by line in every frame they do not do so cleanly:

- The He I window has no left edge on any detector, where the image runs
  off the sensor; channel 0 has no bottom edge and channel 3 no top edge.
  Channel 1, the anchor of the alignment, is placed in :math:`x` by the
  right edge of its He I window and the left edge of its O V window alone,
  both soft, both following a curve that departs from a straight line
  by 0.4 px where the other edges hold to 0.1 or 0.2, and 0.4 px per
  fifteen frames apart in their motion.
- The edges differ in sharpness from 0.7 to over 4 px in the width of
  the fitted step, with a median of 1.6, and their profiles are skewed,
  the dark half of the step wider than the lit by 0.8 px in the median.
  An edge three pixels soft is not the field stop in focus.
- Several edges change through the flight: the top edge of channel 0
  sharpens from 3.8 to 1.9 px at O V, the left edge of channel 2 from 4.3
  to 2.5 px, and the right edge of channel 0 softens from 1.8 to 2.5 px at
  He I, while most hold to a tenth of a pixel.
- The two lines, which share one field stop and one grating, disagree on
  the motion of the same side by up to 0.3 px per fifteen frames.

The sector focus is measured along each channel's dispersion, and the
window's position along the dispersion comes from the two edges
perpendicular to it, which are the edges that behaved worst.  The crossing
of a skewed step moves when its width changes, by a fraction of the
change, and that would appear in this measurement as a focus of the
sector, with the same direction.  The disagreement between the lines puts
that at up to half the effect.  So the data support two readings, the
mirror bending or the along-dispersion fiducials drifting, and do not
decide between them; the first is the simpler physical account and is what
the model carries, with this caveat beside it.  What would decide it: the
drift measured per line rather than merged, with only the sharp edges; a
thermal model of the mirror and its mount against the pattern of the
sectors in time; and for the next flight, temperature sensors on the
primary and an undispersed channel.

What remains open
-----------------

The primary scan did not discriminate on the flight data, and an audit
of the stage shows why it could not as written.  Its logic is sound: a
displacement of the primary scales the sky and not the windows, while a
change of a channel's magnification scales both, so the sky's scale from
the merit and the windows' width from the edges together fix the focal
length and the magnification.  But the inner re-polish at each trial
frees only the sensor's distance and angles, and the sensor's distance
also translates the image by five pixels per millimetre on a sensor yawed
twelve degrees; with no lateral term to undo that, the merit refuses the
move that would restore the sky's scale, the widths never change, and the
width error is flat to a hundredth of a pixel at every displacement,
which is what was observed.  The committed tables therefore hold the
primary at nominal, and the stage stays in the package with
``primary=False`` as the driver's ``ESIS_PRIMARY=0``.  The fix is a
direct two-observable solve, or a re-polish that frees a lateral term and
a residual without the clip.  The modelled windows are two to three
pixels too large on every side against the measured edges, on every
channel, which is a magnification excess of about a third of a percent;
a longer focal length of the primary with every channel's sensor a few
millimetres closer would produce exactly that at a fixed sky scale and is
the first candidate, ahead of the field stop's clear width or the sensor's
placement.

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
