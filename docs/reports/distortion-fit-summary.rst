Calibrating the ESIS-I optics against the flight: a summary
============================================================

.. note::

   A two-page account of the distortion fit for a publication, with the
   minimum set of figures.  The full description of every stage, the
   committed numbers and the reproduction recipe are in
   :doc:`distortion-fit`.  The numbers below are placeholders until the
   committed chain lands; each is marked ``TBD``.

Why a fit is needed
-------------------

ESIS is a slitless spectrograph: one primary mirror forms an image of the
Sun on an octagonal field stop, and four concave varied-line-space gratings,
each viewing that stop through its own sector of the primary's beam,
disperse the field at azimuths 45° apart onto four cameras.  Every bright
line makes a window on each sensor that is the image of the field stop at
that wavelength, and the spectral information lives in the differences
between the four dispersed images.  Recovering a spectrum at every point of
the field therefore requires the mapping from the sky to each sensor to a
small fraction of a pixel, across the field and through the flight: a
pixel is 0.75″ of sky, or 17 km/s of Doppler shift along the dispersion.
The pre-flight model, built from the design and the measured components,
places the images to a few pixels.  The flight data must supply the rest,
and they carry no calibration source: the only references are the Sun
itself, as seen by SDO/AIA at the same time, and the four channels seen
against one another.

The model and what the data can tell it
---------------------------------------

The instrument is modelled in optika as a sequential system of surfaces
with the real coatings, ruling functions and apertures, and is fit through
a small set of physical parameters per channel: the grating's three angles
and its ruling spacing, the sensor's position and three angles, the
instrument's pointing, and the position of the primary along the axis.  A
forward image is made by linearizing the model, a polynomial mapping from
sky and wavelength to the sensor with the vignetting and effective area
that the ray trace gives, and imaging a proxy scene through it: AIA's
171, 193 and 304 Å images resampled onto the sky as a stand-in for the
He I, O V and Mg X emission.  The merit of a parameter set is the
correlation of that image with the Level-1 frame.

One frame does not determine the parameters.  The singular values of the
mapping's Jacobian fall off a cliff after eight, so eight combinations of
the fifteen terms are measurable from a single image and the rest are
null directions: instrument pointing against grating tilt, sensor position
against grating angle, primary displacement against sensor distance.  Two
physical facts break the degeneracies.  The window edges are the image of
the field stop, so they do not move when the instrument's pointing moves
the Sun inside them, while a grating tilt moves edges and Sun together: the
edges separate the pointing from the grating.  And the four channels see
the same Sun at the same instant, so registering them against one another
on the sky resolves a tenth of a pixel where the proxy scene, at 2″
sampling and a different temperature response, resolves one.

The pipeline
------------

1. **Capture.**  For each channel, a seeded differential evolution over the
   nine identifiable terms against the reference frame, followed by a
   restarted Nelder–Mead polish.  This takes the model from a correlation
   of 0.35 to 0.80 with the frame.

2. **Window edges.**  The outline of every window is measured in every
   frame as error-function crossings, medianed over the flight; the
   per-frame departures from the median are kept as the drift of the
   windows.

3. **Outline and shared stages.**  The windows are placed on the measured
   edges with the grating angles and the sensor roll, the sky is refit
   with the remaining terms, the pointing is set to one value for all four
   channels, and the camera terms of each channel are polished.

4. **Internal alignment.**  Each channel's frame is read onto a common sky
   grid through its mapping at He I 584 Å and O V 630 Å, inside the window
   that line illuminates, and cross-correlated tile by tile against
   channel 1; the parameters are moved so that the mapping reproduces the
   measured shifts.  Two lines separate a geometric error, the same shift
   in both, from a dispersion error.

5. **Through the flight.**  The windows drift by up to 1.5 px, which the
   model reproduces as a rigid translation of the field stop of a few
   microns, and the pointing sweeps 6″ in yaw.  A per-frame fit of the
   common pointing, with the windows placed by the smoothed drift, follows
   the Sun.  The channels' skies also separate through the flight by up to
   0.5 px in a pattern no combination of pointing and grating tilt can
   make, since those move the edges too; an axial defocus of the primary
   does exactly that, because each channel views the defocused image
   through its own sector of the primary and sees it shifted along its own
   dispersion.  The defocus is measured from the channels against one
   another in every frame, TBD µm end to end, and applied.

6. **Acceptance.**  The fit is scored on frames it was not fit to, and the
   coalignment metric is measured in every frame: the whole-channel shift
   of each channel against channel 1, and the scatter of the tiles about
   it, in pixels.

Results
-------

The committed tables come from the chain run from scratch on the cluster
on 2026-10-02 with the final code, from the seed-0 captures, in 6.3 hours
of wall time.  The correlation with the reference frame after the capture
is 0.799, 0.837, 0.794, 0.816 on channels 0 to 3, after the shared polish
0.804, 0.847, 0.793, 0.820, and over all thirty frames of the flight with
each frame's pointing applied a mean of 0.799, 0.842, 0.792, 0.816.  The
internal alignment leaves the channels 0.02, 0.00, 0.02, 0.04 px from
channel 1 at the reference frame.  The defocus of the primary drifts by
-16 to +25 µm over the flight, a line with 1.9 µm of scatter.  A second
run of the same chain from a different seed, and the candidate run that
preceded this one, reproduce every one of these numbers to 0.01 in
correlation and 0.02 px in coalignment.

The coalignment metric over all thirty frames, every channel but the
anchor:

=====  ============  ============  ============  ====================
line   rms shift     max shift     tile scatter  rms, frames 6 to 24
=====  ============  ============  ============  ====================
He I   0.207 px      0.623 px      0.326 px      0.128 px
O V    0.185 px      0.475 px      0.243 px      0.110 px
=====  ============  ============  ============  ====================

Only the part of a misregistration along a channel's dispersion is a
velocity error, and a pixel along the dispersion is 18.9 km/s at He I
and 17.4 km/s at O V.  Projected that way the channels agree, over the
frames from the second to the fourth minute of the flight, to 0.11 px or
2.1 km/s rms at He I and 0.09 px or 1.5 km/s at O V, and the worst single
channel and frame in that span is 5 km/s.  Over the whole flight the rms
is 3.5 and 2.7 km/s and the worst 11.2 km/s, at the dark last frames and
in the first minute, where the windows drift fastest; those frames are
the open item, and the metric is what would exclude them from an
inversion.  For scale, the quiet-Sun velocities ESIS measures are of
order 10 km/s, and the tile-to-tile scatter along the dispersion over the
same mid-flight frames, which contains the real Doppler structure of the
scene as well as noise, is 3.4 and 2.3 km/s per 105″ tile.  In the
candidate run that preceded the committed one, dropping the defocus term
from an otherwise identical chain raised the rms over the flight from
0.226 to 0.285 px at He I and from 0.180 to 0.233 px at O V, so the term
buys about a fifth of the residual at no cost to the fit against AIA,
whose correlations do not depend on it.

What limits the coalignment, and what it taught us
--------------------------------------------------

The floor of the measurement is set by the scene, not by the detector.
A 105″ tile of quiet Sun cross-correlated between two channels locates
their relative shift to about 0.2 px, and the median over the field to
about 0.05 px; the tiles' scatter about the median, once the linear
distortion modes are removed, is uncorrelated from tile to tile, so what
remains is not distortion.  Part of that scatter is not noise at all but
signal: a feature moving at 10 km/s is displaced by half a pixel along
each channel's dispersion, in a different direction in each channel, so
real velocity structure appears to the correlation as misregistration
that differs from tile to tile.  In the median over the field it cancels,
because the net flow of the quiet Sun is small; in the tiles it cannot,
and a coalignment measured this way cannot be pushed far below the
Doppler content of the scene.  Four lessons follow for an instrument of
this kind.

*A zero-order image would remove the two degeneracies the data could not
break.*  Every ESIS channel is dispersed, so a uniform Doppler shift of
the whole field, a slow defocus of the primary, and a lateral shift of
the sensor along the dispersion all move a channel's image along its own
dispersion with the edges fixed, and the four channels can only tell one
another how the shifts differ.  An undispersed channel sharing the field
stop would anchor the sky absolutely and to a few hundredths of a pixel,
where the AIA proxy scene anchors it to half a pixel, and would separate
a common Doppler shift from a focus change outright.

*The field stop's edge was the calibration source.*  The octagon's image
is what separated the pointing from the grating, measured the drift of
the optics through the flight to a tenth of a pixel, and placed the
windows; an edge with deliberate fiducial features, a notch or a step in
each side, would fix the rotation and the scale of every window as well
as its position, and would settle the field-stop roll that the images
cannot.

*Microns matter.*  The primary's focus drifted 40 µm over the flight and
the field stop moved 7 µm, and each moved the images by a few tenths of a
pixel: 1.6 px per 0.1 mm of focus, and a pixel per 4 µm of stop.  Either
an athermal metering structure between the primary and the stop, or a
way to measure both in flight, is worth more to the coalignment than any
improvement of the detectors.

*Dispersion and point spread.*  A registration error of a fixed fraction
of a pixel costs fewer km/s the higher the dispersion, so more dispersion
buys velocity precision until the Doppler smearing of the scene, half a
pixel per 10 km/s here, blurs the structure the correlation relies on;
the ESIS dispersion sits comfortably on the right side of that trade.
The channels' point spread functions differ with their gratings and
focus, but a correlation locates a symmetric blur without bias, and the
asymmetric part of an off-axis grating's image is a fraction of its
width, below the floor above; modelling the point spread would sharpen
the inversion more than the coalignment.

Figures
-------

Two figures carry the account.

**Figure 1, the fit at the reference frame.**  For each channel: the
Level-1 frame, the model image of the AIA proxy scene, and their
difference, at the reference frame.  It shows what is being fit and how
well, and the windows of the neighbouring lines on each sensor.
(``figures.py frame``)

**Figure 2, the flight.**  Four panels against time: the fitted pointing
in pitch and yaw; the drift of each channel's windows; the measured and
applied defocus of the primary; and the coalignment metric of every
channel against channel 1 with the acceptance threshold.  It shows the
three time-dependent motions the model carries and that the channels stay
registered to the threshold through the flight.  (``figures.py flight``)

A third figure is worth considering if space allows: the degeneracy
diagram, the singular-value spectrum of one frame's Jacobian beside a
sketch of which motion moves the edges, the sky, or both.  The interactive
blink and channel-difference pages made by ``blink.py`` and ``coalign.py``
are the companion for the website; they cannot go in print.

Cost and repeatability
----------------------

One evaluation of the merit, a linearization of the channel and a
conservative regrid of the scene onto the sensor, takes 1.1 s at the 401
sample scene on a 2026 laptop, on its GPU or on one CPU core alike, and the
same on a cluster GPU node: the cost is the linearization and the Python
around it, not the regrid.  The stages cost, in evaluations per channel:
the capture about 11 000 (a population of 135 for 80 generations) and its
polish 4 000; the outline stage 1 000; the shared polish 2 000 to 5 000;
each frame's pointing 300 across the four channels.  Measured wall times
on the cluster, one GPU and 16 cores per job:

==================  ==========  ======================================
stage               wall time   parallel over
==================  ==========  ======================================
capture             2.2 h       the four channels
window edges        10 min      —
outline and shared  2 h         the four channels (six workers)
defocus history     25 min      —
pointing            22–34 min   the thirty frames
acceptance          15 min      —
pages               8–25 min    —
==================  ==========  ======================================

With the channels and the frames spread over the cluster the chain is
about six hours end to end.  On one machine that runs four processes at a
time it is about nine hours, and serially about a day; a laptop with a
GPU is no faster than one without.  What stops a laptop today is memory,
not time: loading one frame peaks at 90 GB because the AIA proxy scene is
built as a full-resolution cube before it is resampled, although the data
in use are 3 GB.  Caching the resampled scene per frame would bring the
peak to a few gigabytes and the chain within reach of a 32 GB machine in
a working day.

Repeatability was measured by running the whole chain again from a
capture with a different seed.  Everything the data determine repeats:
the correlations with the reference frame agree to 0.012 after the
capture and to 0.002 after the shared polish, the held-out correlations
to 0.002, the outline residuals to 0.1 px, and the coalignment metric
over the flight to 0.02 px rms.  The parameters do not: the two seeds
differ by up to 1.3′ in grating yaw, 1.5 mm in sensor distance and 1 mm
in sensor position, along the null directions the frame cannot see, and
the mapping they describe differs by up to a pixel at the edge of the
field.  The absolute placement of the sky on the sensor is therefore
repeatable to about half a pixel, which is what the AIA proxy scene can
give, while the registration of the channels against one another is
repeatable to a few hundredths.  The whole chain took eight hours of
wall time on the cluster.
