Distortion fit
==============

How the distortion of ESIS-I is fit to the flight data, and how to reproduce
the fit committed to the package.  The numbers in the tables below are read
from the committed tables when this page is built.

What the data determine
-----------------------

The terms fit are chosen by what each does to the image on the sensor.  In
each channel the yaw and pitch of the grating translate the spectrum, and
with it every line's window; the ruling spacing sets the dispersion; the
pitch and yaw of the pointing translate the Sun within the windows; the
sensor's distance along the beam sets the plate scale; and its three angles
rotate the image and tilt it in the two keystone senses.  The other
placements repeat one of these motions or leave the image unchanged to first
order, and are held at their as-built values.  The roll of the field stop
moves only the edges of the windows.  The displacement of the primary trades
against the focus-preserving distance of the sensor.  The in-plane position
of the sensor trades against the grating's yaw and pitch, and both trade
against the pointing: the mapping cannot tell a moved sky from a moved
grating.  A global search over a set that contains such a null direction
lands somewhere along it at random, and two seeds disagree by tens of
pixels; over the nine terms, two seeds agree to two hundredths in merit.

Two things break the degeneracies the frame leaves.  The edges of the
windows are the image of the field stop through the optics, and do not move
when the pointing does, so they say where the grating and sensor put the
image and leave the pointing to explain the rest.  Measured in every frame
of the flight, they move by up to a pixel and a half end to end while the
pointing sweeps seven arcseconds, or nine pixels, and the motion of all
four channels' windows is smooth through the flight and carried per frame.
And the channels compared with one another on the sky resolve a tenth of a
pixel where the proxy scene resolves one.

One more motion is visible only in the channels against one another.
Each channel views the solar image through its own sector of the primary,
so if that image is not exactly at the field stop, each channel sees it
shifted along its own dispersion by the sector's offset times the
defocus, while the stop's edge, an aperture in the plane the grating
images, stays put: 1.6 pixels per 0.1 mm for ESIS-I.  A change of the
focus through the flight therefore separates the channels' skies without
moving their windows, in a pattern no pointing or grating motion can make.

What the data do not determine is held rather than fit.  The angle between
the windows and the sky, which a roll of the instrument or of the field
stop would set, changes the merit by less than three thousandths over four
degrees and is held at the design.  The instrument roll is held at zero,
in the reference and in every frame: the merit cannot measure a roll to an
arcminute.  The displacement of the primary is held at nominal.

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
the Pearson correlation with the frame.

The edge pixels are 0.37 % of a window.  Moving only the edges by a pixel
changes the correlation by 0.0003 to 0.0014; moving the whole image by a
pixel changes it by 0.005.  The merit therefore cannot place the windows,
which is why the edges are measured directly.

The stages
----------

Each stage is a function of :mod:`esis.flights.f1.optics`, and the free set
of every stage is recorded in the table it writes.

1.  **Absolute.**  :func:`~esis.flights.f1.optics.fit_distortion_channel`
    fits each channel on its own to the reference frame, 15, with
    :func:`esis.optics.fit_distortion`: a seeded differential evolution
    over the nine terms one frame determines (grating yaw and pitch, ruling
    spacing, pointing pitch and yaw, and the sensor's focus-preserving
    distance, roll, pitch and yaw), eighty generations of fifteen members
    per dimension, then a restarted Nelder-Mead polish.  Every other term
    stays at the as-built model (:func:`esis.flights.f1.optics.as_built`).

2.  **Edges.**  :func:`~esis.flights.f1.optics.measure_window_edges` fits
    each side of each window's predicted outline, in every frame whose
    signal is at least half the flight's median, as one blurred straight
    edge: its position, slope and width are shared along the side, and each
    row or column crossing it has its own background, step and ramp
    inside, so that bright structure next to the edge cannot pull it.  The
    crossings are medianed over the frames per row or column.  The drift of
    the windows is read separately, from a step fit to every row and column
    on its own (:func:`~esis.flights.f1.optics.window_drift`), which follows
    a side's motion more steadily than its straight edge does.  Both are
    referred to the reference frame: the median edges are moved by the
    drift there, as a cubic in the frame index smooths it, and the drift is
    taken from there.  The middle window sits on its neighbours and is left
    out.

3.  **Outline.**  :func:`~esis.flights.f1.optics.fit_distortion_reference`
    places each channel's windows on the edges through the grating's yaw
    and pitch and the roll of the field stop, which move the windows without
    moving the sky within them, then re-polishes the sky terms against the
    merit (:func:`esis.optics.fit_distortion_outline`).

4.  **Shared.**  The primary, the field-stop roll and the payload pitch and
    yaw belong to the instrument.  They are set to their mean over the
    channels, and each channel's grating and camera placement is polished
    again with them fixed.

5.  **Internal.**  :func:`esis.optics.align_channels` samples every
    channel's frame on a common sky grid through its own distortion at
    He I 584 and O V 630, reading only the window each line illuminates
    (:func:`esis.optics.window_mask`), measures the tile shifts against
    channel 1, and solves for the increments of the same terms whose change
    of the mapping reproduces the shifts.  Using two lines separates a
    geometric error from a dispersion error.

6.  **Focus.**  :func:`~esis.flights.f1.optics.fit_defocus_history`
    measures every channel's sky against channel 1 in every frame, with
    the windows placed by the smoothed drift, and solves the focus of every
    sector of the primary from the shifts through the model's pattern, four
    unknowns from six measured components.  A quadratic in the frame index,
    zero at the reference frame, smooths each sector's history.  The proxy
    scene takes no part: it cannot tell a focus from the internal
    alignment, but the channels can, to a few microns.

7.  **Pointing.**  :func:`~esis.flights.f1.optics.fit_distortion_pointing`
    polishes the pitch and yaw of every frame on the mean merit of the four
    channels, with the roll held, each channel's grating offset by the
    drift of its windows and each sector focused by that frame's value.
    The offsets and the focus ride along in the pointing table for
    :func:`~esis.flights.f1.optics.distortion_fit` to apply.

8.  **Field stop.**  The field stop is a regular octagon by design, but its
    image is not.  :func:`~esis.flights.f1.optics.measure_field_stop`
    measures every side of the O V window in every frame, carries the
    points back to the plane of the stop through the fit, and fits one
    shape common to the four channels after each channel's own rotation,
    translation and scale are taken out.  The fit is made with the stop and
    the stop is measured through the fit, so the two are iterated: the
    first pass is fit with the regular octagon, each later one with the
    stop the one before it measured, until a measurement turns the sides of
    the stop by less than 0.05 deg rms
    (:func:`~esis.flights.f1.optics.field_stop_converged`).  The regular
    octagon's sides are off by about half a degree; once the stop has
    settled, a refit turns them again by up to 0.04 deg, the scatter of the
    fit itself.  The offsets of the sides scatter from pass to pass by as
    much as the octagon is off, about a micron, so they are recorded but do
    not end the iteration.

9.  **Acceptance.**  :func:`~esis.flights.f1.optics.acceptance` scores the
    result on frames across the flight: the merit with each frame's
    pointing applied on six of them, and in every frame the coalignment
    metric, the length of the median tile shift of each channel's sky
    against channel 1 at each aligned line, its component along the
    channel's dispersion, which is the part a velocity is charged for, and
    the scatter of the tiles about the median; with the residual of the
    window outlines against the measured edges.  No inversion is involved.

Results
-------

The correlation of each channel with the reference frame after each stage,
the residual of the window outlines against the edges after the outline
stage, and the median tile shift against channel 1 after the internal
alignment:

.. jupyter-execute::
    :hide-code:

    import numpy as np
    import astropy.table
    import esis

    reference = esis.flights.f1.optics.distortion_fit_table("reference")
    scores = reference.meta["scores"]
    rows = [
        ("absolute", scores["absolute"]),
        ("outline", scores["outline"]["merit"]),
        ("edges [px]", scores["outline"]["edges"]),
        ("shared", scores["shared"]),
        ("aligned [px]", scores["alignment"]),
    ]
    table = astropy.table.Table(
        rows=[(name, *np.round(values, 3)) for name, values in rows],
        names=("stage", "ch0", "ch1", "ch2", "ch3"),
    )
    table

The reference scored across the flight with each frame's pointing applied:
the correlation on six frames, and the coalignment metric over every frame
and every channel but channel 1, in detector pixels, with the component
along the dispersion as a velocity:

.. jupyter-execute::
    :hide-code:

    acceptance = esis.flights.f1.optics.distortion_fit_table("acceptance")
    scored = acceptance[np.isfinite(acceptance["correlation"])]
    correlation = astropy.table.Table(
        rows=[
            (t, *np.round(scored["correlation"][scored["frame"] == t], 3))
            for t in sorted(set(scored["frame"]))
        ],
        names=("frame", "ch0", "ch1", "ch2", "ch3"),
    )
    correlation

.. jupyter-execute::
    :hide-code:

    coalignment = acceptance.meta["coalignment"]
    astropy.table.Table(
        rows=[
            (
                line,
                round(c["rms"], 3),
                round(c["max"], 3),
                round(c["rms_along_dispersion"], 3),
                round(c["rms_along_dispersion"] * c["km_per_s_per_pixel"], 1),
            )
            for line, c in coalignment.items()
        ],
        names=("line", "rms [px]", "max [px]", "along dispersion [px]", "[km/s]"),
    )

Reproducing the committed fit
-----------------------------

The fit needs the Level-1 data and the AIA scene, which are downloaded and
cached on first use, and more than 100 GB of memory, which loading the
Level-1 frames peaks at.  On one machine, from
``docs/reports/distortion_fit``::

    python reproduce.py run <dir>

runs the whole iteration and writes the tables to commit into
``<dir>/committed``.  The package holds all of them but the three only the
figures read, ``window_edges``, ``acceptance_tiles`` and
``acceptance_modes``, which are published as a release of
``esis-mission/esis-data-2019`` and downloaded, by hash, when they are
read.  It runs on the CPU by default; ``--device cuda`` runs
the regridding on a GPU, which needs the ``gpu`` extra (``pip install -e
".[gpu]"``, which adds ``torch``) and a CUDA toolkit numba can find, and
``--workers`` spreads the capture over processes.  Every stage writes its
output into the pass's directory and is skipped if the output is there, so
an interrupted run continues where it stopped.  The tables are compared
with the committed ones by::

    python reproduce.py compare <dir>/committed

On the cluster's GPUs the capture of a channel took one to two and a half
hours, the edges ten minutes, the outline, shared and internal stages one
to two hours, the focus twenty minutes, the pointing of a frame ten to
twenty minutes and the acceptance half an hour to an hour; the host's CPUs
took about as long for the capture.  The first pass needs the four
captures, later passes none, and the 30 frames' pointing dominates them.
``slurm/chain.sh`` is an example of running the stages as jobs on a Slurm
cluster, with the channels and the frames side by side; the account,
partition and GPUs are the cluster's, given in the environment it
documents.

The fit is stochastic: the capture draws its population at random, and
every linearization samples the field and the pupil at random positions
inside its cells.  One seed, ``--seed`` (0 by default), sets all of these
draws, and every table records the seed and the whole environment it was
made in, the interpreter, the platform, the version of every installed
package and the commit of each checkout, in its header.  A run repeats the
committed fit to within the scatter between seeds, not exactly: on GPUs a
seed repeats only to about 0.0005 in correlation.  Over seeds 0, 1, 2, 4
and 5 the registration of the channels against one another varies by
0.02 px rms over the first 29 frames, against a registration of about
0.1 px, and by 0.08 px in the last, faintest frame; the parameters differ
more, along the near-null directions of the problem.  A capture can fall in
a false optimum, a correlation of 0.65 where the others reach 0.79 to 0.85;
the chain cannot continue from one, and another seed is the remedy.

Two pages judge the committed fit by eye: the `blink page
<../_static/distortion_fit/blink.html>`_ blinks every frame of the flight
against its model (``blink.py``), and the `coalignment page
<../_static/distortion_fit/coalign.html>`_ blinks any two channels read
onto a common sky and shows their difference (``coalign.py``).  They need
the Level-1 frames to render and are too large for this repository, so they
are published as a release of ``esis-mission/esis-data-2019``, which
``docs/conf.py`` names and downloads, checking each page's hash; when the
tables change, the pages are rendered again and released under a new tag.

Checks
------

The scripts in ``docs/reports/distortion_fit/checks`` verify choices the
fit makes on a finished pass; they are not part of the fit, and nothing
reads what they write but this page and the paper.

``coregistration.py`` measures, frame by frame, what the focus of each
sector leaves of the channels' motion against one another, and fits it as
an offset of each channel's own pointing, twelve numbers through the
flight; it was the measurement that led to the sector focus.  On the
committed fit the offsets are at most 0.11 arcsec, 0.15 px, and take the
rms shift of the channels against channel 1 from 0.053 to 0.033 px.  No
mechanism stands behind them, so the fit does not apply them.

``straight_edges.py`` refits a pass with the drift of the windows read from
the straight edges instead of the steps of every row and column.  On the
committed fit the two sides of one window move apart by 0.14 to 0.39 px rms
with the straight edges and by 0.06 to 0.20 px with the steps, and the
steps leave the channels registered better, at 0.105 against 0.117 px rms
at He I and 0.074 against 0.108 px at O V.

The focus of each sector of the primary
---------------------------------------

With every frame's pointing, window drift and one defocus of the whole
primary applied, the channels still slid against one another through the
flight: by 0.20 px rms at He I and 0.17 px at O V over the frames bright
enough to measure, and by 0.6 px at the first.  It was a translation of
each channel's whole image, with a tenth of a pixel of magnification and
rotation at most on top; He I and O V agreed on it to a tenth of a pixel,
so it was not a matter of dispersion; and it was smooth in time.

*The image against the window.*  Against frame 15, on each detector: the
motion of the image, from correlating the interior of the He I and O V
windows with the same pixels at frame 15, thirty pixels in from the edges;
and the motion of the window, from the edges.  Taken to the sky and
differenced against channel 1, the pointing drops out, as does any rigid
motion of the field stop, and a grating or a camera that moved would carry
its image and its window together.  The image did not ride with the window:
channel 2's image stayed on channel 1's to 0.02 px while its window moved
0.4 px, and channel 3's image and window moved apart.  So the motion was
not a grating or a camera, and it was not a decenter of the field stop or
of the primary, which move every window, or every image, by one vector.

*The sectors.*  A focus that differs from sector to sector moves each
channel's image along its own dispersion by its own amount, which is a
four-parameter pattern per frame against six measured components.  It
fits: over the flight the rms of the measured components falls from
0.20 px to 0.05 px, and in the median frame to 0.03 px, the floor of the
measurement.  The figure modes of the primary tried first, astigmatism,
coma and trefoil, had failed because they force the four sectors into a
fixed relation; the sectors want their own histories, each smooth in time
to the precision of a single frame, about 3 µm.  Nothing after the field
stop can do this, and neither can any rigid motion of the mirror or the
stop, so if it is real it is the figure of the primary changing unevenly
with temperature, about 100 nm of sag between sectors over the flight.

*The edges as fiducials, and the caveat.*  The conclusion rests on the
window edges marking a rigid outline, and measured side by side and line
by line in every frame they do not do so cleanly:

- The He I window has no left edge on any detector, where the image runs
  off the sensor; channel 0 has no bottom edge and channel 3 no top edge.
- The edges differ in sharpness from 0.7 to over 4 px in the width of the
  fitted step, with a median of 1.6, and their profiles are skewed, the
  dark half of the step wider than the lit by 0.8 px in the median.
- Several edges sharpen or soften through the flight by up to 2 px, while
  most hold to a tenth of a pixel.
- The two lines, which share one field stop and one grating, disagree on
  the motion of the same side by up to 0.3 px per fifteen frames.

The sector focus is measured along each channel's dispersion, and the
window's position along the dispersion comes from the two edges
perpendicular to it, which are the edges that behaved worst.  The crossing
of a skewed step moves when its width changes, and that would appear here
as a focus of the sector.  The disagreement between the lines puts that at
up to half the effect.  So the data support two readings, the mirror
bending or the along-dispersion fiducials drifting, and do not decide
between them; the first is the simpler physical account and is what the
model carries.  What would decide it: the drift measured per line with
only the sharp edges; a thermal model of the mirror and its mount; and for
the next flight, temperature sensors on the primary and an undispersed
channel.

What remains open
-----------------

The model has no knob for a scale along :math:`y` alone, and the
anisotropy of channels 0 and 3 is expressed through the sensor's yaw and
distance; the physical candidate is the per-channel grating radius, which
was measured and differs.  The fit traces ideal materials and is not a
throughput.  The reference is fit to one frame and scored on the others; a
polish over several frames at once is the natural next step.
