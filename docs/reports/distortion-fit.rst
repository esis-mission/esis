Distortion fit
==============

How the distortion of ESIS-I is fit to the flight data, and how to reproduce
the fit committed to the package.

The merit
---------

The fit compares an image of the Sun seen by another instrument (the AIA
proxy scene, :func:`esis.flights.f1.data.synth.scene_aia`) with a Level-1
frame, after imaging the scene through the model of one channel.  The
image is made by :class:`esis.optics.LinearMerit`: the channel is
linearized (:meth:`optika.systems.SequentialSystem.linearize`), the scene
is clipped by the field stop in object space, and it is regridded
conservatively onto the sensor.  The merit is the Pearson correlation with
the frame.

Compared with the sparse ray trace this replaced, the linearized image has
no sampling noise, so the merit is deterministic and smooth, and it needs no
point-spread function: the footprint integral of the regrid is a physically
sized smoothing of its own.  It is three times cheaper per evaluation, and
the regrid runs on a GPU.  Applying the field stop per line, before the
lines are summed, is what keeps each line's spill-over out of its
neighbours' windows; a mask over the summed image does not.

The stages
----------

:func:`esis.flights.f1.optics.fit_distortion_reference` runs three stages
from the as-built model (:func:`esis.flights.f1.optics.as_built`) and the
data alone.

1.  **Absolute.**  Each channel is fit on its own with
    :func:`esis.optics.fit_distortion`: a seeded differential evolution with
    a loose tolerance, which only has to land in the basin, followed by a
    restarted Nelder-Mead polish.  The capture is required: from the
    as-built model a local method alone fails on two channels of four, and
    from starts displaced by a tenth of the bounds it fails five times in
    twelve.  Because the merit is deterministic and the capture is seeded,
    the stage is reproducible.  The sensor placement is held at its as-built
    value here: against the proxy scene it is degenerate with the pointing
    and the grating, and freeing it only slows the capture.  About an hour
    per channel on one GPU.
2.  **Shared.**  The primary displacement, the field-stop roll and the
    payload pitch and yaw belong to the instrument, and a fit which lets
    them differ per channel is using them as stand-ins for the camera
    placement.  They are set to their mean over the channels and each
    channel's own terms are polished again, at a cost of a few thousandths
    in correlation.
3.  **Internal.**  A fit against the proxy scene places a channel to about
    a pixel; the channels compared with one another on the sky plane resolve
    a tenth of a pixel.  :func:`esis.optics.align_channels` samples every
    channel's frame on a common sky grid through its own distortion at
    He I 584 and O V 630, measures the tile shifts against channel 1, and
    solves for the increments of the grating and camera terms whose change
    of the mapping reproduces the shifts.  Using two lines separates a
    geometric error from a dispersion error.  Three passes of forty seconds
    reach the floor of the measurement, about a fifth of a pixel.  The
    pointing, the primary and the field stop are never revised by this
    stage.

The parameters
--------------

:class:`esis.optics.DistortionParameters` carries, per channel, the grating
yaw, pitch and roll, the ruling spacing, the azimuth offset, and the sensor
placement: a focus-preserving distance, roll, pitch, yaw, and in-plane
translation.  The sensor terms are what let one channel's mapping differ
from another's by a scale, an anisotropy and a rotation, which is what the
internal alignment measures and which none of the other parameters can
produce.  The distance is compound: the sensor moves by the parameter and
the grating by :obj:`esis.optics.KAPPA_FOCUS` times less along the beam, so
that the magnification changes without the focus.  The ratio is measured on
the model as 16.9, the longitudinal magnification of a system with a
magnification of four.

Reproducing the committed fit
-----------------------------

The scripts under ``docs/reports/distortion_fit`` run the stages on a
Slurm cluster::

    sbatch --job-name=dist-channel --array=0-3 reproduce.sbatch channel <dir>
    sbatch --job-name=dist-combine reproduce.sbatch combine <dir>
    sbatch --job-name=dist-pointing --array=0-29 reproduce.sbatch pointing <dir>
    sbatch --job-name=dist-gather reproduce.sbatch gather <dir>

The first produces one ECSV per channel, the second the committed
``distortion_reference.ecsv``, the last two the committed
``distortion_pointing.ecsv``.  The environment needs ``named_arrays`` and
``optika`` with device support, ``regridding`` 3.4 with ``torch``, and a
``numba`` that can see the CUDA driver.  Loading the Level-1 frames peaks
above 100 GB of memory, which the job script asks for.

Acceptance
----------

The fit was accepted by inverting frames 14--16 with the MART pipeline at
production scale, with the mapping of each stage in turn.  Every stage
lowers the residual of every channel against the previously committed
reference, and the internal alignment reduces the spurious velocity ramp
across the field of view of every line: at O V from 5.2 to 1.4 km/s, at
O III from 2.7 to 1.5, at O IV from 1.9 to 0.8.
