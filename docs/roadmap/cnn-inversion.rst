CNN Inversion Roadmap
=====================

*As of 2026-11-18.*

A convolutional neural network can invert ESIS images into spectral cubes,
and on synthetic images it already recovers the Doppler shift of O V 630 Å
better than MART run as in the MART tutorial.
It does so only for the instrument and the grid it was trained on,
so the network which ships has to be trained on the tied flight model of ESIS
at the production grid,
the same model which the
`CPU and CUDA MART roadmap <https://ctis.readthedocs.io/en/latest/roadmap/cuda-mart.html>`_
of ctis builds for the Level-4 product.
That dependency sets the path below:
the library and the training set can be finished now,
on the linearized design model of the
`MART tutorial <https://github.com/esis-mission/esis/pull/67>`_,
and the production network follows milestones M6 and M8 of that roadmap.

Where things stand
------------------

.. list-table::
    :header-rows: 1
    :widths: 28 18 18 36

    * - Piece
      - Where
      - Status
      - What it gives
    * - ``NeuralNetworkInverter``,
        a 3D U-Net in PyTorch which maps the backprojection of the images
        through each channel to the scene
      - ctis, `#48 <https://github.com/sun-data/ctis/pull/48>`_
      - Open; two more commits local
        (backprojection in chunks, gradient clipping and early stopping)
      - Training, inversion, saving and loading,
        with the network normalized frame by frame so that it scales with brightness
    * - New noise in every epoch, drawn on the GPU from images without noise
      - Experiment scripts
      - Working
      - Keeps the network from memorizing the noise of the training frames
    * - Backprojection on the GPU through the sparse matrices of the instrument
      - Experiment scripts
      - Working, agrees with ``backproject`` to single precision
      - Makes new noise in every epoch affordable
    * - Validation on whole held-out rasters, random crops, checkpoints
      - Experiment scripts
      - Working
      - An honest stopping point, and less overfitting
    * - Training set of IRIS Si IV rasters
      - Experiment scripts, through ``esis.flights.f1.data.synth.scene_iris``
      - 57 training rasters, each mirrored and rotated into 8 copies
      - Fine spatial structure and many active regions,
        but profiles of another line
    * - Training set of SUMER O V rasters
      - Experiment scripts
      - First set built; a wider survey of the SOHO archive is under way
      - Real profiles of the line ESIS observes
    * - Fetching a released archive with pooch
      - esis, ``_retrieve``
      - In esis
      - The way the weights and the SUMER data would reach users
    * - A dependency of esis on ctis
      - esis
      - None yet
      - Needed before esis can ship an inversion of its own

The experiments ran from scripts outside any repository.
Milestone N4 moves what they found into the paper repository,
and N2 moves what is general into ctis.

What we learned
---------------

All of these are on the linearized design model of ESIS-I,
with images in electrons, shot and read noise,
and pixels counted only where the true signal-to-noise ratio is at least 4.
"Width" is the interquartile range of the line profile.
MART is that of the tutorial, 25 iterations from its first guess;
a tuned MART, stopped at :math:`\chi^2 = 1` and smoothed, is not yet compared.

- **On IRIS scenes, the network beats MART on every moment.**
  Averaged over five held-out IRIS rasters on the 128″ × 176″ grid of the tutorial:

  .. list-table::
      :header-rows: 1

      * - Method
        - Velocity :math:`r`
        - Width :math:`r`
        - Velocity error
        - Width error
        - RMS error of the cube
      * - MART
        - 0.66
        - 0.39
        - 5.9 km/s
        - 7.7 km/s
        - 0.42
      * - Network, 20 training rasters
        - 0.78
        - 0.62
        - 4.4 km/s
        - 4.6 km/s
        - 0.26
      * - Network, 57 rasters, rotations, crops, 32 filters
        - 0.83
        - 0.72
        - 3.8 km/s
        - 4.2 km/s
        - 0.23

  The errors are medians of the absolute error.
  More training rasters, with rotations, gave most of the gain;
  random crops of the field moved the width error from 4.6 to 4.2 km/s;
  doubling the filters, from 1.4 to 5.6 million weights, helped a little.
  Each was one run with one seed.

- **The original design of ctis #2 trains, but more slowly.**
  Ported to PyTorch on the same data,
  six layers at full resolution with kernels of 11 voxels
  reached a validation loss of 1.26 after 75 of 100 epochs,
  against 1.15 and 1.08 for the two U-Nets.
  Its learning rate had to be 3e-5, since 3e-4 diverged,
  so part of the gap may be the schedule.
  It was stopped before a test evaluation.

- **Networks trained on IRIS make O V lines too wide.**
  On real O V profiles from SUMER
  (`Wilhelm et al. 1995 <https://ui.adsabs.harvard.edu/abs/1995SoPh..162..189W>`__)
  the IRIS-trained network is biased wide,
  with median errors of the width of 8 to 17 km/s against 5 to 8 km/s for MART:
  the true widths of O V in quiet Sun are about 35 km/s,
  where the Si IV profiles stretched to O V by ``scene_iris`` are about 46 to 50 km/s.

- **A network trained on SUMER alone removes the bias, and explains a third of the width.**
  On held-out SUMER frames on an 86″ square field:

  .. list-table::
      :header-rows: 1

      * - Frames
        - Method
        - Velocity :math:`r`
        - Velocity error
        - Width :math:`r`
        - Width error
        - RMS error of the cube
      * - Quiet Sun, 1996-09-16, 9 tiles
        - SUMER-trained
        - 0.83
        - 2.0 km/s
        - 0.50
        - 1.8 km/s
        - 0.14
      * -
        - IRIS-trained
        - 0.76
        - 2.6 km/s
        - 0.31
        - 8.5 km/s
        - 0.22
      * -
        - MART
        - 0.44
        - 3.9 km/s
        - 0.02
        - 6.0 km/s
        - 0.39
      * - Sunspot, 1996-08-02, 1 tile
        - SUMER-trained
        - 0.87
        - 2.1 km/s
        - 0.47
        - 2.7 km/s
        - 0.16
      * -
        - IRIS-trained
        - 0.79
        - 2.9 km/s
        - 0.17
        - 12.5 km/s
        - 0.23
      * -
        - MART
        - 0.54
        - 5.5 km/s
        - 0.06
        - 5.4 km/s
        - 0.32

  The small width error is mostly because the true widths vary little:
  over all 26 held-out SUMER frames they spread by only 3.8 km/s,
  half the range between their 16th and 84th percentiles,
  and a constant answer would err by that much against 3.2 km/s for the network.
  The network explains 34% of the variance of the width,
  and 74% of that of the velocity.
  It recovers red shifts to +40 km/s,
  but loses blue shifts beyond about -25 km/s,
  which may be because few training scenes have them.
  These frames share SUMER's resolution, detector and processing with the training scenes,
  so they favour the SUMER-trained network.

- **SUMER data need more cleaning than IRIS data.**
  O V is in second order, and the level-1 files are calibrated for the first;
  the ratio of the two radiometric arrays in each file corrects it,
  and the quiet-Sun radiances which result, 0.34 to 0.52 W sr\ :sup:`-1` m\ :sup:`-2`,
  agree with
  `Vernazza & Reeves (1978) <https://ui.adsabs.harvard.edu/abs/1978ApJS...37..485V>`__.
  The widths carry stripes of a few km/s fixed to the position along the slit,
  some rasters have a smooth trend of the shift of up to ±10 km/s,
  and the instrumental width,
  1.95 pixels FWHM for detector A
  (`Chae et al. 1998 <https://ui.adsabs.harvard.edu/abs/1998ApJ...505..957C>`__),
  is 20 km/s FWHM for O V.
  Half the rasters record only ±125 km/s about the line.

- **Most SUMER rasters are smaller than the field of the tutorial.**
  Nearly all use the 120″ slit, so none of the active-region rasters
  fits the 128″ × 176″ field.
  An 86″ square field, 112 pixels of the grid, fits all of them and most of the quiet Sun.
  The network is fully convolutional,
  so a network trained on small fields runs on the whole field of ESIS.

- **The SUMER archive holds a few dozen usable O V rasters.**
  Of 197 archives whose catalogue entries could hold the line,
  79 have a window on O V, holding 102 rasters,
  of which 41 observations were kept after screening:
  27 of quiet Sun, coronal holes and network,
  and 14 of active regions, loops and sunspots,
  most of them about 100″ × 120″.
  A survey of 746 more archives, including the limb, is under way.

Milestones
----------

Eight milestones take the network from experiments to an ESIS product.
N1 to N5 run on the design model and can start now;
N6 waits for M6 and M8 of the MART roadmap of ctis.

.. list-table::
    :header-rows: 1
    :widths: 6 38 14 12 30

    * -
      - Milestone
      - Where
      - Needs
      - Gate
    * - N1
      - ``NeuralNetworkInverter`` as it stands, with the two local commits
      - ctis
      -
      - Tests and review pass
    * - N2
      - Training in the library:
        backprojection on the GPU through the operators,
        new noise in every epoch from the noise model of the instrument,
        validation frames given by the caller,
        random crops, a mask of the voxels in the loss, checkpoints
      - ctis
      - N1
      - The library alone reproduces the results of the experiment scripts
    * - N3
      - A reproducible SUMER training set:
        a table of the rasters, passes and bad raster steps used,
        and a builder of scenes from the level-1 files,
        with the second-order radiometry,
        the trend and the stripes removed,
        the instrumental width removed,
        and a mask of the velocities each raster records
      - esis
      -
      - Moment maps of every raster free of trends and stripes;
        widths flat along the slit to about 1 km/s
    * - N4
      - The model, on the design model:
        SUMER alone or SUMER and IRIS,
        the masked loss, the size of the field,
        a small ensemble for an uncertainty,
        and the loss of blue shifts
      - Paper repository
      - N2, N3
      - Better than a tuned MART on held-out SUMER and IRIS rasters on every moment
    * - N5
      - Robustness:
        errors of the model put in on purpose
        (point-spread function, distortion, flat field, exposure),
        and the lines which contaminate the passband
      - Paper repository
      - N4
      - Degrades no faster than MART;
        the effect of the contaminants on the moments of O V measured
    * - N6
      - The production network:
        trained on the tied flight model at the production grid,
        0.75″ and 24 velocity bins,
        and run on the whole field in overlapping tiles
      - ctis, esis
      - N5, M6, M8
      - As good as N4 on synthetic frames of the flight model;
        fits in the memory of the workstation
    * - N7
      - The release:
        the weights in a release of esis-data-2019,
        a function of esis which fetches them with ``_retrieve``
        and loads them with ``NeuralNetworkInverter.load``,
        and ctis with PyTorch as an optional dependency of esis
      - esis, esis-data-2019
      - N6
      - A function of esis returns the cube of the network for a flight frame
    * - N8
      - The flight:
        the network on the frames of ESIS-I,
        its cube imaged again and compared with the frames,
        and a map of where it and MART disagree
      - Paper repository
      - N7
      - The cube agrees with the frames within the noise

The contaminants of N5 cannot be learned from the training scenes alone.
Their dispersed images land, by a rough estimate, tens to hundreds of pixels from the O V image of the same place,
in a different direction in each channel,
further than the network can see,
and outside the field of most training scenes.
The plan is a MART of the whole field with every line,
each contaminant constrained to one velocity bin,
whose contaminants are imaged and subtracted from the frames;
the network then inverts the rest,
and is given the image of the contaminants as an input too,
so that it knows where their noise is.
In training, the contamination of one channel can be faked with a patch of another scene,
scaled by a random ratio and subtracted imperfectly.

Data
----

Following the data policy of ESIS,
only final products are published, in the form the software that made them wrote,
and only in releases of the data repository, never in this package.

- **The weights.**
  The trained network is a final product.
  It goes in a release of esis-data-2019, for example ``cnn-v1.0``,
  and esis fetches it with ``_retrieve``.
- **The SUMER data.**
  SUMER data have been in the public domain since 26 April 2000,
  with SUMER to be acknowledged as their source and the team told of the papers
  (`SUMER data policy <https://www2.mps.mpg.de/projects/soho/sumer/text/datavail.html>`_).
  The level-1 archives used, as SUMER distributed them,
  go in a release of esis-data-2019, for example ``sumer-v1.0``,
  as ``aia-v1.0`` holds the AIA images,
  so that the training set survives the SOHO archive.
  The README credits SUMER on SOHO, a project of international cooperation between ESA and NASA,
  gives the dataset DOI, `10.5270/esa-shl1wiw <https://doi.org/10.5270/esa-shl1wiw>`_,
  and points back to the SOHO archive.
  The release waits for the survey, so that its list of files is final.
- **The training frames.**
  These are in-between products, tied to one model of the instrument and one grid,
  and are not published;
  N3 makes them again from the SUMER release.

The paper
---------

The paper is written beside the milestones,
in a repository of its own built in CI like the other ESIS papers,
with its figures made by scripts in that repository
and the weights and data fetched from the releases.
Everything up to N5 runs on the design model,
so only the flight section waits for N6 and N7,
and the critical path of the paper is that of the product.

1. Introduction: computed tomography imaging spectroscopy, MART, and earlier networks.
2. The instrument and its forward model.
3. The training data: the SUMER survey, its calibration and cleaning, and the augmentation.
4. The network and its training.
5. Validation against a tuned MART on held-out SUMER and IRIS rasters.
6. Robustness to errors of the model, and the contaminants.
7. The ESIS-I flight.
8. Limits, and ESIS-II.

Risks and open questions
------------------------

- **The flight model (N6).**
  The network depends on the tied flight model and the streamed operators
  of M6 and M8 of the MART roadmap,
  and is no better than that model is.
  N5 measures how fast it degrades when the model is wrong.
- **The prior of the training set (N4).**
  A network fills in what four channels cannot measure with what its training scenes looked like.
  Events which are rare in the training set,
  such as explosive events and strong blue shifts,
  are pulled toward the ordinary,
  so the cube of the network should be checked against the frames and against MART,
  not trusted alone.
- **SUMER only, or SUMER and IRIS (N4).**
  SUMER gives the right profiles of O V but few active regions,
  most of them small and faint;
  IRIS gives many active regions and finer structure, but the wrong widths.
  Mixing them, or correcting the widths of ``scene_iris`` with SUMER, are the alternatives.
- **The instrumental width of SUMER (N3).**
  The 1.95 pixels is for detector A;
  17 of the 59 rasters which passed the survey used detector B,
  whose width is not yet known,
  and whether it depends on the slit is not yet known either.
- **Where the SUMER builder lives (N3).**
  In esis, beside ``scene_iris``, so that the production network can be trained again,
  or in the paper repository, as an analysis of its own.
- **The interface of the training (N2).**
  The changes to ``NeuralNetworkInverter.train`` are to be proposed before they are made.
- **The paper (all).**
  Its repository, journal and authors are to be decided.
