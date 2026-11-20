CNN Inversion Roadmap
=====================

*As of 2026-11-20.*

A convolutional neural network can invert ESIS images into spectral cubes,
and on synthetic images it already recovers the line profiles of O V 630 Å,
and their shifts and widths,
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
    * - A loss over the observed voxels alone,
        and a loss on the line profiles
      - Experiment scripts
      - Working
      - Rasters with narrow spectral windows can be used;
        the loss on the profiles improves velocity, width and profile by 4 to 7%
    * - Training set of IRIS Si IV rasters
      - Experiment scripts, through ``esis.flights.f1.data.synth.scene_iris``
      - 57 training rasters, each mirrored and rotated into 8 copies;
        not used since the SUMER set
      - Fine spatial structure and many active regions,
        but profiles of another line
    * - Training set of SUMER O V rasters
      - Experiment scripts
      - The survey of the SOHO archive is complete:
        161 passes, cut into 1311 frames
      - Real profiles of the line ESIS observes
    * - The throughput of the model of ESIS-I
      - esis, `#132 <https://github.com/esis-mission/esis/pull/132>`_
        and `#133 <https://github.com/esis-mission/esis/pull/133>`_
      - Merged, not yet released
      - The as-built model matches the mean O V signal of the flight frames to about 1%
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
The results on SUMER since 2026-11-19 use the filters of esis #133.
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

- **A network trained on SUMER alone beats MART on every moment and on the profiles.**
  On 21 held-out frames of three test rasters on an 86″ square field
  (an active region, 1996-08-02; the edge of the polar coronal hole, 1996-09-08;
  quiet Sun at disk centre, 1996-09-16),
  leaving out the frames with fewer than a tenth of their pixels at a signal-to-noise ratio of 4:

  .. list-table::
      :header-rows: 1

      * - Method
        - Velocity :math:`r`
        - Velocity error
        - Width :math:`r`
        - Width error
        - Profile distance
        - Radiance error
        - RMS error of the cube
      * - MART
        - 0.52
        - 5.5 km/s
        - 0.16
        - 9.5 km/s
        - 9.3 km/s
        - 8.4%
        - 0.31
      * - Network, loss on the voxels
        - 0.81
        - 3.1 km/s
        - 0.61
        - 4.2 km/s
        - 4.6 km/s
        - 5.0%
        - 0.13
      * - Network, with the loss on the profiles
        - 0.81
        - 3.0 km/s
        - 0.64
        - 3.9 km/s
        - 4.3 km/s
        - 5.6%
        - 0.13

  The velocity and width errors are root mean squares,
  the radiance error is the median of the absolute relative error,
  and the profile distance is the median over the pixels of how far, in km/s,
  the light of the reconstructed profile has to move to make the true one,
  both normalized to unit area.
  The three validation rasters give the same order.
  The networks were trained on 1273 frames of 155 rasters.
  They recover red shifts to +40 km/s
  and the bipolar explosive event of ±40 km/s in the quiet Sun,
  but not the blue shift of -40 km/s in the active region.
  These frames share SUMER's resolution, detector and processing with the training scenes,
  so they favour the networks.

- **The networks are limited by what four channels measure, not by noise, data or design.**
  On the previous SUMER set:

  - The noise of the SUMER truth itself,
    found by drawing its photon counts again,
    is 1.4 km/s in velocity and 1.7 km/s in width,
    against errors of the network of 3.5 and 4.1 km/s.
  - Images without noise barely help the network
    (velocity error from 3.6 to 3.1 km/s, width unchanged),
    though they help MART a lot.
  - Half of the training rasters leave the validation loss only about 10% higher,
    and a floor plus a term in :math:`1/N` puts unlimited data only about 4% below the full set.
  - Three designs reach the same floor:
    the U-Net with the loss on the voxels,
    the U-Net with a loss on the profiles of equal weight,
    and a network over space which gives a profile at each pixel.

- **Some features are invisible to ESIS-I.**
  Imaging each error through the instrument and comparing it with the noise of the images:

  - In the quiet Sun, no pixel's error of the network's profile changes the images by 3σ,
    even with every other voxel at the truth;
    the median is 0.6σ.
  - The blue shoulders of the polar profiles, 17 to 21% of a pixel's light,
    change the images by only 1.1σ and 1.9σ.
  - In the active regions, 14 to 20% of the pixels have errors visible at 3σ,
    on the strong flows;
    where the network misfits the images,
    MART started from it and stopped at the :math:`\chi^2` of the truth
    lowers the error of the cube by only 3 to 6%, and that of the velocity not at all.
  - Profiles with two components, 420 of them in 66 frames,
    mostly explosive events and polar jets,
    come out with two peaks from the networks at only 0 to 5% of their pixels,
    and the networks never invent them;
    MART splits 54% of them, but also 13% of the profiles with one component.

  So no inverter can recover everything at the signal of ESIS-I,
  and the cube of the network needs an uncertainty at each pixel.

- **A loss on the line profiles helps, with a trade at the extremes of signal.**
  The Wasserstein distance between the true and reconstructed profiles,
  both normalized to unit area,
  weighted 4 times the error of the voxels (about 80% of the loss),
  improves velocity, width and profile distance by 4 to 7%
  on the test and validation rasters, and on each of them.
  In bins of signal-to-noise ratio, the gain is largest for faint pixels,
  whose radiance it overestimates (by 10% at a ratio of 4),
  while the width of the brightest pixels, above 60, is 20 to 30% worse.
  Larger weights are being tried.

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
  Also:

  - ``CDELT1`` is per ångström of the first order in some level-1 files
    and of the second in others.
  - Half the rasters record only ±125 km/s about the line.
    The wings beyond are kept by subtracting the two neighbouring lines where they are weak,
    with a mask of the voxels each raster observed, which the loss respects.
  - The spectral window of 29 passes keeps less than 60 km/s of the red side of the line,
    which loses up to 18% of its light and makes them look blue;
    they are left out.
  - Interleaved sequences of one study can share exposures,
    which are counted once.
  - Some passes off the limb hold only faint sky, and are left out by name.

- **Most SUMER rasters are smaller than the field of the tutorial.**
  Nearly all use the 120″ slit, so none of the active-region rasters
  fits the 128″ × 176″ field.
  An 86″ square field, 112 pixels of the grid, fits all of them and most of the quiet Sun.
  The network is fully convolutional,
  so a network trained on small fields runs on the whole field of ESIS.

- **The SUMER archive is surveyed.**
  Of the 6563 level-1 files of SUMER in ESA's catalogue,
  2426 have a wavelength range which covers O V,
  but the range spans every window of a file,
  so only a downloaded file shows whether it has a window on O V.
  About 1140 were downloaded and screened.
  The set kept is 161 passes, cut into 1311 frames of 86″,
  on the disk, at the limb and in the polar coronal holes.
  Left out are 24 rasters with steps of 3″, which would add 403 frames,
  and 123 files with only level-0 data, which would need a calibration of our own.

- **The model of ESIS-I matches the brightness of the flight.**
  The flight frames were 2 to 3 times brighter in O V than the model.
  With the oxide grown on the coatings between their measurement and the flight
  (esis #132), and 2 nm of oxide on each side of the filters instead of 4 (esis #133),
  the as-built model matches the mean O V signal of the four channels to about 1%.
  Channels 0 and 2 stay about 20% brighter than the model, and 1 and 3 about 17% fainter,
  a difference left out of the model until its cause is known.
  The design model, on which the networks are trained,
  has the new filters but not the coatings of #132,
  and is about 15% fainter than the as-built model in O V.

Milestones
----------

Eight milestones take the network from experiments to an ESIS product.
N1 to N5 run on the design model and can start now;
N6 waits for M6 and M8 of the MART roadmap of ctis.

.. list-table::
    :header-rows: 1
    :widths: 5 32 11 9 24 19

    * -
      - Milestone
      - Where
      - Needs
      - Gate
      - Status
    * - N1
      - ``NeuralNetworkInverter`` as it stands, with the two local commits
      - ctis
      -
      - Tests and review pass
      - Open; two commits local
    * - N2
      - Training in the library:
        backprojection on the GPU through the operators,
        new noise in every epoch from the noise model of the instrument,
        validation frames given by the caller,
        random crops, a mask of the voxels in the loss,
        the loss on the line profiles and the weight of each pixel in it,
        checkpoints
      - ctis
      - N1
      - The library alone reproduces the results of the experiment scripts
      - Not started
    * - N3
      - A reproducible SUMER training set:
        a table of the rasters, passes and bad raster steps used,
        and a builder of scenes from the level-1 files,
        with the second-order radiometry,
        the trend and the stripes removed,
        the instrumental width removed,
        the wings kept by subtracting the neighbouring lines,
        a mask of the voxels each raster observed,
        and passes with a cut red wing, repeated exposures and empty sky left out
      - esis
      -
      - Moment maps of every raster free of trends and stripes;
        widths flat along the slit to about 1 km/s
      - Survey done;
        the cleaning works in the experiment scripts
    * - N4
      - The model, on the design model:
        SUMER alone, the masked loss,
        the weight of the loss on the profiles at each level of signal,
        IRIS for pretraining,
        an uncertainty at each pixel,
        from an ensemble or from a second output of the network,
        and the loss of blue shifts
      - Paper repository
      - N2, N3
      - Better than a tuned MART on the held-out SUMER rasters,
        on every moment and on the profile distance,
        at every level of signal
      - SUMER alone, the masked loss and the loss on the profiles chosen;
        tuned MART not yet compared;
        no uncertainty yet
    * - N5
      - Robustness:
        the network trained on the design model run on images of the as-built model,
        errors of the model put in on purpose
        (point-spread function, distortion, flat field, exposure),
        and the lines which contaminate the passband
      - Paper repository
      - N4
      - Degrades no faster than MART;
        the effect of the contaminants on the moments of O V measured
      - Not started
    * - N6
      - The production network:
        trained on the tied flight model at the production grid,
        0.75″ and 24 velocity bins,
        and run on the whole field in overlapping tiles
      - ctis, esis
      - N5, M6, M8
      - As good as N4 on synthetic frames of the flight model;
        fits in the memory of the workstation
      - Waits for M6 and M8
    * - N7
      - The release:
        the weights in a release of esis-data-2019,
        a function of esis which fetches them with ``_retrieve``
        and loads them with ``NeuralNetworkInverter.load``,
        and ctis with PyTorch as an optional dependency of esis
      - esis, esis-data-2019
      - N6
      - A function of esis returns the cube of the network for a flight frame
      - Waits
    * - N8
      - The flight:
        the network on the frames of ESIS-I,
        its cube imaged again and compared with the frames,
        and a map of where it and MART disagree
      - Paper repository
      - N7
      - The cube agrees with the frames within the noise
      - Waits

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
This waits until the inversion without contaminants is as good as it can be made.

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
  The survey is complete, so the list of files is final.
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
5. Validation against a tuned MART on held-out SUMER rasters, at each level of signal.
6. Robustness to errors of the model, and the contaminants.
7. The ESIS-I flight.
8. What four channels can and cannot measure, and ESIS-II.

Risks and open questions
------------------------

- **The flight model (N6).**
  The network depends on the tied flight model and the streamed operators
  of M6 and M8 of the MART roadmap,
  and is no better than that model is.
  Its throughput now matches the flight frames on average,
  but the channels differ from it by about ±20% for a reason not yet known,
  and the networks are trained on the design model, about 15% fainter in O V.
  N5 tests that mismatch first, and measures how fast the network degrades when the model is wrong.
- **The prior of the training set (N4).**
  A network fills in what four channels cannot measure with what its training scenes looked like.
  On the held-out rasters the networks flatten strong flows
  (the blue shift of -40 km/s in the 1996-08-02 active region is lost),
  pull widths toward the typical (none comes out below about 30 km/s),
  and recover few profiles with two components.
  Part of this no inverter can recover at the signal of ESIS-I;
  the cube of the network should carry an uncertainty at each pixel,
  and be checked against the frames and against MART, not trusted alone.
- **The loss on the profiles (N4).**
  It favours the typical pixels over the faintest and the brightest.
  Each pixel's weight in it saturates with radiance,
  and may need to grow with the signal instead.
- **SUMER only, or SUMER and IRIS (N4).**
  SUMER alone is the choice so far:
  it gives the right profiles of O V, but few active regions, most of them small and faint.
  IRIS gives many active regions and finer structure, but the wrong widths;
  pretraining on IRIS and then training on SUMER is untested.
- **The instrumental width of SUMER (N3).**
  The 1.95 pixels is for detector A;
  the width of detector B, which some rasters used,
  and whether it depends on the slit, are not yet known.
- **Where the SUMER builder lives (N3).**
  In esis, beside ``scene_iris``, so that the production network can be trained again,
  or in the paper repository, as an analysis of its own.
- **The interface of the training (N2).**
  The changes to ``NeuralNetworkInverter.train`` are to be proposed before they are made.
- **The paper (all).**
  Its repository, journal and authors are to be decided.
