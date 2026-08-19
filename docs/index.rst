Introduction
============
The EUV Snapshot Imaging Spectrograph (ESIS) is a NASA
`sounding rocket <https://en.wikipedia.org/wiki/Sounding_rocket>`_ mission
designed to measure the speed of plasma in the
`solar atmosphere <https://en.wikipedia.org/wiki/Sun#Atmosphere>`_.
ESIS was launched from
`White Sands Missile Range <https://en.wikipedia.org/wiki/White_Sands_Missile_Range>`_
on September 30th, 2019, and is planned to launch again in 2027.

.. figure:: _static/esis-rail.avif

    The ESIS instrument on the rail preparing for launch. Image credit: NSROC
    and Catharine Bunn.

ESIS is a
`computed tomography imaging spectrometer (CTIS) <https://en.wikipedia.org/wiki/Computed_tomography_imaging_spectrometer>`_:
four cameras look at the Sun through gratings mounted at different azimuths,
so every camera records a dispersed image of the whole field of view in a
single exposure.
Inverting the four overlapping projections recovers a spatial-spectral cube,
which measures the
`Doppler shift <https://en.wikipedia.org/wiki/Doppler_effect>`_ of the
`spectral lines <https://en.wikipedia.org/wiki/Spectral_line>`_ in the ESIS
passband without ever scanning a slit across the target.

Below is an example of one image captured during the 2019 flight.

.. jupyter-execute::
    :hide-code:

    import matplotlib.pyplot as plt
    import named_arrays as na
    import esis

    a = esis.flights.f1.data.level_1()
    a = a[{a.axis_channel: 2}]

    fig, ax = plt.subplots(
        constrained_layout=True,
        figsize=(8, 4),
        dpi=300,
    )
    ax.set_axis_off()

    vmin = 0
    vmax = a.outputs.percentile(99.9).ndarray.value

    a = a[{a.axis_time: 15}]

    img = na.plt.pcolormesh(
        a.inputs.pixel,
        C=a.outputs.value,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
    )
    ax.text(
        x=0.01,
        y=0.05,
        s=a.channel.ndarray,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
    )
    ax.text(
        x=.99,
        y=0.05,
        s=a.inputs.time.ndarray,
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="white",
    );

|

This library provides a model of the ESIS optical system as well as utilities
to interpret the images in terms of physically-meaningful quantities.

|

Installation
============
ESIS is published on PyPI and can be installed using pip:

.. code-block:: bash

    pip install euv-snapshot-imaging-spectrograph

To work on the package itself, install it from source:

.. code-block:: bash

    git clone https://github.com/esis-mission/esis
    cd esis
    pip install -e .[test]

The Level-0 flight images are too large to ship with the package, so
:func:`esis.flights.f1.data.path_directory` downloads them the first time they
are needed and unpacks them into ``~/.esis/data`` (override with the
``ESIS_DATA_DIR`` environment variable).
Derived products are memoized by ``esis.memory`` under ``~/.esis/cache``.

|

Features
========

- A parametric model of the ESIS optical system
  (:class:`esis.optics.FrontAperture`, :class:`esis.optics.CentralObscuration`,
  :class:`esis.optics.PrimaryMirror`, :class:`esis.optics.FieldStop`,
  :class:`esis.optics.Grating`, :class:`esis.optics.Filter`, and
  :class:`esis.optics.Camera`), assembled into an
  :class:`esis.optics.Instrument` and built on :doc:`optika <optika:index>`.
- Raytrace-based characterization of the instrument: point spread function,
  effective area, throughput, vignetting, and distortion.
- Both the final optical design (:func:`esis.flights.f1.optics.design`) and an
  as-built model of the instrument that actually flew
  (:func:`esis.flights.f1.optics.as_built`), along with the best-fit distortion
  parameters recovered from the flight images
  (:func:`esis.flights.f1.optics.distortion_fit`).
- A data-reduction pipeline organized by processing level, from the raw FITS
  files (:class:`esis.data.Level_0`) to calibrated maps of the photons incident
  on each sensor (:class:`esis.data.Level_1`), with bias and dark subtraction
  and cosmic-ray removal.
- Synthetic solar scenes assembled from SDO/AIA
  (:func:`esis.data.synth.scene_aia`) and IRIS
  (:func:`esis.data.synth.scene_iris`) observations, for validating the forward
  model against a known truth.
- Flight-specific configurations for both the 2019 flight
  (:mod:`esis.flights.f1`) and the planned 2027 flight
  (:mod:`esis.flights.f2`), including the :mod:`esis.nsroc` timelines.
- Every model and dataset is an n-dimensional
  :doc:`named-arrays <named_arrays:index>` object, so wavelength, field, pupil,
  channel, and time are addressed by name, and uncertainties propagate
  automatically.

|

Mission Requirements
====================
The performance required of the optical system for mission success, available
programmatically from :func:`esis.flights.f1.optics.requirements`.

.. list-table::
    :header-rows: 1
    :widths: 40 30

    * - Quantity
      - Requirement
    * - Spatial resolution
      - 1.5 Mm
    * - Spectral resolution
      - 18 km/s
    * - Field of view
      - 10 arcmin
    * - Signal-to-noise ratio
      - 17.3
    * - Cadence
      - 15 s
    * - Observation length
      - 150 s

|

Tutorials
=========
A series of notebooks which demonstrate the functionality of this package.

Flight 1 (2019)
---------------
.. toctree::
    :maxdepth: 2

    reports/point-spread-function
    reports/throughput
    reports/aia-image-simulation
    reports/level-0
    reports/level-1

Flight 2 (Planned 2027)
-----------------------
.. toctree::
    :maxdepth: 2

    reports/f2/design
    reports/f2/grating

|

API Reference
=============
An in-depth explanation of all the functions, classes, etc. that are implemented
as part of this library.

.. autosummary::
    :toctree: _autosummary
    :template: module_custom.rst
    :recursive:

    esis

|

Publications
============
The instrument and the results of the first flight are described in the mission
paper, :cite:t:`Parker2022`
(`publisher <https://iopscience.iop.org/article/10.3847/1538-4357/ac8eaa/meta>`_,
`doi:10.3847/1538-4357/ac8eaa <https://doi.org/10.3847/1538-4357/ac8eaa>`_).
Please cite this paper if you use this package in your research.

The Level-0 flight data is archived separately and should be cited alongside
it:
`doi:10.5281/zenodo.21997280 <https://doi.org/10.5281/zenodo.21997280>`_
(also available as ``esis.flights.f1.data.doi``).

|

Future Work
===========
.. toctree::
    :maxdepth: 1

    roadmap.rst
    schedule.rst

|

References
==========

.. bibliography::

|

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
