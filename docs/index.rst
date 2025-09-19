Introduction
============
The EUV Snapshot Imaging Spectrograph (ESIS) is a sounding rocket-based tomographic imaging spectrograph that was
launched from White Sands Missile Range on September 30th, 2019 :cite:p:`Parker2022`.

.. figure:: _static/esis-rail.avif

    The ESIS instrument on the rail preparing for launch. Image credit: NSROC
    and Catharine Bunn.

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
    vmax = a.outputs.percentile(99.99).ndarray.value

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
        y=0.98,
        s=a.channel.ndarray,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
    )
    ax.text(
        x=.99,
        y=0.98,
        s=a.inputs.time.ndarray,
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="white",
    );

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

Tutorials
=========
A series of notebooks which demonstrate the functionality of this package.

Flight 1 (2019)
---------------
.. toctree::

    reports/point-spread-function
    reports/throughput
    reports/level-0
    reports/level-1

|

Publications
============
`Mission Paper
<https://iopscience.iop.org/article/10.3847/1538-4357/ac8eaa/meta>`_

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
