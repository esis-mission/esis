Introduction
============
The EUV Snapshot Imaging Spectrograph (ESIS) is a sounding rocket-based tomographic imaging spectrograph that was
launched from White Sands Missile Range on September 30th, 2019 :cite:p:`Parker2022`.

.. figure:: _static/esis-rail.avif

    The ESIS instrument on the rail preparing for launch. Image credit: NSROC
    and Catharine Bunn.

.. jupyter-execute::
    :hide-code:

    import named_arrays as na
    import esis

    a = esis.flights.f1.data.level_1()
    a = a[{a.axis_channel: 2}]

    fig, ax = na.plt.subplots(
        figsize=(9, 4.15),
        constrained_layout=True,
    )
    na.plt.set_xlabel("detector $x$ (pix)", ax=ax)
    na.plt.set_ylabel("detector $y$ (pix)", ax=ax)
    a.to_jshtml(
        ax=ax,
        vmax=a.outputs.percentile(99.9).ndarray,
        cmap="gray",
    )

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
