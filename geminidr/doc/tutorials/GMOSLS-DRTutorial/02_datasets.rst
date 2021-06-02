.. datasets.rst

.. _datasets:

********************************
Setting up and tutorial datasets
********************************

.. _datasetup:

Downloading the tutorial datasets
=================================

All the data needed to run this tutorial are found in the tutorial's data
package:

    `<http://www.gemini.edu/sciops/data/software/datapkgs/gmosls_tutorial_datapkg-v1.tar>`_

Download it and unpack it somewhere convenient.

.. highlight:: bash

::

    cd <somewhere convenient>
    tar xvf gmosls_tutorial_datapkg-v1.tar
    bunzip2 gmosls_tutorial/playdata/*.bz2

The datasets are found in the subdirectory ``gmosls_tutorial/playdata``, and
we will work in the subdirectory named ``gmosls_tutorial/playground``.

.. note:: All the raw data can also be downloaded from the Gemini Observatory
     Archive.  Using the tutorial data package is probably more convenient.

Datasets descriptions
=====================

.. _datamultisource:

Dataset for Example 1: Multi-source Longslit
--------------------------------------------

This is a GMOS longslit observation.  The primary target is a point source,
a white dwarf, and there happens to be another star in the slit.  We will use
this observation to show how multiple sources are automatically found and
extracted.   The sequence dithers along the dispersion axis and along the slit.
DRAGONS will adjust for the difference in central wavelength and spatial
position and stack those automatically.

The grating is B600 and the two central wavelengths are 650 and 660 nm.  The
sequence is a (Flat - Sci - Sci - Flat) - (Flat - Sci - Sci - Flat) with the
first group at 650 nm and the other at 660 nm.  The arc were taken in the
following morning.  The spectrophotometric standard was obtained a month before
the science observation.

The calibrations we use for this example are:

* Biases, for both the science observation and the spectrophotometric
  standard observation.
* Spectroscopic flats taken with each of the science and standard observations.
* Arcs, for both the science and the standard observations.
* A spectrophotometric standard.

Here is the files breakdown.  All the files are included in the tutorial data
package.  They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------------+---------------------------------+
| Science             || N20180526S1024-1025 (650 nm)   |
|                     || N20180526S1028-1029 (660 nm)   |
+---------------------+---------------------------------+
| Science biases      || N20180525S0292-296             |
|                     || N20180527S0848-852             |
+---------------------+---------------------------------+
| Science flats       || N20180526S1023 (650 nm)        |
|                     || N20180526S1026 (650 nm)        |
|                     || N20180526S1027 (660 nm)        |
|                     || N20180526S1030 (660 nm)        |
+---------------------+---------------------------------+
| Science arcs        || N20180527S0001 (650 nm)        |
|                     || N20180527S0002 (660 nm)        |
+---------------------+---------------------------------+
| Standard (Feige 34) || N20180423S0024 (650 nm)        |
+---------------------+---------------------------------+
| Standard biases     || N20180423S0148-152             |
|                     || N20180422S0144-148             |
+---------------------+---------------------------------+
| Standard flats      || N20180423S0025 (650nm)         |
+---------------------+---------------------------------+
| Standard arc        || N20180423S0110 (650 nm)        |
+---------------------+---------------------------------+

