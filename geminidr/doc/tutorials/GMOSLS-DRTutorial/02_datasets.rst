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

.. _datamultisrc:

Dataset for Example 1: Multi-source Longslit
--------------------------------------------
This is a GMOS longslit observation.  The primary targets are a M dwarf and
white dwarf pair separated by 13 arc seconds.  We will use this observation
to show how multiplec sources are automatically found and extracted.  The
sequence dithers along the dispersion axis and along the slit.  DRAGONS will
adjust for the difference in central wavelength and spatial positions, and
then stack the aligned spectra automatically.

The data uses the B600 grating on GMOS South, an order-blocking filter.  The
central wavelengths are 650 nm and 660 nm.  The sequence is::

   Flat - Science - Science - Flat - Flat - Science - Science - Flat

with the first group of four at 650nm and the second at 660 nm.  The arcs were
taken in the afternoon following the observations.  The spectrophotometric
standard was obtained the next night.

The calibrations we for this example are:

* Biases.  Since the science data and the spectrophotometric standard were
  taken one day apart, we use the same set of biases for both.
* Spectroscopic flats taken with each of the science and standard observations.
* Arcs, for both the science and the standard observations.
* A spectrophotometric standard.

Here is the files breakdown.  All the files are included in the tutorial data
package.  They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------------+---------------------------------+
| Science             || S20180419S0041-42 (650 nm)     |
|                     || S20180419S0045-46 (660 nm)     |
+---------------------+---------------------------------+
| Science biases      || S20180419S0236-240             |
|                     || S20180420S0222-226             |
+---------------------+---------------------------------+
| Science flats       || S20180419S0040 (650 nm)        |
|                     || S20180419S0043 (650 nm)        |
|                     || S20180419S0044 (660 nm)        |
|                     || S20180419S0047 (660 nm)        |
+---------------------+---------------------------------+
| Science arcs        || S20180420S0019 (650 nm)        |
|                     || S20180420S0020 (660 nm)        |
+---------------------+---------------------------------+
| Standard (EG131)    || S20180420S0200 (650 nm)        |
+---------------------+---------------------------------+
| Standard biases     || S20180419S0236-240             |
|                     || S20180420S0222-226             |
+---------------------+---------------------------------+
| Standard flats      || S20180420S0201 (650nm)         |
+---------------------+---------------------------------+
| Standard arc        || S20180420S0301 (650 nm)        |
+---------------------+---------------------------------+



.. _datagmosnb600:

Dataset for Example 2: Custom reduction for GMOS-N B600
-------------------------------------------------------

This is a GMOS longslit observation.  The primary targets are a point source,
two white dwarfs, and there happens to be other stars in the slit. The sequence
dithers along the dispersion axis and along the slit.  DRAGONS will adjust for
the difference in central wavelength and spatial position and stack those
automatically.

The grating is B600 and the two central wavelengths are 650 and 660 nm.  The
sequence is like for Example 1 (Flat - Sci - Sci - Flat) -
(Flat - Sci - Sci - Flat) with the first group at 650 nm and the other at
660 nm.  The arc were taken in the following afternoon.  The spectrophotometric
standard was obtained a month before the science observation.

The GMOS North B600 grating exhibit features that the standard DRAGONS, and
Gemini IRAF, reduction does not remove properly.  In this example, we are
lucky that the standard stars has a HST Calspec model which offers much higher
wavelength resolution than the models included in IRAF, which DRAGONS also
includes.  This example will show how to customize the reduction to obtain a
sensible spectrum in the end.

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
| HST Calspec         || feige34_stis_006.fits          |
+---------------------+---------------------------------+

The HST Calspec archive is available at

     `<https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/>`_
