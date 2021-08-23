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

.. _datadithered:

Dataset for Example 1: Dithered Point Source Longslit
-----------------------------------------------------
This is a GMOS longslit observation.  The primary target is a DB white
dwarf candidate.  We will use this observation to show how a basic longslit
sequence is reduced with DRAGONS.  The
sequence dithers along the dispersion axis and along the slit.  DRAGONS will
adjust for the difference in central wavelength and spatial positions, and
then stack the aligned spectra automatically.

The data uses the B600 grating on GMOS South.  The
central wavelengths are 515 nm and 530 nm.  We are using a subset of the
original sequence to keep the data volume low.  The effective sequence is::

   (Science - Flat - Science - Arc) - (Science - Flat - Science - Arc)

with the first group of four at 515 and the second at 530 nm.  The
spectrophotometry standard was obtained about a month before the science
observation.

The calibrations we for this example are:

* Biases.  The science and the standard observations are often taken with
  different Region-of-Interest (ROI) as the standard uses only the central area.
  Therefore we need two sets of biases, one for the science's "Full Frame" ROI,
  and one for the standard's "Central Spectrum" ROI.
* Spectroscopic flats taken with each of the science and standard observations.
* Arcs, for both the science and the standard observations.
* A spectrophotometric standard.

Here is the files breakdown.  All the files are included in the tutorial data
package.  They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------------+---------------------------------+
| Science             || S20171022S0087,89 (515 nm)     |
|                     || S20171022S0095,97 (530 nm)     |
+---------------------+---------------------------------+
| Science biases      || S20171021S0265-269             |
|                     || S20171023S0032-036             |
+---------------------+---------------------------------+
| Science flats       || S20171022S0088 (515 nm)        |
|                     || S20171022S0096 (530 nm)        |
+---------------------+---------------------------------+
| Science arcs        || S20171022S0092 (515 nm)        |
|                     || S20171022S0099 (530 nm)        |
+---------------------+---------------------------------+
| Standard (LTT2415)  || S20170826S0160 (515 nm)        |
+---------------------+---------------------------------+
| Standard biases     || S20170825S0347-351             |
|                     || S20170826S0224-228             |
+---------------------+---------------------------------+
| Standard flats      || S20170826S0161 (515 nm)        |
+---------------------+---------------------------------+
| Standard arc        || S20170826S0162 (515 nm)        |
+---------------------+---------------------------------+



