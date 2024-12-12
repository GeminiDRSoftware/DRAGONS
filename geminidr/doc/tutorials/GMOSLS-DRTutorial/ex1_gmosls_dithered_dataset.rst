.. ex1_gmosls_dithered_dataset.rst

.. _datadithered:

*********************************
Example 1 - Datasets descriptions
*********************************

Dithered Point Source Longslit
------------------------------
This is a GMOS longslit observation.  The primary target is a DB white
dwarf candidate.  We will use this observation to show how a basic longslit
sequence is reduced with DRAGONS.  The
sequence dithers along the dispersion axis and along the slit.  DRAGONS will
adjust for the difference in central wavelength and spatial positions, and
then stack the aligned spectra automatically.

The data were obtained in 2017 using the B600 grating on GMOS South.  GMOS
was equipped with the Hamamatsu detectors at the time.  The
central wavelengths are 515 nm and 530 nm.  We are using a subset of the
original sequence to keep the data volume low.  The effective sequence is::

   [Science, Flat,  Science, Arc], [Science, Flat, Science, Arc]

with the first group of four at 515 and the second at 530 nm.  The
spectrophotometry standard was obtained about a month before the science
observation.

The calibrations we use for this example are:

* BPM.  The bad pixel masks are now found in the Gemini Science Archive
  instead of being packaged with the software. They are associated like the
  other calibrations.  Here we need the 2x2 binning GMOS-S Hamamatsu
  BPM valid for data taken in 2017.  (The date in the name is the "valid from"
  date.)
* Biases.  The science and the standard observations are often taken with
  different Region-of-Interest (ROI) as the standard uses only the central area.
  Therefore we need two sets of biases, one for the science's "Full Frame" ROI,
  and one for the standard's "Central Spectrum" ROI.
* Spectroscopic flats taken with each of the science and standard observations.
* Arcs, for both the science and the standard observations.
* A spectrophotometric standard.

.. warning::  The Bad Pixel Masks (BPMs) are now found in the archive rather
   than packaged with the software.  You must get the static BPM from the
   archive.  See :ref:`getBPM` in :ref:`tips_and_tricks`.

Here is the breakdown of the files.  All the files are included in the tutorial data
package.  They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------------+---------------------------------------------+
| Science             || S20171022S0087,89 (515 nm)                 |
|                     || S20171022S0095,97 (530 nm)                 |
+---------------------+---------------------------------------------+
| Science biases      || S20171021S0265-269                         |
|                     || S20171023S0032-036                         |
+---------------------+---------------------------------------------+
| Science flats       || S20171022S0088 (515 nm)                    |
|                     || S20171022S0096 (530 nm)                    |
+---------------------+---------------------------------------------+
| Science arcs        || S20171022S0092 (515 nm)                    |
|                     || S20171022S0099 (530 nm)                    |
+---------------------+---------------------------------------------+
| Standard (LTT2415)  || S20170826S0160 (515 nm)                    |
+---------------------+---------------------------------------------+
| Standard biases     || S20170825S0347-351                         |
|                     || S20170826S0224-228                         |
+---------------------+---------------------------------------------+
| Standard flats      || S20170826S0161 (515 nm)                    |
+---------------------+---------------------------------------------+
| Standard arc        || S20170826S0162 (515 nm)                    |
+---------------------+---------------------------------------------+
| BPM                 || bpm_20140601_gmos-s_Ham_22_full_12amp.fits |
+---------------------+---------------------------------------------+
