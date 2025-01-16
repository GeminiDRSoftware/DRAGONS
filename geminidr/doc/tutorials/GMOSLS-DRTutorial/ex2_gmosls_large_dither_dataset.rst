.. ex2_gmosls_large_dither_dataset.rst

.. _data_large_dither:

*******************************
Example 2 - Dataset description
*******************************
The dataset used in this example is from a GMOS longslit observation from the program GS-2022A-FT-110.
The primary target is the central galaxy of an Odd Radio Circle, `ORC J0102-2450 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.505L..11K/abstract>`_.
The observation sequence makes a large wavelength dither (several tens of nanometer) to circumvent the issues with
GMOS South amplifier #5, which began in January 2022. During reduction, DRAGONS will adjust for the difference in
central wavelength and then stack the aligned spectra automatically.

The observation uses the R400 grating on GMOS South. The central wavelengths
for this dataset are 795 nm and 705 nm.
The effective sequence is::

   [Flat, Science], [Science, Flat], [Arc, Arc]

with the first group at 795 nm and the second at 705 nm, the two required
arcs obtained in the morning.

The spectrophotometry standard was observed about three days before the
science observation.


The calibrations we use for this example are:

* BPM.  The bad pixel masks are now found in the Gemini Science Archive
  instead of being packaged with the software. They are associated like the
  other calibrations.  Here we need the 2x2 binning GMOS-S Hamamatsu
  BPM valid for data affected by the amplifier #5 issues. (The date in the name is the "valid from"
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
package. They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------------+---------------------------------------------+
| Science             || S20220611S0717 (705 nm)                    |
|                     || S20220611S0716 (795 nm)                    |
+---------------------+---------------------------------------------+
| Science biases      || S20220610S0182-186                         |
|                     || S20220611S0827,829,830,832,834             |
+---------------------+---------------------------------------------+
| Science flats       || S20220611S0718 (705 nm)                    |
|                     || S20220611S0715 (795 nm)                    |
+---------------------+---------------------------------------------+
| Science arcs        || S20220611S0782 (705 nm)                    |
|                     || S20220611S0779 (795 nm)                    |
+---------------------+---------------------------------------------+
| Standard (LTT7379)  || S20220608S0098 (705 nm)                    |
|                     || S20220608S0101 (795 nm)                    |
+---------------------+---------------------------------------------+
| Standard biases     || S20220608S0186-190                         |
|                     || S20220609S0206-210                         |
+---------------------+---------------------------------------------+
| Standard flats      || S20220608S0099 (705 nm)                    |
|                     || S20220608S0100 (795 nm)                    |
+---------------------+---------------------------------------------+
| Standard arc        || S20220608S0124 (705 nm)                    |
|                     || S20220608S0125 (795 nm)                    |
+---------------------+---------------------------------------------+
| BPM                 || bpm_20220128_gmos-s_Ham_22_full_12amp.fits |
+---------------------+---------------------------------------------+
