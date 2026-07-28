.. ex4_gmosls_nsred_dataset.rst

.. _nsred_dataset:

*********************************
Example 4 - Datasets descriptions
*********************************

Nod-and-Shuffle Correct for Extra Order
---------------------------------------
This is a GMOS longslit nod-and-shuffle observation.  The target is a high
redshift quasar. We will use this observation to show how to recognize when
the light from the second order shows up and how to correct for it using the
interactive tools.

The particularity with this data are that the setting is quite
red and the second order shows up in the spectrum. The configuration uses the
OG515 blocking filter and the second order light appears at 1030nm.

The data uses the R400 grating on GMOS North equipped with the EEV CCDs.
The central wavelengths are 880 nm, 890 nm, and 900 nm.  The sequence is::

   [Flat, Science], [Science, Flat], [Flat - Science]

with the first group of four at 900nm, the second at 890nm, and the last at
880nm.  The arcs were obtained in the morning.  The
spectrophotometry standard was obtained about 2 months after the science
observation.

The calibrations we use for this example are:

* BPM.  The bad pixel masks are now found in the Gemini Science Archive
  instead of being packaged with the software. They are associated like the
  other calibrations.  Here we need the 1x2 binning GMOS-N Hamamatsu
  BPM valid for data taken in 2017.  (The date in the name is the "valid from"
  date.)
* Biases.  The science and the standard observations are often taken with
  different Region-of-Interest (ROI) as the standard uses only the central area.
  Therefore we need two sets of biases, one for the science's "Full Frame" ROI,
  and one for the standard's "Central Spectrum" ROI.  Here we use only 5 biases
  for each setting to minimize the amount of data needed.  For a science
  reduction, please consider using 10 to 20 biases.
* Darks.  For the EEV CCDs, nod-and-shuffle darks are required to map the
  the charge trapping that happens when the electrons are shuffle around.
* Spectroscopic flats taken with each of the science and standard observations.
* Arcs, for both the science and the standard observations.
* A spectrophotometric standard.

.. warning::  The Bad Pixel Masks (BPMs) are now found in the archive rather
   than packaged with the software.  You must get the static BPM from the
   archive.  See :ref:`getBPM` in :ref:`tips_and_tricks`.

Here is the files breakdown.  All the files are included in the tutorial data
package.  They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------------+--------------------------------------------+
| Science             || N20080830S0261 (900 nm)                   |
|                     || N20080830S0262 (890 nm)                   |
|                     || N20080830S0265 (880 nm)                   |
+---------------------+--------------------------------------------+
| Science biases      || N20080830S0527-531                        |
+---------------------+--------------------------------------------+
| Science flats       || N20080830S0260 (900 nm)                   |
|                     || N20080830S0263 (890 nm)                   |
|                     || N20080830S0264 (880 nm)                   |
+---------------------+--------------------------------------------+
| Science arcs        || N20080830S0491 (900 nm)                   |
|                     || N20080830S0492 (890 nm)                   |
|                     || N20080830S0493 (880 nm)                   |
+---------------------+--------------------------------------------+
| Standard (G191B2B)  || N20190902S0046 (900 nm)                   |
+---------------------+--------------------------------------------+
| Standard biases     || N20081011S0313-317                        |
+---------------------+--------------------------------------------+
| Standard flats      || N20081010S0534 (900 nm)                   |
+---------------------+--------------------------------------------+
| Standard arc        || N20081010S0552 (900 nm)                   |
+---------------------+--------------------------------------------+
| BPM                 || bpm_20010801_gmos-n_EEV_22_full_3amp.fits |
+---------------------+--------------------------------------------+
