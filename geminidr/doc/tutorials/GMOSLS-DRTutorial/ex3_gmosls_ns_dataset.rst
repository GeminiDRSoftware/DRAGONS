.. ex3_gmosls_ns_dataset.rst

.. _ns_dataset:

*********************************
Example 3 - Datasets descriptions
*********************************

Point Source Longslit Nod-and-Shuffle
-------------------------------------
This is a GMOS longslit nod-and-shuffle observation.  The target is a quasar.
We will use this observation to show how a basic longslit nod-and-shuffle
sequence is reduced with DRAGONS.  The sequence dithers along the dispersion
axis.  DRAGONS will subtract the sky using the two beams, align and stack the
2-D, extract and flux calibrate the spectrum.

The data uses the R400 grating on GMOS North equipped with the Hamamatsu CCDs.
The central wavelengths are 700 nm and 710 nm.  The sequence is::

   [Flat, 3 x Science, Flat, Arc], [Arc, Flat, 3 x Science, Flat]

with the first group of four at 700nm and the second at 710 nm.  The
spectrophotometry standard was obtained about three weeks before the science
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
* Spectroscopic flats taken with each of the science and standard observations.
* Arcs, for both the science and the standard observations.
* A spectrophotometric standard.

.. warning::  The Bad Pixel Masks (BPMs) are now found in the archive rather
   than packaged with the software.  You must get the static BPM from the
   archive.  See :ref:`getBPM` in :ref:`tips_and_tricks`.

Here is the breakdown of the files.  All the files are included in the tutorial data
package.  They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------------+---------------------------------------------+
| Science             || N20190926S0130-32 (700 nm)                 |
|                     || N20190926S0137-39 (710 nm)                 |
+---------------------+---------------------------------------------+
| Science biases      || N20190926S0230-234                         |
+---------------------+---------------------------------------------+
| Science flats       || N20190926S0129,133 (700 nm)                |
|                     || N20190926S0136,140 (710 nm)                |
+---------------------+---------------------------------------------+
| Science arcs        || N20190926S0134 (700 nm)                    |
|                     || N20190926S0135 (710 nm)                    |
+---------------------+---------------------------------------------+
| Standard (G191B2B)  || N20190902S0046 (700 nm)                    |
+---------------------+---------------------------------------------+
| Standard biases     || N20190902S0089-093                         |
+---------------------+---------------------------------------------+
| Standard flats      || N20190902S0047 (700 nm)                    |
+---------------------+---------------------------------------------+
| Standard arc        || N20190902S0062 (700 nm)                    |
+---------------------+---------------------------------------------+
| BPM                 || bpm_20170306_gmos-n_Ham_12_full_12amp.fits |
+---------------------+---------------------------------------------+

.. note::  The nod-and-shuffle dark current in the Hamamatsu CCDs has been
           found to be low enough to be ignored.  They can be requested if
           desired by the PI.  DRAGONS will use a dark if there is one.

           In contrast, for the EEV CCDs and the ee2vv CCDs, nod-and-shuffle
           darks are required, not optional.
