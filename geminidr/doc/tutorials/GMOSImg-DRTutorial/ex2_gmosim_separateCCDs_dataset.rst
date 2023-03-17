.. ex2_gmosim_separateCCDs_dataset.rst

.. _separateCCDs_dataset:

*********************************
Example 2 - Datasets descriptions
*********************************

Separate CCDs
-------------

This is a GMOS-N imaging observation of the galaxy Bootes V obtained
in the r-band.

The calibratons we use in this example are:

* BPM.  The bad pixel masks are now found in the Gemini Science Archive
  instead of being packaged with the software. They are associated like the
  other calibrations.  Here we need the 2x2 binning GMOS-N Hamamatsu
  BPM valid for data taken in 2022.  (The date in the name is the "valid from"
  date.)
* Biases for the science frames and the twilight flats.  Note that the
  twilight flats were obtained several before after the science observations.
  We are going to use two set of biases, each contemporary to the science
  and the twilight frame, respectively.
* Twilight flats.

.. important::
    For accurate photometry, observations of photometric standard of various
    colors should be obtained and observed on each of the 3 CCDs.

+---------------+---------------------+--------------------------------+
| Science       || N20220627S0115-119 || 350 s, i-band                 |
+---------------+---------------------+--------------------------------+
| Bias          || N20220613S0180-184 || For science                   |
|               || N20220627S0222-226 || For twilights                 |
+---------------+---------------------+--------------------------------+
| Twilight Flats|| N20220613S0138-142 || r-band                        |
+---------------+---------------------+--------------------------------+
| BPM           || bpm_20220303_gmos-n_Ham_22_full_12amp.fits          |
+---------------+------------------------------------------------------+
