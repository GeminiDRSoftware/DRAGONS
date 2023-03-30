.. ex1_gmosim_starfield_dataset.rst

.. _starfield_dataset:

*********************************
Example 1 - Datasets descriptions
*********************************

Star field with dithers
-----------------------

This is a GMOS-N imaging observation of a field of stars with dithers obtained
in the i-band.

The calibratons we use in this example are:

* BPM.  The bad pixel masks are now found in the Gemini Science Archive
  instead of being packaged with the software. They are associated like the
  other calibrations.  Here we need the 2x2 binning GMOS-N Hamamatsu
  BPM valid for data taken in 2017.  (The date in the name is the "valid from"
  date.)
* Biases for the science frames and the twilight flats.  Note that the
  twilight flats were obtained just the day after the science observations.
  We can use the same master bias.  If the twilights had been taken several
  days before or after the science observations, it would be preferable to
  use biases contemporary to the twilights.
* Twilight flats.

.. warning::  The Bad Pixel Masks (BPMs) are now found in the archive rather
   than packaged with the software.  You must get the static BPM from the
   archive.  See :ref:`getBPM` in :ref:`tips_and_tricks`.

+---------------+---------------------+--------------------------------+
| Science       || N20170614S0201-205 || 10 s, i-band                  |
+---------------+---------------------+--------------------------------+
| Bias          || N20170613S0180-184 |                                |
|               || N20170615S0534-538 |                                |
+---------------+---------------------+--------------------------------+
| Twilight Flats|| N20170702S0178-182 || 40 to 16 s, i-band            |
+---------------+---------------------+--------------------------------+
| BPM           || bpm_20170306_gmos-n_Ham_22_full_12amp.fits          |
+---------------+------------------------------------------------------+
