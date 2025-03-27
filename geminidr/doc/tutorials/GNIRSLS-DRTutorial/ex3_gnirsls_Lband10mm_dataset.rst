.. ex3_gnirsls_Lband10mm_dataset.rst

.. include:: symbols.txt

.. _gnirsls_Lband10mm_dataset:

********************************
Example 3 - Datasets description
********************************

L-band Longslit Point Source (10 l/mm grating)
==============================================
In this example, we will reduce the GNIRS L-band longslit observation
of a Be-star.

This observation uses the 10 l/mm grating, the longred camera, a 0.1 arcsec
slit, and is centered at 3.7 |um|.  The dither pattern is a standard
ABBA.

The calibrations we use for this example are:

* BPM. The bad pixel masks are now found in the Gemini Science Archive
  instead of being packaged with the software. They are associated like the
  other calibrations.  (The date in the name is the "valid from"
  date.)
* Flats taken in the same configuration as the science.  They were obtained
  at night right after the science observations.
* Arcs.  No arcs are obtained for L-band data.  The wavelength calibration
  is done using the sky lines in the science observation.
* A telluric standard observation taken in the same configuration as the
  science and obtained at night just before or just after the science
  observation sequences, and at a similar airmass.

.. warning::  The Bad Pixel Masks (BPMs) are now found in the archive rather
   than packaged with the software.  You must get the static BPM from the
   archive.  See :ref:`getBPM` in :ref:`tips_and_tricks`.

Here is the breakdown of the files.  All the files are included in the tutorial data
package.  They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------------+----------------------------------------------+
| Science             || N20180114S0121-124                          |
+---------------------+----------------------------------------------+
| Science flats       || N20180114S0125-132                          |
+---------------------+----------------------------------------------+
| Telluric            || N20180114S0113-116                          |
+---------------------+----------------------------------------------+
| BPM                 || bpm_20121101_gnirs_gnirsn_11_full_1amp.fits |
+---------------------+----------------------------------------------+

