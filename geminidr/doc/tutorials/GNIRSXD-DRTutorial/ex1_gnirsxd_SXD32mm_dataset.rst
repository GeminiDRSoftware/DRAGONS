.. ex1_gnirsxd_SXD32mm_dataset.rst

.. include:: symbols.txt

.. _gnirsxd_SXD32mm_dataset:

********************************
Example 1 - Datasets description
********************************

Dithered Point Source XD with ShortBlue camera and 32 l/mm grating
==================================================================
In this example, we will reduce the GNIRS crossed-dispersed observation of
a supernova type II 54 days after explosion.

This observation uses the 32 l/mm grating, the short-blue camera and the 0.675
arcsec slit.   The dither pattern is the standard ABBA, repeated 5 times.

The calibrations we use for this example are:

* BPM. The bad pixel masks are now found in the Gemini Observatory Archive
  instead of being packaged with the software. They are associated like the
  other calibrations.  (The date in the name is the "valid from"
  date.)
* Flats taken in the same configuration as the science.  They were obtained
  at night right after the science observations.  The flats are two series
  taken with two different lamps to ensure good exposure in each order.  The
  software will combine them appropriately.
* Pinholes taken in the same configuration as the science.  They were obtained
  the morning after the science observations.
* An arc taken in the same configuration as the science and obtained at
  night at the end of the science observation sequences.
* A telluric standard observation taken in the same configuration as the
  science and obtained at night, in this case, just before the science
  observation sequence, and at a similar airmass.

.. warning::  The Bad Pixel Masks (BPMs) are now found in the archive rather
   than packaged with the software.  You must get the static BPM from the
   archive.  See :ref:`getBPM` in :ref:`tips_and_tricks`.

Here is the breakdown of the files.  All the files are included in the tutorial data
package.  They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------------+----------------------------------------------+
| Science             || N20170113S0146-165                          |
+---------------------+----------------------------------------------+
| Science flats       || N20170113S0168-183                          |
+---------------------+----------------------------------------------+
| Pinholes            || N20170113S0569-573                          |
+---------------------+----------------------------------------------+
| Science arcs        || N20170113S0166-167                          |
+---------------------+----------------------------------------------+
| Telluric            || N20170113S0123-138                          |
+---------------------+----------------------------------------------+
| BPM                 || bpm_20121101_gnirs_gnirsn_11_full_1amp.fits |
+---------------------+----------------------------------------------+

