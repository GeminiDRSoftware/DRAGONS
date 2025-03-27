.. ex2_gnirsls_Kband32mm_dataset.rst

.. include:: symbols.txt

.. _gnirsls_Jband111mm_dataset:

********************************
Example 2 - Datasets description
********************************

J-band Dithered Point Source Longslit with 111 l/mm grating
===========================================================
In this example, we will reduce the GNIRS J-band longslit observation of
a metal-poor M dwarf.

This observation uses the 111 l/mm grating, the short-blue camera, a 0.3 arcsec
slit, and is set to a central wavelength of 1.22 |um|.   The dither pattern is
two consecutive ABBA sequences.

The calibrations we use for this example are:

* BPM. The bad pixel masks are now found in the Gemini Science Archive
  instead of being packaged with the software. They are associated like the
  other calibrations.  (The date in the name is the "valid from"
  date.)
* Flats taken in the same configuration as the science.  They were obtained
  at night right after the science observations.
* An arc taken in the same configuration as the science and also obtained at
  night at the end of the science observation sequences.
* A telluric standard observation taken in the same configuration as the
  science and obtained at night just before or just after the science
  observation sequences, and at a similar airmass.

.. warning::  The Bad Pixel Masks (BPMs) are now found in the archive rather
   than packaged with the software.  You must get the static BPM from the
   archive.  See :ref:`getBPM` in :ref:`tips_and_tricks`.

Here is the breakdown of the files.  All the files are included in the tutorial data
package.  They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------------+----------------------------------------------+
| Science             || N20180201S0052-59                           |
+---------------------+----------------------------------------------+
| Science flats       || N20180201S0060-64                           |
+---------------------+----------------------------------------------+
| Science arcs        || N20180201S0065                              |
+---------------------+----------------------------------------------+
| Telluric            || N20180201S0071-74                           |
+---------------------+----------------------------------------------+
| BPM                 || bpm_20100716_gnirs_gnirsn_11_full_1amp.fits |
+---------------------+----------------------------------------------+
