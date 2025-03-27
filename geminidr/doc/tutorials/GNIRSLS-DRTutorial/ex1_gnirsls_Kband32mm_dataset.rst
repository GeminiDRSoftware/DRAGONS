.. ex1_gnirsls_Kband32mm_dataset.rst

.. include:: symbols.txt

.. _gnirsls_Kband32mm_dataset:

********************************
Example 1 - Datasets description
********************************

K-band Dithered Point Source Longslit with 32 l/mm grating
==========================================================
In this example, we will reduce the GNIRS K-band longslit observation of
"SDSSJ162449.00+321702.0", a white dwarf.

This observation uses the 32 l/mm grating, the short-blue camera, a 0.3 arcsec
slit, and is set to a central wavelength of 2.2 |um|.   The dither pattern is
the standard ABBA.

The calibrations we use for this example are:

* BPM. The bad pixel masks are now found in the Gemini Observatory Archive
  instead of being packaged with the software. They are associated like the
  other calibrations.  (The date in the name is the "valid from"
  date.)
* Flats taken in the same configuration as the science.  They were obtained
  at night right after the science observations.
* An arc taken in the same configuration as the science and obtained at
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
| Science             || N20170609S0127-130                          |
+---------------------+----------------------------------------------+
| Science flats       || N20170609S0131-135                          |
+---------------------+----------------------------------------------+
| Science arcs        || N20170609S0136                              |
+---------------------+----------------------------------------------+
| Telluric            || N20170609S0118-121                          |
+---------------------+----------------------------------------------+
| BPM                 || bpm_20121101_gnirs_gnirsn_11_full_1amp.fits |
+---------------------+----------------------------------------------+

