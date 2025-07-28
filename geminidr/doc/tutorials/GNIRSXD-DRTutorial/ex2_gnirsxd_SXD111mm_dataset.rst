.. ex2_gnirsxd_SXD111mm_dataset.rst

.. include:: symbols.txt

.. _gnirsxd_SXD111mm_dataset:

********************************
Example 2 - Datasets description
********************************

Dithered Point Source XD with ShortBlue + 111 l/mm grating
==========================================================
In this example, we will reduce the GNIRS crossed-dispersed observation of
the eruption of V3890 Sgr, a recurrent nova.

This observation uses the 111 l/mm grating, the short-blue camera and the 0.3
arcsec slit.  The dither pattern is the standard ABBA, one set for each of the
three central wavelength settings.  The results from the three wavelength
settings will be stitched together at the end.

The calibrations we use for this example are:

* **BPM**. The bad pixel masks are now found in the Gemini Observatory Archive
  instead of being packaged with the software. They are associated like the
  other calibrations.  (The date in the name is the "valid from"
  date.)
* **Flats** taken in the same three configurations as the science.  They were
  obtained between the telluric and the science observations.  For each
  central wavelength setting, the flats are taken with two different lamps to
  ensure good exposure in each order.  The software will combine them
  appropriately.
* **Pinholes** are not available for this program.  During the reduction we
  will skip the pinhole steps.
* **Arcs** taken in the same configurations as the science and obtained at
  night between the telluric and the science observations.
* A **telluric standard** observation taken in the same configurations as the
  science and obtained at night, in this case, just before the science
  observation sequence, and at a similar airmass.

.. warning::  The Bad Pixel Masks (BPMs) are now found in the archive rather
   than packaged with the software.  You must get the static BPM from the
   archive.  See :ref:`getBPM` in :ref:`tips_and_tricks`.

Here is the breakdown of the files.  All the files are included in the tutorial data
package.  They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------------+----------------------------------------------+
| Science             || N20190928S0085-88  (1.55 |um|)              |
|                     || N20190928S0090-93  (1.68 |um|)              |
|                     || N20190928S0094-97  (1.81 |um|)              |
+---------------------+----------------------------------------------+
| Science flats       || N20190928S0117-132 (1.55 |um|)              |
|                     || N20190928S0135-150 (1.68 |um|)              |
|                     || N20190928S0153-168 (1.81 |um|)              |
+---------------------+----------------------------------------------+
| Pinholes            || None available                              |
+---------------------+----------------------------------------------+
| Science arcs        || N20190928S0115-116 (1.55 |um|)              |
|                     || N20190928S0133-134 (1.68 |um|)              |
|                     || N20190928S0151-152 (1.81 |um|)              |
+---------------------+----------------------------------------------+
| Telluric            || N20190928S0103-106 (1.55 |um|)              |
|                     || N20190928S0107-110 (1.68 |um|)              |
|                     || N20190928S0111-114 (1.81 |um|)              |
+---------------------+----------------------------------------------+
| BPM                 || bpm_20121101_gnirs_gnirsn_11_full_1amp.fits |
+---------------------+----------------------------------------------+

