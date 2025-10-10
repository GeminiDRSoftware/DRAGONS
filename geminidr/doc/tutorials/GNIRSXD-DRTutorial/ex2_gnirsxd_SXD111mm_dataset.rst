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
| Science             || N20191013S0006-09  (1.55 |um|)              |
|                     || N20191013S0010-13  (1.68 |um|)              |
|                     || N20191013S0014-17  (1.81 |um|)              |
+---------------------+----------------------------------------------+
| Science flats       || N20191013S0036-51 (1.55 |um|)               |
|                     || N20191013S0054-69 (1.68 |um|)               |
|                     || N20191013S0072-87 (1.81 |um|)               |
+---------------------+----------------------------------------------+
| Pinholes            || None available                              |
+---------------------+----------------------------------------------+
| Science arcs        || N20191013S0034-35 (1.55 |um|)               |
|                     || N20191013S0052-53 (1.68 |um|)               |
|                     || N20191013S0070-71 (1.81 |um|)               |
+---------------------+----------------------------------------------+
| Telluric            || N20191013S0022-25 (1.55 |um|)               |
|                     || N20191013S0026-29 (1.68 |um|)               |
|                     || N20191013S0030-33 (1.81 |um|)               |
+---------------------+----------------------------------------------+
| BPM                 || bpm_20121101_gnirs_gnirsn_11_full_1amp.fits |
+---------------------+----------------------------------------------+

