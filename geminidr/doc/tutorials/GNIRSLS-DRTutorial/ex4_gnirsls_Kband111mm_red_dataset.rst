.. ex4_gnirsls_Kband111mm_red_dataset.rst

.. include:: symbols.txt

.. _gnirs_Kband111mm_red_example:

********************************
Example 4 - Datasets description
********************************

Beyond 2.3 microns K-band Longslit Point Source (111 l/mm grating)
==================================================================

In this example, we will reduce a GNIRS K-band longslit observation with a
central wavelength of 2.365 |um| with the 111 l/mm grating and the Long Blue
camera.  The dither pattern is a ABBA-ABBA sequence.  The slit width is
0.1 arcsec.

The target is the hypergiant :math:`{\rho}` Cas.

The calibrations we use for this example are:

* BPM. The bad pixel masks are now found in the Gemini Science Archive
  instead of being packaged with the software. They are associated like the
  other calibrations.  (The date in the name is the "valid from"
  date.)
* Flats taken in the same configuration as the science.  They were obtained
  at night right after the science observations.
* An arc taken in the same configuration as the science and also obtained at
  night at the end of the science observation sequences.  The coarse wavelength
  solution that we get from the arc is used as a starting point for the
  computation of the solution derived from telluric absorption lines.

.. * telluric

.. warning::  The Bad Pixel Masks (BPMs) are now found in the archive rather
   than packaged with the software.  You must get the static BPM from the
   archive.  See :ref:`getBPM` in :ref:`tips_and_tricks`.

Here is the breakdown of the files.  All the files are included in the tutorial data
package.  They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------------+----------------------------------------------+
| Science             || N20201026S0100-107                          |
+---------------------+----------------------------------------------+
| Science flats       || N20201026S0108-113                          |
+---------------------+----------------------------------------------+
| Science arcs        || N20201026S0114                              |
+---------------------+----------------------------------------------+
| BPM                 || bpm_20121101_gnirs_gnirsn_11_full_1amp.fits |
+---------------------+----------------------------------------------+

.. need to add the telluric frames and calibrations when supported.
.. .. | Telluric || N20201026S0120-123 |
