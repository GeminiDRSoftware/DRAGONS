.. ex4_gnirsls_Kband111mm_red_dataset.rst

.. include:: symbols.txt

.. _gnirs_Kband111mm_red_example:

********************************
Example 4 - Datasets description
********************************

Beyond 2.3 microns K-band Longslit Point Source (111 l/mm grating)
==================================================================

In this example, we will reduce a GNIRS K-band longslit observation with a
central wavelength of 2.33 |um| with the 111 l/mm grating and the Long Blue
camera.  The dither pattern is a ABBA sequence.  The slit width is
0.3 arcsec.

The target is HD 179821, a star with a debated evolutionary state.  It is
believed to be either a post-asymtotic giant star or a yellow hypergiant.
There is an interesting discussion of the star based on this data in
Kraus, M. et al, 2023, Galaxies, volume 11, 76 (https://doi.org/10.3390/galaxies11030076).

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
* A telluric standard observation taken in the same configuration as the
  science and obtained at night just before or just after the science
  observation sequences, and at a similar airmass.

.. warning::  The Bad Pixel Masks (BPMs) are now found in the archive rather
   than packaged with the software.  You must get the static BPM from the
   archive.  See :ref:`getBPM` in :ref:`tips_and_tricks`.

Here is the breakdown of the files.  All the files are included in the tutorial data
package.  They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------------+----------------------------------------------+
| Science             || N20210407S0173-176                          |
+---------------------+----------------------------------------------+
| Science flats       || N20210407S0177-180                          |
+---------------------+----------------------------------------------+
| Science arcs        || N20210407S0181-182                          |
+---------------------+----------------------------------------------+
| Telluric            || N20210407S0188-191                          |
+---------------------+----------------------------------------------+
| BPM                 || bpm_20121101_gnirs_gnirsn_11_full_1amp.fits |
+---------------------+----------------------------------------------+

