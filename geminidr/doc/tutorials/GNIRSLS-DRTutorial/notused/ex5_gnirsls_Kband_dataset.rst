.. ex1_gnirsls_Kband_dataset.rst

.. _data_Kband:

********************************
Example 1 - Datasets description
********************************

K-band Dithered Point Source Longslit
-------------------------------------
This is a GNIRS K-band longslit observation.  The primary target is an
ultra-cool dwarf.  We will us this observation to how how a longslit
K-band sequence, with the 111/mm grating and centered around 2.3um is reduced
with DRAGONS.  The sequence dithers along the slit.  DRAGONS will do the sky
subtraction, then stack the aligned spectra automatically.

Because the arc lamp has only three lines and the OH and O_2 lines are absent
beyond 2.3um, we will need to get the wavelength calibrations from the
telluric absorption lines in the science spectrum.

The calibrations we use for this example are:

* BPM.  The bad pixel masks are now found in the Gemini Science Archive
  instead of being packaged with the software. They are associated like the
  other calibrations.  (The date in the name is the "valid from"
  date.)
* Flats taken in the same configuration as the science.  They were obtained
  at night right after the science observations.
* An arc taken in the same configuration as the science and also obtained at
  night at the end of the science observation sequences.  Even though we
  expect to have to use the telluric absorption lines in the science data,
  it is recommended to get the arc anyway as a backup solution.

.. * telluric

.. todo:  I might need to download and reduce the telluric to get the
   wavecal from the telluric absorption feature as the science target is
   a busy spectrum.

.. warning::  The Bad Pixel Masks (BPMs) are now found in the archive rather
   than packaged with the software.  You must get the static BPM from the
   archive.  See :ref:`getBPM` in :ref:`tips_and_tricks`.

Here is the breakdown of the files.  All the files are included in the tutorial data
package.  They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------------+----------------------------------------------+
| Science             || N20180106S0158-165                          |
+---------------------+----------------------------------------------+
| Science flats       || N20180106S0166                              |
+---------------------+----------------------------------------------+
| Science arcs        || N20180106S0172                              |
+---------------------+----------------------------------------------+
| BPM                 || bpm_20100716_gnirs_gnirsn_11_full_1amp.fits |
+---------------------+----------------------------------------------+

.. need to add the telluric frames and calibrations when supported.
