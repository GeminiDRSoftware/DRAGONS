.. ex2_f2ls_R3KJband_dataset.rst

.. include:: symbols.txt

.. _f2ls_R3KJband_dataset:

********************************
Example 2 - Datasets description
********************************

In this example, we will reduce Flamingos 2 R3K longslit observations
centered in the J-band of the recurrent nova V1047 Cen.

The 2-pixel slit is used and the central wavelength is set to 1.25 |um|.
The dither pattern is a standard ABBA.

The calibrations we use for this example are:

* Darks with exposure times matching the flats, the arcs, the telluric
  star observation, and the science observations.  The darks are taken
  once a week.  See :ref:`why_darks` for addtional information.
* A Lamp-on flat taken in the same configuration as the science and the telluric
  star and obtained at night after the science. No additional flat was
  obtained after the telluric star.  The one we have will be used for both
  the science and the telluric.
* Arcs taken taken in the same configuration as the science and the telluric
  star and obtained at night after the science and telluric sequences.
* A telluric standard observation taken in the same configuration as the
  science and obtained, in this case, just before the start of the science
  sequence, and at a similar airmass.

Here is the breakdown of the files.  All the files are included in the
tutorial data package.  They can also be downloaded from the Gemini
Observatory Archive (GOA).  The science program is GS-2019A-DD-111.

+----------------------------+---------------------------------------------+
| Science                    || S20190702S0107-110                         |
+----------------------------+---------------------------------------------+
| Science darks (300s)       || S20190706S0431-437                         |
+----------------------------+---------------------------------------------+
| Science flat               || S20190702S0111                             |
+----------------------------+---------------------------------------------+
| Science flat darks (5s)    || S20190629S0029-035                         |
+----------------------------+---------------------------------------------+
| Science arc                || S20190702S0112                             |
+----------------------------+---------------------------------------------+
| Science arc darks (60s)    || S20190629S0085-091                         |
+----------------------------+---------------------------------------------+
| Science arc flat           || Same as science flat                       |
+----------------------------+---------------------------------------------+
| Telluric                   || S20190702S0099-102                         |
+----------------------------+---------------------------------------------+
| Telluric darks (25s)       || S20190706S0340-346                         |
+----------------------------+---------------------------------------------+
| Telluric flat              || Same as science flat                       |
+----------------------------+---------------------------------------------+
| Telluric flat darks (16s)  || Same as science flat darks                 |
+----------------------------+---------------------------------------------+
| Telluric arc               || S20190702S0103                             |
+----------------------------+---------------------------------------------+
| Telluric arc darks         || Same as science arc darks                  |
+----------------------------+---------------------------------------------+
| Telluric arc flat          || Same as telluric flat                      |
+----------------------------+---------------------------------------------+



