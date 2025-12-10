.. ex3_f2ls_R3KKband_dataset.rst

.. include:: symbols.txt

.. _f2ls_R3KKband_dataset:

********************************
Example 3 - Datasets description
********************************

In this example, we will reduce Flamingos 2 R3K longslit observations
centered in the K-band of a high-mass X-ray binary.

The 2-pixel slit is used and the central wavelength is set to 2.2 |um|.
For expediency, we will reduced only the last eight frames of the long
sequence.  The pattern is AABB-AABB.

The calibrations we use for this example are:

* Darks with exposure times matching the flats, the telluric star observation,
  and the science observations.  The darks are taken once a week.
* Lamp-on flats taken in the same configuration as the science and the telluric
  star and obtained at night after the science and telluric sequences.
* Lamp-off flats with exposure times matching the arcs.  In the K-band, there
  is a thermal background in the arc lamp observations.  Removing it with a
  lamp-off flat helps with the measurement of the wavelength solution.  If
  obtained, the lamp-off flat is taken at night just before the arc.
* Arcs taken at night after the science data and after the telluric data.
* A telluric standard observation taken in the same configuration as the
  science and obtained, in this case, just after the completion of the science
  sequence, and at a similar airmass.

Here is the breakdown of the files.  All the files are included in the
tutorial data package.  They can also be downloaded from the Gemini
Observatory Archive (GOA).

+----------------------------+---------------------------------------------+
| Science                    || S20230606S0083-090                         |
+----------------------------+---------------------------------------------+
| Science darks (120s)       || S20230610S0434-440                         |
+----------------------------+---------------------------------------------+
| Science flat               || S20230606S0091                             |
+----------------------------+---------------------------------------------+
| Science arc                || S20230606S0093                             |
+----------------------------+---------------------------------------------+
| Science arc darks (180s)  || S20230610S0343-349                          |
+----------------------------+---------------------------------------------+
| Telluric                   || S20230606S0097-100                         |
+----------------------------+---------------------------------------------+
| Telluric darks (15s)       || S20230610S0200,203,205,208,211,213         |
+----------------------------+---------------------------------------------+
| Telluric flat              || S20230606S0101                             |
+----------------------------+---------------------------------------------+
| Telluric arc               || S20230606S0103                             |
+----------------------------+---------------------------------------------+
| Telluric arc lamp-off flat || S20230606S0102                             |
+----------------------------+---------------------------------------------+
| Flat darks (16s)           || S20230610S0217,220,223,225,227,230,232,235 |
+----------------------------+---------------------------------------------+

arc darks (180s)