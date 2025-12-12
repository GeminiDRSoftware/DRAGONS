.. ex1_f2ls_JHHK_dataset.rst

.. include:: symbols.txt

.. _f2ls_JHHK_dataset:

********************************
Example 1 - Datasets description
********************************

In this example, we will reduce Flamingos 2 longslit observations
a ??? obtained with the JH and the HK gratings.

The ?-pixel slit is used.

The calibrations we use for this example are:


NEEDS TO BE UPDATED, THIS IS EXAMPLE 3




* Darks with exposure times matching the flats, the arcs, the telluric
  star observation,
  and the science observations.  The darks are taken once a week.
* Lamp-on flats taken in the same configuration as the science and the telluric
  star and obtained at night after the science and telluric sequences.
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
| Telluric arc darks         || S20230610S0343-349                         |
+----------------------------+---------------------------------------------+
| Flat darks (16s)           || S20230610S0217,220,223,225,227,230,232,235 |
+----------------------------+---------------------------------------------+

.. * Lamp-off flats with exposure times matching the arcs.  In the K-band, there
..  is a thermal background in the arc lamp observations.  Removing it with a
..  lamp-off flat helps with the measurement of the wavelength solution.  If
..  obtained, the lamp-off flat is taken at night just before the arc.


Show what a dark looks like to convey why a dark correction is required.
Since no lampoff for flats, we need to dark correct.  Line detection in
arc lamps would be compromised with those dark lines still in.  And
for science/telluric, the dark current/features are known to be unstable
so extra steps can help.  (But I need to try without.)