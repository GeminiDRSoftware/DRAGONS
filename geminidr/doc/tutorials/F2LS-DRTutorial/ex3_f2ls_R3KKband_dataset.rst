.. ex3_f2ls_R3KKband_dataset.rst

.. include:: symbols.txt

.. _f2ls_R3KKband_dataset:

********************************
Example 3 - Datasets description
********************************

In this example, we will reduce Flamingos 2 R3K longslit observations
centered in the K-band of the superluminal microquasar GRS 1915+105.

The 2-pixel slit is used and the central wavelength is set to 2.2 |um|.
For expediency, we will reduced only the last eight frames of the long
sequence.  The dither pattern is AABB-AABB.

The calibrations we use for this example are:

* Darks with exposure times matching the flats, the arcs, the telluric
  star observation, and the science observations.  The darks are taken
  once a week.  See :ref:`why_darks` for addtional information.
* Lamp-on flats taken in the same configuration as the science and the telluric
  star and obtained at night after the science and telluric sequences.
* Arcs taken taken in the same configuration as the science and the telluric
  star and obtained at night after the science and telluric sequences.
* A telluric standard observation taken in the same configuration as the
  science and obtained, in this case, just after the completion of the science
  sequence, and at a similar airmass.

Here is the breakdown of the files.  All the files are included in the
tutorial data package.  They can also be downloaded from the Gemini
Observatory Archive (GOA).  The science program is GS-2023A-DD-109.

+----------------------------+---------------------------------------------+
| Science                    || S20230606S0083-090                         |
+----------------------------+---------------------------------------------+
| Science darks (120s)       || S20230610S0434-440                         |
+----------------------------+---------------------------------------------+
| Science flat               || S20230606S0091                             |
+----------------------------+---------------------------------------------+
| Science flat darks (16s)   || S20230610S0217,220,223,225,227,230,232,235 |
+----------------------------+---------------------------------------------+
| Science arc                || S20230606S0093                             |
+----------------------------+---------------------------------------------+
| Science arc darks (180s)  || S20230610S0343-349                          |
+----------------------------+---------------------------------------------+
| Science arc flat           || Same as science flat                       |
+----------------------------+---------------------------------------------+
| Telluric                   || S20230606S0097-100                         |
+----------------------------+---------------------------------------------+
| Telluric darks (15s)       || S20230610S0200,203,205,208,211,213         |
+----------------------------+---------------------------------------------+
| Telluric flat              || S20230606S0101                             |
+----------------------------+---------------------------------------------+
| Telluric flat darks (16s)  || Same as science flat darks                 |
+----------------------------+---------------------------------------------+
| Telluric arc               || S20230606S0103                             |
+----------------------------+---------------------------------------------+
| Telluric arc darks         || Same as science arc darks                  |
+----------------------------+---------------------------------------------+
| Telluric arc flat          || Same as telluric flat                      |
+----------------------------+---------------------------------------------+

.. note::
    In the R3K K-band configuration, a lamp-off flat matching the arc's
    exposure time is obtained with the sequence.  There is thermal emission
    in the arc lamp data.  The lamp-off is used to removed that background,
    which is a required step when using the Gemini IRAF package.

    This lamp-off and the removal of the thermal background is NOT required
    in DRAGONS.  The wavelength solution algorithm in DRAGONS takes care of
    the background. In DRAGONS, the recommended reduction procedure is to
    ignore the lamp-off flat.  However, if you still wish to use it, see
    :ref:`arc_lampoff` for how to do it.


