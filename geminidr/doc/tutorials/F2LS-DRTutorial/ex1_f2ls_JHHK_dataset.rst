.. ex1_f2ls_JHHK_dataset.rst

.. include:: symbols.txt

.. _f2ls_JHHK_dataset:

********************************
Example 1 - Datasets description
********************************

.. todo:  add a warning that the read_mode need to match for dark association
          I got a case where some flats and arcs are taken in 1 and others
          in the same program for same disperser with 8.

In this example, we will reduce Flamingos 2 longslit observations
of the 2022 eruption of the recurrent nova U Sco.  The data were
obtained with the JH and the HK gratings.

The 2-pixel slit is used.  The dither sequence is ABBA-ABBA.

The calibrations we use for this example are:

* Darks with exposure times matching the flats, the arcs, the telluric
  star observation, and the science observations.  The darks are
  taken once a week.
* Lamp-on flats taken in the same configurations as the science and
  the telluric star observations. They are obtained at night after the
  science and telluric sequences.
* Arcs taken at night after the science data and after the telluric data.
* Telluric standard observations taken in the same configurations as the
  science and obtained either before or after the completion of the science
  sequence, and at a similar airmass.

Here is the breakdown of the files.  All the files are included in the
tutorial data package.  They can also be downloaded from the Gemini
Observatory Archive (GOA).

There are two configurations used in this example: one with the JH grating
and one with the HK grating.   We separate them into two tables for clarity.

**JH grating**

+----------------------------+---------------------------------------------+
| Science                    || S20220617S0044-047                         |
+----------------------------+---------------------------------------------+
| Science darks (25s)        ||  S20220618S0556-562                        |
+----------------------------+---------------------------------------------+
| Science flat               || S20220617S0048                             |
+----------------------------+---------------------------------------------+
| Science flat darks (8s)    || S20220618S0423-429                         |
+----------------------------+---------------------------------------------+
| Science arc                || S20220617S0049                             |
+----------------------------+---------------------------------------------+
| Science arc darks (180s)  || S20220618S0465-471                          |
+----------------------------+---------------------------------------------+
| Science arc flat           || Same as science flat                       |
+----------------------------+---------------------------------------------+
| Telluric (HIP 83920)       || S20220617S0073-076                         |
+----------------------------+---------------------------------------------+
| Telluric darks (15s)       || S20220618S0451-457                         |
+----------------------------+---------------------------------------------+
| Telluric flat              || S20220617S0077                             |
+----------------------------+---------------------------------------------+
| Telluric flat darks (16s)  || Same as science flat darks                 |
+----------------------------+---------------------------------------------+
| Telluric arc               || S20220617S0078                             |
+----------------------------+---------------------------------------------+
| Telluric arc darks         || Same as science arc darks                  |
+----------------------------+---------------------------------------------+
| Telluric arc flat          || Same as telluric flat                      |
+----------------------------+---------------------------------------------+

**HK grating**

+----------------------------+---------------------------------------------+
| Science                    || S20220617S0038-041                         |
+----------------------------+---------------------------------------------+
| Science darks (25s)        ||  S20220618S0556-562                        |
+----------------------------+---------------------------------------------+
| Science flat               || S20220617S0042                             |
+----------------------------+---------------------------------------------+
| Science flat darks (90s)   || S20220618S0521-527                         |
+----------------------------+---------------------------------------------+
| Science arc                || S20220617S0043                             |
+----------------------------+---------------------------------------------+
| Science arc darks (60s)   || S20220618S0507-513                          |
+----------------------------+---------------------------------------------+
| Science arc flat           || Same as science flat                       |
+----------------------------+---------------------------------------------+
| Telluric (HIP 79156)       || S20220617S0027-030                         |
+----------------------------+---------------------------------------------+
| Telluric darks (6s)       || S20220702S0026-034                          |
+----------------------------+---------------------------------------------+
| Telluric flat              || S20220617S0031                             |
+----------------------------+---------------------------------------------+
| Telluric flat darks (16s)  || Same as science flat darks                 |
+----------------------------+---------------------------------------------+
| Telluric arc               || S20220617S0032                             |
+----------------------------+---------------------------------------------+
| Telluric arc darks         || Same as science arc darks                  |
+----------------------------+---------------------------------------------+
| Telluric arc flat          || Same as telluric flat                      |
+----------------------------+---------------------------------------------+
