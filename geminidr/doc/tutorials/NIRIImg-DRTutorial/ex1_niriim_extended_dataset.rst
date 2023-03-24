.. ex1_niriim_extended_dataset.rst

.. _extended_dataset:

*********************************
Example 1 - Datasets descriptions
*********************************

Extended source with offset to sky
----------------------------------

This is a NIRI imaging observation of an extended source, a galaxy showing
as a dense field of stars.  The observation sequence uses an offset to a nearby
blank portion of the sky to monitor the sky levels since there are no area in
the science observation that is not "contaminated" by the galaxy.

The calibrations we use for this example are:

* BPM. The bad pixel masks are now found in the Gemini Science Archive
  instead of being packaged with the software. They are associated like the
  other calibrations.  (The date in the name is the "valid from"
  date.)
* Darks for the science and sky offset frames.
* Flats, as a sequence of lamps-on and lamps-off exposures.
* Short darks to use with the flats to create a bad pixel mask.
* A set of standard star observations.

Here is the breakdown of the files.  They are included in the tutorial data package.
They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------+--------------------------------------------+
| Science       || N20160102S0270-274 (on-target)            |
|               || N20160102S0275-279 (on-sky)               |
+---------------+--------------------------------------------+
| Science darks || N20160102S0423-432 (20 sec, like Science) |
+---------------+--------------------------------------------+
| Flats         || N20160102S0373-382 (lamps-on)             |
|               || N20160102S0363-372 (lamps-off)            |
+---------------+--------------------------------------------+
| Short darks   || N20160103S0463-472                        |
+---------------+--------------------------------------------+
| Standard star || N20160102S0295-299                        |
+---------------+--------------------------------------------+
| BPM           || bpm_20010317_niri_niri_11_full_1amp.fits  |
+---------------+--------------------------------------------+


A note about finding the short darks in the GOA.  Those darks are used solely
to create a fresh bad pixel mask (BPM).  In the archive, the calibration
association will not find those darks, they need to be searched for
explicitly. If you need to find short darks for your program, do as follow:

* Set a date range around the dates of your science observations.
* Set **Instrument** to NIRI.
* Set **Obs.Type** to DARK.
* Set the exposure time to 1 second.
