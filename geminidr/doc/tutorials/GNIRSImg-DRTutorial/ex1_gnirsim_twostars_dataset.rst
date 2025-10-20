.. ex1_gnirsim_twostars_dataset.rst

.. role:: raw-html(raw)
   :format: html

.. |verticalpadding| replace:: :raw-html:`<br>`

.. _twostars_dataset:

*********************************
Example 1 - Datasets descriptions
*********************************

Acquisition keyhole imaging of two stars with dithers
-----------------------------------------------------

This is a GNIRS acquisition keyhole imaging observation of two point sources.
The observation sequence uses dither-on-target. Dithered observations nearby
in time will be used for sky subtraction of each frame.

The calibrations we use for this example are:

* BPM. The bad pixel masks are now found in the Gemini Science Archive
  instead of being packaged with the software. They are associated like the
  other calibrations.  (The date in the name is the "valid from"
  date.)
* Darks for the science frames
* Flats, as a sequence of lamps-on and lamps-off exposures

.. warning::  The Bad Pixel Masks (BPMs) are now found in the archive rather
   than packaged with the software.  You must get the static BPM from the
   archive.  See :ref:`getBPM` in :ref:`tips_and_tricks`.

Here is the breakdown of the files.  They are included in a tutorial data package.
They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------+----------------------------------------------+
| Science       || N20120117S0014-33 (J-band, on-target)       |
+---------------+----------------------------------------------+
| Science darks || N20120102S0538-547 (60 sec, like Science)   |
+---------------+----------------------------------------------+
| Flats         || N20120117S0034-41 (lamps-on)                |
|               || N20120117S0042-49 (lamps-off)               |
+---------------+----------------------------------------------+
| BPM           || bpm_20100716_gnirs_gnirsn_11_full_1amp.fits |
+---------------+----------------------------------------------+

A note about finding the darks in the GOA.  Since GNIRS is not an imager and
imaging through the keyhole is done only in extreme circumstances, the archive
does not have calibration association rules for the darks in this case.  One
needs to manually search for the darks.  Here is the search that was done to
find the darks for this observation sequence:

* Set a date range around the dates of the science observations.  In this case
  we used "20120101-20120131".
* Set **Instrument** to GNIRS.
* Set **Obs.Type** to DARK.
* Set the exposure time to 60 seconds.
* On the result table, select the darks with a "Pass" setting in the "QA" column.
