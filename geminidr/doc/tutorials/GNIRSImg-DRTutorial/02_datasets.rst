.. datasets.rst

.. _datasets:

********************************
Setting up and tutorial datasets
********************************

.. _datasetup:

Downloading the tutorial datasets
=================================

All the data needed to run this tutorial are found in the tutorial's data
package:

    `<http://www.gemini.edu/sciops/data/software/datapkgs/gnirsimg_tutorial_datapkg-v1.tar>`_

Download it and unpack it somewhere convenient.

.. highlight:: bash

::

    cd <somewhere convenient>
    tar xvf gnirsimg_tutorial_datapkg-v1.tar
    bunzip2 gnirsimg_tutorial/playdata/*.bz2

The datasets are found in the subdirectory ``gnirsimg_tutorial/playdata``, and we
will work in the subdirectory named ``gnirsimg_tutorial/playground``.

.. note:: All the raw data can also be downloaded from the Gemini Observatory
   Archive.  Using the tutorial data package is probably more convenient.


Datasets descriptions
=====================

.. _datakeyhole:

Dataset for example 1: Point source through the acquisition keyhole
-------------------------------------------------------------------

This is a GNIRS acquisition keyhole imaging observation of a point source.
The observation sequence uses dither-on-target.  Dithered observations
nearby in time will be used for sky subtraction of each frame.

The calibrations we use for this example include:

* Darks for the science frames
* Flats, as a sequence of lamps-on and lamps-off exposures

Here is the files breakdown.  They are included in a tutorial data package.
They can also be downloaded from the Gemini Observatory Archive (GOA).

+---------------+--------------------------------------------+
| Science       || N20120117S0014-33 (J-band, on-target)     |
+---------------+--------------------------------------------+
| Science darks || N20120102S0538-547 (60 sec, like Science) |
+---------------+--------------------------------------------+
| Flats         || N20120117S0034-41 (lamps-on)              |
|               || N20120117S0042-49 (lamps-off)             |
+---------------+--------------------------------------------+

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
