.. ex1_gsaoiim_offsetsky_dataset.rst

.. _offsetsky_dataset:

*********************************
Example 1 - Datasets descriptions
*********************************

Crowded with offset to sky
--------------------------

The data are a GSAOI observation of the resolved outskirt of a nearby galaxy.
The observation is a dither-on-target with offset-to-sky sequence.

The table below contains a summary of the dataset downloaded in the previous
section.  Note that for GSAOI, the dark current is low enough that there is
no need to correct for it.

The calibrations we use in this example are:

* BPM.  The bad pixel masks are now found in the Gemini Science Archive
  instead of being packaged with the software. They are associated like the
  other calibrations.  Here we need the 2x2 binning GMOS-N Hamamatsu
  BPM valid for data taken in 2017.  (The date in the name is the "valid from"
  date.)
* Flats, as a sequence of lamps-on and lamps-off exposures.
* A Standard star that could be used for photometry.

.. warning::  The Bad Pixel Masks (BPMs) are now found in the archive rather
   than packaged with the software.  You must get the static BPM from the
   archive.  See :ref:`getBPM` in :ref:`tips_and_tricks`.


+---------------+---------------------+--------------------------------+
| Science       || S20170505S0095-110 || Kshort-band, on target, 60 s  |
+---------------+---------------------+--------------------------------+
| Flats         || S20170505S0030-044 || Lamp on, Kshort, for science  |
|               || S20170505S0060-074 || Lamp off, Kshort, for science |
+---------------+---------------------+--------------------------------+
| Standard star || S20170504S0114-117 || Kshort, standard star, 30 s   |
+---------------+---------------------+--------------------------------+
| BMP           || bpm_20121104_gsaoi_gsaoi_11_full_4amp.fits          |
+---------------+---------------------+--------------------------------+
