.. ex1_f2im_ontarget_dataset.rst

.. _ontarget_dataset:

*********************************
Example 1 - Datasets descriptions
*********************************

Small sources with dither on target
-----------------------------------

This is a Flamingos-2 imaging observation of a field of stars and distant
galaxies with dither on target for sky subtraction.   We will use this
observation to show how a basic Flamingos 2 imaging sequence is reduced with
DRAGONS.

The sequence is an dithered-on-target sequence.  DRAGONS will recognized such
a sequence and will identify frames in the sequence to use for sky subtraction.
If your target is extended and you have used offsets to a blank section of
the sky to use for sky subtraction, please look at the NIRI tutorial where the
example uses an extended source.  The process will be very similar.

The data used here is obtained in the Y-band.  Please check the
:ref:`tips_and_tricks` section for information about the other bands,
especially for K-band.

The calibrations we use in this example are:

* Darks for the science frames.
* Flats, as a sequence of lamps-on and lamps-off exposures.
* Short darks to use with the flats to create a bad pixel mask.

The table below contains a summary of the files needed for this example:

+---------------+---------------------+--------------------------------+
| Science       || S20131121S0075-083 | Y-band, 120 s                  |
+---------------+---------------------+--------------------------------+
| Darks         || S20131121S0369-375 | 2 s, short darks for BPM       |
|               +---------------------+--------------------------------+
|               || S20131120S0115-120 | 120 s, for science data        |
|               || S20131121S0010     |                                |
|               || S20131122S0012     |                                |
|               || S20131122S0438-439 |                                |
+---------------+---------------------+--------------------------------+
| Flats         || S20131129S0320-323 | 20 s, Lamp On, Y-band          |
|               +---------------------+--------------------------------+
|               || S20131126S1111-116 | 20 s, Lamp Off, Y-band         |
+---------------+---------------------+--------------------------------+
