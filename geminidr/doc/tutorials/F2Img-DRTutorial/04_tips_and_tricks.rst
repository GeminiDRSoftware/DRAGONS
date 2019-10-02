.. 04_tips_and_tricks.rst

.. include:: DRAGONSlinks.txt

.. _tips_and_tricks:

***************
Tips and Tricks
***************
This is a collection of tips and tricks that can be useful for reducing
different data, or to do it slightly differently from what is presented
in the example.


Flatfields
==========
Y, J, and H-bands
-----------------
Flamingos-2 Y, J and H master flats are created from lamps-on and
lamps-off flats. Both types are passed in together to the
"|reduce|" command. The order does not matter. The software
separates the lamps-on and lamps-off flats and use them
appropriately.

K-band
------
For K-band master flats, lamp-off flats and darks are used. In
that case both flats (lamp-off only for K-band) and darks need
to be fed to "|reduce|". The darks' exposure time must match that
of the flats. The first input file to "|reduce|" must be a flat
for the correct recipe library to be selected. After that the
software will sort out how to use the inputs appropriately to
produce the flat. For example::

    $ reduce @flats_K.list @darks_for_flats.list

The K-band thermal emission from the GCAL shutter depends upon the
temperature at the time of the exposure, and includes some spatial
structure. Therefore the distribution of emission is not necessarily
consistent, except for sequential exposures. So it is best to combine
lamp-off exposures from a single day.


.. _bypassing_caldb:

Bypassing automatic calibration association
===========================================
We can think of two reasons why a user might want to bypass the calibration
manager and the automatic processed calibration association. The first is
to override the automatic selection, to force the use of a different processed
calibration than what the system finds. The second is if there is a problem
with the calibration manager and it is not working for some reason.

Whatever the specific situation, the following syntax can be used to bypass
the calibration manager and set the input processed calibration yourself::

     $ reduce @sci_images.list --user_cal processed_dark:S20131120S0115_dark.fits processed_flat:S20131129S0320_flat.fits

The list of recognized processed calibration is:

* processed_arc
* processed_bias
* processed_dark
* processed_flat
* processed_fringe
