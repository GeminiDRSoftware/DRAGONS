.. 04_tips_and_tricks.rst

.. _tips_and_tricks:

***************
Tips and Tricks
***************
This is a collection of tips and tricks that can be useful for reducing
different data, or to do it slightly differently from what is presented
in the example.

Bad Pixel Masks
===============

Please note that at this time, there are no static bad pixel masks for
Flamingos-2 data.  DRAGONS will simply acknowledge that in the logs and
continue with the reduction.

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

.. _checkWCS:

Checking WCS of science frames
==============================
For Flamingos-2 data, it is useful to check the World Coordinate System (WCS)
of the science data. DRAGONS will fix some small discrepancy but sometimes
the WCS are not written correctly in the headers causing difficulties with
the sky subtraction and frame alignment.

We recommend running ``checkWCS`` on the science files.

::

   $ reduce -r checkWCS @sci_images.list

   ======================================================================
   RECIPE: checkWCS
   ======================================================================
   PRIMITIVE: checkWCS
   -------------------
   Using S20200104S0075.fits as the reference
   S20200104S0080.fits has a discrepancy of 2.00 arcsec
   S20200104S0082.fits has a discrepancy of 2.01 arcsec
   S20200104S0091.fits has a discrepancy of 2.01 arcsec
   .

If any frames get flagged, like in the example above, you can still proceed
but after the reduction, do review the logs to check for any unusual matching
of the sources during ``adjustWCSToReference`` step, in particular the line
about the "Number of correlated sources".  If one of the highlighted frame
has a much lower number of correlated sources than the others, the algorithm
is unable to overcome the discrepancy; remove the file from the input list
and reduce again.

In general, discrepancies of the order of what is shown above do not cause
problems.  When the discrepancy matches the size of the dither, then you will
have issues and it is best to simply remove that file from you file list right
away.  When such a large discrepancy happens, the WCS of that file is likely
to have accidentally inherited the WCS of the previous frame which is
obviously very wrong.

.. note::  From the API, run ``checkWCS`` like this:

    .. code-block::

        checkwcs = Reduce()
        checkwcs.files = list_of_science_images
        checkwcs.recipename = 'checkWCS'
        checkwcs.runr()


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
* processed_standard

.. _useful_parameters:

Useful parameters
=================

skip_primitive
--------------
I might happen that you will want or need to not run a primitive in a recipe.
You could copy the recipe over and edit it.  Or you could invoke the
``skip_primitive`` parameter to tell DRAGONS to completely skip that step.

Let's say that you want the data aligned but not stacked.  You would do::

    reduce @sci.lis -p stackFrames:skip_primitive=True


write_outputs
-------------
When debugging or when there's a need to inspect intermediate products, you
might want to write the output of a specific primitive to disk.  This is done
with the ``write_outputs`` parameter.

For example, to write the sky subtracted frames before alignment and stacking,
you would do::

    reduce @sci.lis -p skyCorrect:write_outputs=True

