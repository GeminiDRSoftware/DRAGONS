.. 05_tips_and_tricks.rst

.. _tips_and_tricks:

***************
Tips and Tricks
***************

This is a collection of tips and tricks that can be useful for reducing
different data, or to do it slightly differently from what is presented
in the example.

.. _checkWCS:

Checking WCS of science frames
==============================
For GNIRS data, it is useful to check the World Coordinate System (WCS)
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




.. _bypass_caldb:

Bypassing automatic calibration association
===========================================
We can think of two reasons why a user might want to bypass the calibration
manager and the automatic processed calibration association.  The first is
to override the automatic selection, to force the use of a different processed
calibration than what the system finds.  The second is if there is a problem
with the calibration manager and it is not working for some reason.

Whatever the specific situation, the following syntax can be used to bypass
the calibration manager and set the input processed calibration yourself::

     $ reduce @target.lis --user_cal processed_dark:N20120102S0538_dark.fits processed_flat:N20120117S0034_flat.fits

The list of recognized processed calibration is:

* processed_arc
* processed_bias
* processed_dark
* processed_flat
* processed_fringe
* processed_standard

