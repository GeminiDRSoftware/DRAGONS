.. 05_tips_and_tricks.rst

.. include:: DRAGONSlinks.txt

.. _tips_and_tricks:

***************
Tips and Tricks
***************

This is a collection of tips and tricks that can be useful for reducing
different data, or to do it slightly differently from what is presented
in the example.

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

     $ reduce @sci_images.list --user_cal processed_dark:N20120102S0538_dark.fits processed_flat:N20120117S0034_flat.fits

The list of recognized processed calibration is:

* processed_arc
* processed_bias
* processed_dark
* processed_flat
* processed_fringe

