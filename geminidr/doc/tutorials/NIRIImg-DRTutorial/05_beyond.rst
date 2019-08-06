.. 04_beyond.rst

.. _beyond:

*************************
Going beyond the examples
*************************

Tips and Tricks
===============
This is a collection of tips and tricks that can be useful for reducing
different data, or to do it slightly differently from what is presented
in the example.

Bypassing automatic calibration association
--------------------------------------------
We can think of two reasons why a user might want to bypass the calibration
manager and the automatic processed calibration association.  The first is
to override the automatic selection, to force the use of a different processed
calibration than what the system finds.  The second is if there is a problem
with the calibration manager and it is not working for some reason.

Whatever the specific situation, the following syntax can be used to bypass
the calibration manager and set the input processed calibration yourself::

     $ reduce @sci_images.list --user_cal processed_dark:N20160102S0423_dark.fits processed_flat:N20160102S0373_flat.fits

The list of recognized processed calibration is:

* processed_arc
* processed_bias
* processed_dark
* processed_flat
* processed_fringe


Issues and Limitations
======================

Memory Issues
-------------
Some primitives use a lot of RAM memory and they can cause a
crash. Memory management in Python is notoriously difficult. The
DRAGONS's team is constantly trying to improve memory management
within ``astrodata`` and the DRAGONS recipes and primitives.  If
an "Out of memory" crash happens to you, if possible for your
observation sequence, try to run the pipeline on fewer images at the time,
like for each dither pattern sequence separately.

For NIRI, this issue is relatively rare given that the NIRI detector is fairly
small, but it could happen when trying to reduce a very large number of
frames in one go.

.. todo::  We need to show the user how to bring them all back
     together in a final stack at the end.  This means showing
     what custom recipe to use and how to invoke it.


.. _double_messaging:

Double messaging issue
----------------------
If you run ``Reduce`` without setting up a logger, you will notice that the
output messages appear twice.  To prevent this behaviour set up a logger.
This will send one of the output stream to a file, keeping the other on the
screen.  We recommend using the DRAGONS logger located in the
``logutils`` module and its ``config()`` function:


.. code-block:: python
    :linenos:

    from gempy.utils import logutils
    logutils.config(file_name='niri_tutorial.log')