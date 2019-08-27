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

Flatfields
----------
Y, J, and H-bands
+++++++++++++++++
Flamingos-2 Y, J and H master flats are created from lamps-on and
lamps-off flats. Both types are passed in together to the
``reduce`` command. The order does not matter. The software
separates the lamps-on and lamps-off flats and use them
appropriately.

K-band
++++++
For K-band master flats, lamp-off flats and darks are used. In
that case both flats (lamp-off only for K-band) and darks need
to be fed to ``reduce``. The darks' exposure time must match that
of the flats. The first input file to ``reduce`` must be a flat
for the correct recipe library to be selected. After that the
software will still sort out how to use the inputs appropriately to
produce the flat. For example::

    $ reduce @flats_K.list @darks_for_flats.list

The K-band thermal emission from the GCAL shutter depends upon the
temperature at the time of the exposure, and includes some spatial
structure. Therefore the distribution of emission is not necessarily
consistent, except for sequential exposures. So it is best to combine
lamp-off exposures from a single day.


Bypassing automatic calibration association
--------------------------------------------
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


Issues and Limitations
======================

Memory Issues
-------------
Some primitives use a lot of RAM memory and they can cause a
crash. Memory management in Python is notoriously difficult. The
DRAGONS's team is constantly trying to improve memory management
within ``astrodata`` and the DRAGONS recipes and primitives. If
an "Out of memory" crash happens to you, if possible for your
observation sequence, try to run the pipeline on fewer images at the time,
like for each dither pattern sequence separately.

.. todo::  We need to show the user how to bring them all back
     together in a final stack at the end. This means showing
     what custom recipe to use and how to invoke it.

.. _issue_p2:

Emission from PWFS2 guide probe
-------------------------------
The PWFS2 guide probe leaves a signature on imaging data that cannot be
removed. Ideally, one would be using the OIWFS, the On-Instrument Wave Front
Sensor, but at the time of this writing, it is not yet available, (see
`F2 instrument status note <https://www.gemini.edu/sciops/instruments/flamingos2/status-and-availability>`_
for Sep. 5, 2003). The effect of the PWFS2 guide probe will in many cases
compromise photometry in the region affected.

.. _double_messaging:

Double messaging issue
----------------------
If you run ``Reduce`` without setting up a logger, you will notice that the
output messages appear twice. To prevent this behaviour set up a logger.
This will send one of the output stream to a file, keeping the other on the
screen. We recommend using the DRAGONS logger located in the
:mod:`gempy.utils.logutils` module and its
:func:`~gempy.utils.logutils.config()` function:


.. code-block:: python
    :linenos:

    from gempy.utils import logutils
    logutils.config(file_name='f2_data_reduction.log')