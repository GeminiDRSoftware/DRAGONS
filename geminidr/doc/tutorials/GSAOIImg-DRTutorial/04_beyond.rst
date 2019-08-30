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

Sky Subtraction
---------------
For sky subtraction, there are two input parameters to ``skyCorrect`` that
users should be aware of:  ``scale_sky`` and ``offset_sky``.  Both serve to
match the sky frames to the target frame before the subtraction.  The first,
``scale_sky`` is multiplicative and is turned off by default for GSAOI, while
the second, ``offset_sky`` is additive and is turned **on** by default for
GSAOI.

The reason why ``offset_sky`` is favored for GSAOI is that often the flux in
individual pixels can be very low and that is observed to make the
multiplicative scale less accurate.  In any case, from experience, it was
found that ``offset_sky==True`` was more successful, more often, with GSAOI
data, which is why it was set as the default.

Depending on the data and the science objectives, those two input parameters
might have to be experimented with.  The only combination we would not
recommend is setting both of them on.  (The software will not let you either.)

When there are offset to sky, it is likely to be because the target fills the
field of view and there is no usable sky.  In those cases, all sky scaling
and offsetting should be turned off (``skyCorrect:scale_sky=False`` and
``skyCorrect:offset_sky=False``).  There is no sky to measure in the target
frame, any attempts at scaling or offsetting will result in an over subtraction
of the sky.


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


.. _double_messaging:

Double messaging issue
----------------------
If you run the ``Reduce`` API without setting up a logger, you will notice
that the output messages appear twice. To prevent this behaviour set up a
logger. This will send one of the output stream to a file, keeping the other
on the screen. We recommend using the DRAGONS logger located in the
:mod:`gempy.utils.logutils` module and its
:func:`~gempy.utils.logutils.config()` function:


.. code-block:: python
    :linenos:

    from gempy.utils import logutils
    logutils.config(file_name='gsaoi_data_reduction.log')