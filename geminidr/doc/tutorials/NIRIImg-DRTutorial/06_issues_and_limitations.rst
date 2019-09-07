.. 06_issues_and_limitations.rst

.. include:: DRAGONS.txt

.. _issues_and_limitations:

**********************
Issues and Limitations
**********************

Memory Issues
=============
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
======================
If you run ``Reduce`` without setting up a logger, you will notice that the
output messages appear twice.  To prevent this behaviour set up a logger.
This will send one of the output stream to a file, keeping the other on the
screen.  We recommend using the DRAGONS logger located in the
``logutils`` module and its ``config()`` function:


.. code-block:: python
    :linenos:

    from gempy.utils import logutils
    logutils.config(file_name='niri_tutorial.log')
