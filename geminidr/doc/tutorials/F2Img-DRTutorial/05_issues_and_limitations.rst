.. 05_issues_and_limitations.rst

.. _issues_and_limitations:

**********************
Issues and Limitations
**********************

Memory Issues
=============
Some primitives use a lot of RAM memory and they can cause a
crash. Memory management in Python is notoriously difficult. The
DRAGONS's team is constantly trying to improve memory management
within ``astrodata`` and the DRAGONS recipes and primitives. If
an "Out of memory" crash happens to you, if possible for your
observation sequence, try to run the pipeline on fewer images at the time,
like for each dither pattern sequence separately.

Then to align and stack the pieces, run the ``alignAndStack`` recipe:

.. code-block:: bash

    $ reduce @list_of_stacks -r alignAndStack


.. _issue_p2:

Emission from PWFS2 guide probe
===============================
The PWFS2 guide probe leaves a signature on imaging data that cannot be
removed. Ideally, one would be using the OIWFS, the On-Instrument Wave Front
Sensor, but there has been issues with it.  See
`<https://www.gemini.edu/sciops/instruments/flamingos2/status-and-availability>`_
for the current status of the instrument. The effect of the PWFS2 guide probe
will in many cases compromise photometry in the region affected.

.. _double_messaging:

Double messaging issue
======================
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
