.. 05_issues_and_limitations.rst

.. include:: DRAGONSlinks.txt

.. _issues_and_limitations:

**********************
Issues and Limitations
**********************

Missing Processed Dark Calibration Association Rules
====================================================
GNIRS not being an imager, and the keyhole being normally used only for
acquisition, it turns out that there are no calibration association rules
between GNIRS keyhole images and darks.  This is recently discovered
limitation that we plan to fix in a future release.  In the meantime, the
user can simply specify the dark on the command line as shown in the previous
chapter in the :ref:`bypass_caldb` section.


Memory Issues
=============
Some primitives use a lot of RAM memory and they can cause a
crash. Memory management in Python is notoriously difficult. The
DRAGONS's team is constantly trying to improve memory management
within ``astrodata`` and the DRAGONS recipes and primitives.  If
an "Out of memory" crash happens to you, if possible for your
observation sequence, try to run the pipeline on fewer images at the time,
like for each dither pattern sequence separately.

Then to align and stack the pieces, run the ``alignAndStack`` recipe:

.. code-block:: bash

    $ reduce @list_of_stacks -r alignAndStack

For GNIRS, this issue is very rare given that the GNIRS detector is fairly
small and also rarely used for imaging, but it could happen when trying to
reduce a very large number of frames in one go.


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
    logutils.config(file_name='gnirs_tutorial.log')
