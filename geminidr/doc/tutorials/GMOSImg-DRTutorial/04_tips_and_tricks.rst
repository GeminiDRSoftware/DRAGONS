.. 04_tips_and_tricks.rst

.. include:: DRAGONSlinks.txt

.. _tips_and_tricks:

***************
Tips and Tricks
***************

This is a collection of tips and tricks that can be useful for reducing
different data, or to do it slightly differently from what is presented
in the example.

.. _process_fringe_frame:

Create Master Fringe Frame
==========================
The reduction of some datasets requires a master fringe frame. The filters
that need a fringe frame are shown in the appendix
:ref:`Fringe Correction Tables <fringe_correction_tables>`.

To create the master fringe frame from the dithered science observations and
add it to the calibration database:

.. code-block:: bash

    $ reduce @list_of_science.txt -r makeProcessedFringe
    $ caldb add N20170614S0201_fringe.fits

This command line will produce an image with the ``_fringe`` suffix in the
current working directory.

Again, note that this step is only needed for images obtained with some
detector and filter combinations. Make sure you checked the
`Fringe Correction Tables <fringe_correction_tables>`_.

The above can be done with the API as follow:

.. code-block:: python
    :linenos:

    reduce_fringe = Reduce()
    reduce_fringe.files.extend(list_of_science)
    reduce_fringe.recipename = 'makeProcessedFringe'
    reduce_fringe.runr()

    caldb.add_cal(reduce_fringe.output_filenames[0])



.. _bypassing_caldb:

Bypass automatic calibration association
========================================
We can think of two reasons why a user might want to bypass the calibration
manager and the automatic processed calibration association. The first is
to override the automatic selection, to force the use of a different processed
calibration than what the system finds. The second is if there is a problem
with the calibration manager and it is not working for some reason.

Whatever the specific situation, the following syntax can be used to bypass
the calibration manager and set the input processed calibration yourself.

.. code-block:: bash

     $ reduce @sci_images.list --user_cal processed_bias:S20001231S0001_bias.fits processed_flat:S20001231S0002_flat.fits

The list of recognized processed calibration is:

* processed_arc
* processed_bias
* processed_dark
* processed_flat
* processed_fringe


Browse Recipes and Primitives
=============================
"|reduce|", either the command line or the API class, is the tool that selects
and run a "recipe".  A recipe is a sequence of operations called "primitives".
Each primitives has a defined set of input parameters with default values that
can be overriden by the user.

The "|showrecipes|" command line is used to show the default recipe for a
file, a specific recipe for that file, or all the recipes associated with
the file. It is fully documented in:

    * `Recipe System - User's Manual: showrecipes <https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#showrecipes>`_

Once you know the recipe and primitives it is calling, you can explore the
primitives' parameters using the "|showpars|" command line. The tool is fully
documented in:

    * `Recipe System - User's Manual: showpars <https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#showpars>`_


Customizing input parameters
============================

From the command line, setting the value of a primitive input parameter is
done as follow:

.. code-block:: bash

    $ reduce @sci.lis -p stackFrames:scale=True

The ``-p`` flag indicates that the following items are parameter changes.  The
syntax is ``<primitive_name>:<parameter_name>=<value>``

From the API, the ``uparms`` attribute to the ``Reduce`` instance is used.

.. code-block:: python
    :linenos:

    reduce_science.uparms.append(("stackFrames:scale", True))


Setting the output suffix
=========================
When troubleshooting an issue or trying various settings to optimize a
reduction, it might be useful to name the final recipe output differently for
each attempt.

Only the **suffix** of the final output file can be changed, not its full name.

From the command line:

.. code-block:: bash

    $ reduce @sci.lis --suffix='newsuffix'

From the API:

.. code-block:: python
    :linenos:

    reduce_science.suffix = "newsuffix"
    reduce_science.runr()
