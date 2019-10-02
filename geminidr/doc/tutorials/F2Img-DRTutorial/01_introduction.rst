.. 01_introduction.rst

.. _introduction:

************
Introduction
************

This tutorial covers the basics of reducing
`Flamingos-2 <https://www.gemini.edu/sciops/instruments/flamingos2/>`_  data
using `DRAGONS <https://dragons.readthedocs.io/>`_.

The next two sections explain what are the required software and the data set
that we use throughout the tutorial.
:ref:`Chapter 2: Data Reduction <command_line_data_reduction>` contains a
quick example on how to reduce data using the DRAGONS command line tools.
:ref:`Chapter 3: Reduction with API <api_data_reduction>` shows how we can
reduce the data using DRAGONS packages from within Python.


.. _requirements:

Software Requirements
=====================

Before you start, make sure you have `DRAGONS
<https://dragons.readthedocs.io/>`_ properly installed and configured on your
machine. You can test that by typing the following commands:

.. code-block:: bash

    $ conda activate geminiconda
    $ python -c "import astrodata"

Where ``geminiconda`` is the name of the conda environment where DRAGONS has
been installed. If you have an error message, make sure:

    - Conda is properly installed;

    - A Conda Virtual Environment is properly created and is active;

    - AstroConda (STScI) is properly installed within the Virtual Environment;

    - DRAGONS was successfully installed within the Conda Virtual Environment;


.. _datasetup:

Downloading the tutorial datasets
=================================

All the data needed to run this tutorial are found in the tutorial's data
package:

    `<http://www.gemini.edu/sciops/data/software/datapkgs/f2img_tutorial_datapkg-v1.tar>`_

Download it and unpack it somewhere convenient.

.. highlight:: bash

::

    cd <somewhere convenient>
    tar xvf f2img_tutorial_datapkg-v1.tar
    bunzip2 f2img_tutorial/playdata/*.bz2

The datasets are found in the subdirectory ``f2img_tutorial/playdata``, and we
will work in the subdirectory named ``f2img_tutorial/playground``.

.. note:: All the raw data can also be downloaded from the Gemini Observatory
          Archive. Using the tutorial data package is probably more convenient
          but if you really want to learn how to search for and retrieve the
          data yourself, see the step-by-step instructions in the appendix,
          :ref:`goadownload`.


.. _about_data_set:

About the dataset
=================

Dither-on-target
----------------
This is a Flamingos-2 imaging observation of a star and distant galaxy field
with dither on target for sky subtraction.

The calibrations we use in this example include:

* Darks for the science frames.
* Flats, as a sequence of lamps-on and lamps-off exposures.
* Short darks to use with the flats to create a bad pixel mask.

The table below contains a summary of the files needed for this example:

+---------------+---------------------+--------------------------------+
| Science       || S20131121S0075-083 | Y-band, 120 s                  |
+---------------+---------------------+--------------------------------+
| Darks         || S20131121S0369-375 | 2 s, short darks for BPM       |
|               +---------------------+--------------------------------+
|               || S20131120S0115-120 | 120 s, for science data        |
|               || S20131121S0010     |                                |
|               || S20131122S0012     |                                |
|               || S20131122S0438-439 |                                |
+---------------+---------------------+--------------------------------+
| Flats         || S20131129S0320-323 | 20 s, Lamp On, Y-band          |
|               +---------------------+--------------------------------+
|               || S20131126S1111-116 | 20 s, Lamp Off, Y-band         |
+---------------+---------------------+--------------------------------+
