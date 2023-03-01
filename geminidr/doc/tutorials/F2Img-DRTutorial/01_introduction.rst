.. 01_introduction.rst

.. _introduction:

************
Introduction
************

This tutorial covers the basics of reducing
`Flamingos-2 <https://www.gemini.edu/sciops/instruments/flamingos2/>`_  data
using `DRAGONS <https://dragons.readthedocs.io/>`_.

The F2 imaging tutorial series, for now, contains only one scientific
example: the reduction of a field of stars and distant
galaxies with dither on target for sky subtraction.

The reduction can be done in two different ways:

* From the terminal using the command line.
* From Python using the DRAGONS classes and functions.

We show how to run the same reduction using both methods.

* :ref:`ontarget_example`

More examples will be added in the future.

The next two sections explain what are the required software and the data set
that we use throughout the tutorial.

.. _requirements:

Software Requirements
=====================

Before you start, make sure you have `DRAGONS
<https://dragons.readthedocs.io/>`_ properly installed and configured on your
machine. You can test that by typing the following commands:

.. code-block:: bash

    $ conda activate dragons
    $ python -c "import astrodata"

Where ``dragons`` is the name of the conda environment where DRAGONS has
been installed. If you have an error message, make sure:

    - Conda is properly installed;

    - A Conda Virtual Environment is properly created and is active;

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

