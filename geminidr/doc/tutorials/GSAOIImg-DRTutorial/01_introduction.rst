.. 01_introduction.rst

.. _`AstroConda`: https://astroconda.readthedocs.io/en/latest/
.. _`DRAGONS`: https://github.com/GeminiDRSoftware/DRAGONS
.. _`installing AstroConda`: https://astroconda.readthedocs.io/en/latest/getting_started.html#getting-started-jump
.. _`installing DRAGONS`: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/install.html


.. _introduction:

************
Introduction
************

This tutorial covers the basics on reducing
`GSAOI <https://www.gemini.edu/sciops/instruments/gsaoi/>`_ (Gemini
South Adaptive Optics Imager) data using `DRAGONS`_ (Data
Reduction for Astronomy from Gemini Observatory North and
South).

The next two sections explain what are the required software and the data set
that we use throughout the tutorial.
`Chapter 2: Data Reduction <command_line_data_reduction>`_ contains a
quick example on how to reduce data using the DRAGONS command line tools.


.. _requirements:

Requirements
============

`DRAGONS`_ requires several libraries that could be installed individually. Most
of them can be obtained by `installing AstroConda`_. Just click the link and
follow the guidelines. It should take no more than ten minutes (if your internet
is fast).

Once you have `AstroConda`_ installed and you have set your new Virtual
Environment, download `DRAGONS`_ and install it. You can find the last release
version in the link below:

https://github.com/GeminiDRSoftware/DRAGONS/archive/v2.1.0-b1.zip

Decompress the downloaded file and access it using a (bash) terminal. Make sure
your Virtual Environment is activated. The installation steps are better
explained in `installing DRAGONS`_.


.. _download_sample_files:

Download Sample Files
=====================

GSAOI images suffer from a lot of distortion. Because of that, we chose to
run this tutorial on globular clusters that have point sources in the whole
field-of-view. The selected data set was observed for the GS-2017B-Q-53-15
program, in Dec 10, 2017, and is related to the
`Miller, 2019 <https://ui.adsabs.harvard.edu/#abs/2019AAS...23325007M/abstract>`_
published work. You can search and download the files on the
`Gemini Archive <https://archive.gemini.edu/searchform>`_ using the information
above, or simply copy the link below and past to your browser:::

    https://archive.gemini.edu/searchform/object=NGC+104/cols=CTOWEQ/notengineering/GSAOI/ra=6.0223292/20170201-20171231/science/dec=-72.0814444/NotFail/OBJECT


