.. intro.rst

.. _intro_usermanual:

************
Introduction
************

This is the AstroData User's Manual. AstroData is a DRAGONS package.
The current chapter covers basic concepts
like what is the |astrodata| package and how to install it (together with the
other DRAGONS' packages). :ref:`Chapter 2 <structure>`
explains with more details what is |AstroData| and how the data is represented
using it. :ref:`Chapter 3 <iomef>` describes input and output operations and
how multi-extension (MEF) FITS files are represented. :ref:`Chapter 4 <tags>`
provides information regarding the |TagSet| class, its usage and a few advanced
topics. In :ref:`Chapter 5 <headers>` you will find information about the FITS
headers and how to access/modify the metadata. The last two chapters,
:ref:`Chapter 6 <pixel-data>` and :ref:`Chapter 7 <tables>` cover more details
about how to read, manipulate and write pixel data and tables, respectively.


If you are looking for a quick reference, please, have a look on the
:doc:`../cheatsheet`.

Reference Documents
===================

    - |DRAGONS|
    - :doc:`../cheatsheet`
    - |RSUserManual|
    - |RSProgManual|

What is |astrodata|?
====================

|astrodata| is a package that wraps together tools to represent internally
astronomical datasets stored on disks and to properly parse their metadata
using the |AstroData| and the |TagSet| classes. |astrodata| provides uniform
interfaces for working on datasets from different
instruments. Once a dataset has been opened with |astrodata|, the object
"knows about itself". Information like instrument, observation mode, and how
to access headers, is readily available through the uniform interface. All
the details are coded inside the class associated with the instrument, that
class then provides the interface. The appropriate class is selected
automatically when the file is opened and inspected by |astrodata|.

Currently |astrodata| implements a representation for Multi-Extension FITS
(MEF) files. (Other representations can be implemented.)


.. _install:

Installing Astrodata
====================

The |astrodata| package has a few dependencies, |astropy|, |numpy| and others.
The best way to get everything you need is to install Miniconda, and the
|dragons| stack from conda-forge and Gemini's public conda channel.

|astrodata| itself is part of |DRAGONS|. It is available from the
repository, as a tar file, or as a conda package. The bare |astrodata| package
does not do much by itself, it needs a companion instrument definitions
package. For Gemini, this is ``gemini_instruments``, also included in
|DRAGONS|.

.. note::  We are in the process of making ``astrodata`` an Astropy affiliated
        package.  For now, |DRAGONS| uses the ``astrodata`` integrated with
        DRAGONS not the affiliated package.

Installing Miniforge and the DRAGONS stack
------------------------------------------
This is required whether you are installing |DRAGONS| from the
repository, the tar file or the conda package.

To avoid duplication, please follow the installation guide provided in the
Recipe System User Manual:

  |RSUserInstall|


Smoke test the Astrodata installation
-------------------------------------
From the configured bash shell::

    $ type python
    python is hashed (<home_path>/anaconda3/envs/dragons/python)

    Make sure that python is indeed pointing to the Anaconda environment you
    have just set up.

::

    $ python
    >>> import astrodata
    >>> import gemini_instruments

    Expected result: Just a python prompt and no error messages.

Source code availability
------------------------
The source code is available on Github:

    `<https://github.com/GeminiDRSoftware/DRAGONS>`_

.. _datapkg:

Try it yourself
===============

**Try it yourself**

Download the data package if you wish to follow along and run the
examples presented in this manual.  It is available at:

    `<https://www.gemini.edu/sciops/data/software/datapkgs/ad_usermanual_datapkg-v1.tar>`_

Unpack it::

    $ cd <somewhere_convenient>
    $ tar xvf ad_usermanual_datapkg-v1.tar
    $ bunzip2 ad_usermanual/playdata/*.bz2

Then ::

    $ cd ad_usermanual/playground
    $ python


Astrodata Support
=================

Astrodata is developed and supported by staff at the Gemini Observatory.
Questions about the reduction of Gemini data should be directed to the
Gemini Helpdesk system at
`<https://noirlab.atlassian.net/servicedesk/customer/portal/12>`_
The github issue tracker can be used to report software bugs in DRAGONS
(`<https://github.com/GeminiDRSoftware/DRAGONS>`_).
