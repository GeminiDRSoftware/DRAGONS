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

    - |DRAGONS|_
    - :doc:`../cheatsheet`
    - `Recipe System Users Manual`_
    - `Recipe System Programmers Manual`_

What is |astrodata|?
====================

|astrodata| is a package that wraps together tools to represent internally
astronomical datasets stored on disks and to properly parse their metadata
using the |AstroData| and the |TagSet| classes. |astrodata| provides uniform
interfaces for working on datasets from different
instruments. Once a dataset has been opened with |astrodata|, the object
"knowns about itself". Information like instrument, observation mode, and how
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
The best way to get everything you need is to install Anaconda_, and the
|dragons| stack from the AstroConda channel.

|astrodata| itself is part of |DRAGONS|. It is available from the
repository, as tar file, or as a conda package. The bare |astrodata| package
does not do much by itself, it needs a companion instrument definitions
package. For Gemini, this is ``gemini_instruments``, also included in
|DRAGONS|.

Installing Anaconda and the DRAGONS stack
-----------------------------------------
This is required whether you are installing |DRAGONS| from the
repository, the tar file or the conda package.

#. Install Anaconda.
    Go to https://www.anaconda.com/download/ and install the latest 64-bit
    Anaconda.

#. Open a bash session.
    Anaconda requires bash. If you are not familiar with bash, note that the
    shell configuration files are named ``.bash_profile`` and ``.bashrc``.
    During the installation, a PATH setting has been added to your
    ``.bash_profile`` or ``.bashrc`` to add the Anaconda bin directory to
    the ``PATH``.

#. Activate Anaconda.
    Normal Python 3 installation puts the software in ``~/anaconda3/``.::

    $ conda init

#. Set up conda channels.
    Configure the ``conda`` package manager to look in the AstroConda channel
    hosted by STScI, and in the GEMINI Conda Channel. This is a one-time step.
    It affects current and future Anaconda installations belonging to the same
    user on the same machine.::

    $ conda config --add channels http://ssb.stsci.edu/astroconda
    $ conda config --add channels http://astroconda.gemini.edu/public

#. Create an environment.
    To keep things clean, Anaconda offers virtual environments.  Each project
    can use its own environment.  For example, if you do not want to modify
    the software packages needed for previous project, just create a new
    environment for the new project.

    Here we set up an environment where the ``DRAGONS`` dependencies can
    be installed without affecting the rest of the system when not using that
    virtual environement.  The new virtual environment here is named
    ``dragons``.  The software has been tested with Python 3.7 hence we
    recommend that you use this specific version of Python with DRAGONS.
    ::

    $ conda create -n dragons python=3.7 dragons stsci


#. Activate your new virtual environment.
    ::

    $ conda activate dragons


#. Configure DRAGONS.
    These configurations are not stricktly required when using only |astrodata|.
    It however likely that if you are using |astrodata| you will be using
    DRAGONS too at some point.  So let's configure it to have it ready to go.

    DRAGONS requires a configuration file located in ``~/.geminidr/``.  The
    ``rsys.cfg`` file contains basic configuration for DRAGONS local calibration
    manager used by ``reduce``.

    ::

        $ cd ~
        $ mkdir .geminidr
        $ cd .geminidr
        $ touch rsys.cfg

    Open ``rsys.cfg`` with your favorite editor and add these lines::

        [calibs]
        standalone = True
        database_dir = ~/.geminidr/

    Next time you start a DRAGONS project, set the ``database_dir`` to a path
    of your liking, this is where the local calibration database will be written.

    Then configure buffers for ``ds9``::

        $ cd ~/
        $ cp $CONDA_PREFIX/lib/python3.7/site-packages/gempy/numdisplay/imtoolrc ~/.imtoolrc
        $ vi .bash_profile (or use your favority editor)
             Add this line to the .bash_profile:
                export IMTOOLRC=~/.imtoolrc


Update an existing DRAGONS installation
---------------------------------------

To check for newer version::

    $ conda search dragons

    The * will show which version is installed if multiple packages are available.

To update to the newest version::

    $ conda update dragons


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

    `<http://www.gemini.edu/sciops/data/software/datapkgs/ad_usermanual_datapkg-v1.tar>`_

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
Gemini Helpdesk system at ``https://www.gemini.edu/sciops/helpdesk/``
The github issue tracker can be used to report software bugs in DRAGONS.
