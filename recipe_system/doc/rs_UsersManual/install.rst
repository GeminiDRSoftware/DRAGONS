.. install.rst

.. |anaconda_link| raw:: html

    <a href="https://www.anaconda.com/distribution/#download-section" target="_blank">https://www.anaconda.com/distribution/#download-section</a>

.. |miniforge_link| raw:: html

    <a href="https://github.com/conda-forge/miniforge" target="_blank">https://github.com/conda-forge/miniforge</a>

.. |miniforgelinux| raw:: html

    <a href="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" target="_blank">Miniforge3-Linux-x86_64.sh</a>

.. |miniforgemacosx| raw:: html

    <a href="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh" target="_blank">Miniforge3-MacOSX-x86_64.sh</a>

.. |geminiiraf_link| raw:: html

   <a href="https://www.gemini.edu/observing/phase-iii/reducing-data/gemini-iraf-data-reduction-software" target="_blank">https://www.gemini.edu/observing/phase-iii/reducing-data/gemini-iraf-data-reduction-software</a>

.. _install:

************
Installation
************

The Recipe System is the automation part of DRAGONS.  DRAGONS is available
as a conda package.  The installation instructions below will install all
the necessary dependencies.

The use of the ``bash`` shell is recommended.

Install Miniforge
=================

We recommend using Miniforge instead of Anaconda.  It is smaller and installs
contents from conda-forge which is now our main channel for python packages.
The overall size of the installation will therefore be much smaller than
a full Anaconda installation.  The instructions below are for Miniforge.

Anaconda will work just fine too if you prefer or have a business need for
it.  We just don't provide instructions for its installation.  Whether you use
Miniforge or Anaconda will not impact the DRAGONS installation as long as
your using the correct conda channels.

Download and install Miniforge
------------------------------
If you already have Miniforge installed (or Anaconda and don't want to move
to Miniforge), you can skip this step and go to the
:ref:`Install DRAGONS <install_dragons>` section below.

If not, then your first step is to get and install Miniforge.  You can download
it from the Miniforge github page.  The direct links to the Linux and Mac
installer are provided in the table below.

+--------------------------------------------------+
|  **Miniforge Page**: |miniforge_link|            |
+--------------------------------------------------+
|  **Linux x64_86 Installer**: |miniforgelinux|    |
+--------------------------------------------------+
|  **MacOSX x64_86 Installer**:  |miniforgemacosx| |
+--------------------------------------------------+


.. warning::  arm64 MacOS Users!!!  (That's M1/M2/M etc) DRAGONS is not yet
      built with the arm64 architecture. Some dependencies are also either not
      available for arm64 or not reliable.  The x86 build will work.  All you
      need to do is install the x64_86 version of Miniforge (or Anaconda).
      That way, the x64_86 binaries will automatically be seeked, and found.

      arm64 machines can run x86 binaries via the Rosetta interface.  It is
      seamless.

The current version of DRAGONS has been tested with Python 3.10.  At the time
of this writing, Miniforge installs Python 3.10 by default.  (Other version
of Python can subsequently be installed.)

To install, run the installer that you have downloaded.
Type the following in a terminal, replacing the ``.sh`` file name with the name
of the file you have downloaded.

::

    $ /bin/bash Miniforge3-MacOSX-x86_64.sh

(``$`` indicates the terminal prompt.)

.. note::  For the arm64 Macs, it will tell you that the architecture does not
           match.  That's okay, type `yes` to accept.

At ``"Do you wish to update your shell profile to automatically initialize conda?"``,
answer `no`.   The script sometimes put the "conda init" information in the
wrong shell file (observed on Mac).  To avoid confusion do the initialization
manually::

    $ ~/miniforge3/bin/conda init

.. note::  To prevent the "base" environment from loading automatically, do::

   $ conda config --set auto_activate_base false


Verify Miniforge installation
-----------------------------
Make sure that ``~/miniforge3/bin/conda`` is in your ``PATH`` by doing::

    $ which conda

It should show a path with ``miniforge3``, not ``anaconda``.

.. note:: If you had a previous installation of Anaconda, you might need to
          find the Anaconda's "conda initialize" block and comment it out.
          Look in files like .bash_profile, .bashrc, .zshrc.

The `conda init` command should have added conda configurations to the
``~/.bash_profile`` for you (or ``.bashrc``, ``.zshrc``).  If ``conda`` is not found,
try::

    $ source ~/.bash_profile

The code Miniforge adds to the ``.bash_profile`` will automatically activate
Miniforge.  To activate or deactivate Miniforge manually::

    $ conda activate
    $ conda deactivate


.. _install_dragons:

Install DRAGONS
===============
With Miniforge installed and ready to go, now we can install DRAGONS and
the necessary dependencies.

Add conda-forge and the Gemini channel.  Those channels host the conda packages
that we will need.

::

    $ conda config --add channels conda-forge
    $ conda config --add channels http://astroconda.gemini.edu/public

The content of the `~/.condarc` file should look like this (the order matters)::

   channels:
     - http://astroconda.gemini.edu/public
     - conda-forge

The next step is to create a virtual environment and install the DRAGONS
software and its dependencies in it.  The name of the environment can be
anything you like.  Here we use "dragons" as the name and we request
Python 3.10.

::

    $ conda create -n dragons python=3.10 numpy<2 dragons ds9

.. note:: DRAGONS is not currently compatible with the recent release of `numpy`
          version 2.  We're working on it.  In the meantime, ensure that
          `numpy` v1 is installed.

To use this environment, activate it::

    $ conda activate dragons

You will need to activate the environment whenever you start a new shell.
If you are planning to use it all the time, you might want to add the
command to your ``.bash_profile``, after the "conda init" block.

.. .. note::
    For Linux users only.

..    As a side note, if you are going to use PyRAF regularly, for example to
    reduce Gemini data not yet supported in DRAGONS, you should install the
    ``iraf-all`` and ``pyraf-all`` conda packages as well.

.. ..    $ conda create -n geminiconda python=3.10 iraf-all pyraf-all ds9 dragons

..    DRAGONS and the Recipe System do not need IRAF or PyRAF, however. See the
..    Gemini website for information on how to configure IRAF (|geminiiraf_link|)

.. _configure:

Configure DRAGONS
=================
DRAGONS requires a configuration file ``dragonsrc`` that is located in
``~/.dragons/``::

    $ cd ~
    $ mkdir .dragons
    $ cd .dragons
    $ touch dragonsrc

Open ``dragonsrc`` with your favorite editor and add these lines::

    [interactive]
    browser = safari

    [calibs]
    databases = ~/.dragons/dragons.db get

The browser can be set to any of "safari", "chrome", or "firefox", depending
on your preferences.  The path and name of the calibration database can be
anything, as long at the path exists.  The "get" means that DRAGONS will get
calibrations from that database.  The "store" option can be added after the
"get" to have DRAGONS automatically add new processed calibrations to the
database.  See any of the tutorials to learn how to use the calibration manager.

On a new installation, you will need to configure ``ds9`` buffer
configurations::

    $ cd ~
    $ cp $CONDA_PREFIX/lib/python3.10/site-packages/gempy/numdisplay/imtoolrc ~/.imtoolrc
    $ vi .bash_profile   # or use your favorite editor

      Add this line to the .bash_profile:
        export IMTOOLRC=~/.imtoolrc

It is recommended to increase the Operating System limit on the number of
opened files.  We have seen an increase in reports of the error
"Too many open files" when reducing spectroscopy data.

In your `.bash_profile`, add the following line to overcome that OS limitation::

    ulimit -n 1024



.. _test:

Test the installation
=====================

Start up the Python interpreter and import ``astrodata`` and the
``gemini_instruments`` packages::

    $ python
    >>> import astrodata
    >>> import gemini_instruments

If the imports are successful, i.e. no errors show up, exit Python (Ctrl-D).

Now test that ``reduce`` runs. There may be some delay as package modules
are compiled and loaded::

    $ reduce --help

This will print the ``reduce`` help to the screen.

If you have Gemini FITS files available, you can test that DRAGONS
is functioning as expected as follow (replace the file name with the name
of your file)::

    $ reduce N20180106S0700.fits -r prepare

If all is well, you will see something like::

			--- reduce, v3.1.0 ---
    All submitted files appear valid
    Found 'prepare' as a primitive.
    ================================================================================
    RECIPE: prepare
    ================================================================================
    PRIMITIVE: prepare
    ------------------
      PRIMITIVE: validateData
      -----------------------
      .
      PRIMITIVE: standardizeStructure
      -------------------------------
      .
      PRIMITIVE: standardizeHeaders
      -----------------------------
         PRIMITIVE: standardizeObservatoryHeaders
         ----------------------------------------
         Updating keywords that are common to all Gemini data
         .
         PRIMITIVE: standardizeInstrumentHeaders
         ---------------------------------------
         Updating keywords that are specific to NIRI
         .
      .
    .
    Wrote N20180106S0700_prepared.fits in output directory

    reduce completed successfully.


