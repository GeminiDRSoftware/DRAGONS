.. install.rst

.. |anaconda_link| raw:: html

    <a href="https://www.anaconda.com/distribution/#download-section" target="_blank">https://www.anaconda.com/distribution/#download-section</a>

.. |geminiiraf_link| raw:: html

   <a href="https://www.gemini.edu/observing/phase-iii/reducing-data/gemini-iraf-data-reduction-software" target="_blank">https://www.gemini.edu/observing/phase-iii/reducing-data/gemini-iraf-data-reduction-software</a>

.. _install:

************
Installation
************

The Recipe System is distributed as part of DRAGONS.  DRAGONS is available
as a conda package.  The installation instructions below will install all
the necessary dependencies.

The use of the ``bash`` shell is recommended.

Install Anaconda
================

Download and install Anaconda
-----------------------------
If you already have Anaconda installed, you can skip this step and go to
the :ref:`Install DRAGONS <install_dragons>` section below.  If not, then your
first step is to get and install Anaconda.  You can download it from the
Anaconda website.  We recommend clicking on the "Get Additional Installers"
instead of using the green Download button as it will allow you to do a finer
selection.  Here we show how to use the "Command Line Installer"

+-----------------------------------------+
|  **Download Anaconda**: |anaconda_link| |
+-----------------------------------------+

.. warning::  M1 MacOS Users!!!  DRAGONS is not yet built with the M1
      architecture. The x86 build will work anyway.  But you have to be careful
      when you install Anaconda.

      We recommend that you use the "64-bit Command Line Installer", ie. the
      non-M1 version.  This version is the x86 (aka Intel) build. When using
      that version, by default, x86 binaries of the various conda packages will
      be installed.

      M1 can run x86 binaries via the Rosetta interface.  It is seamless.

The current version of DRAGONS has been tested with Python 3.9 and Python 3.10.
We recommend that you install the standard Python 3 version of Anaconda, the
specific Python version can be adjusted later, if necessary.

If you have downloaded the graphical installer, follow the graphical installer
instructions.  Install in your home directory.  It should be the default.

If you have downloaded the command-line installer (recommended), type the
following in a terminal, replacing the ``.sh`` file name to the name of the
file you have downloaded.  The ``/bin/bash -l`` line is not needed if you are
already using bash.  The command-line installer allows for more customization
of the installation.

::

    $ /bin/bash -l
    $ chmod a+x Anaconda3-2022.10-MacOSX-x86_64.sh
    $ ./Anaconda3-2022.10-MacOSX-x86_64.sh

(``$`` indicates the terminal prompt.)

.. note::  To prevent the Anaconda "base" environment from loading
   automatically, do::

   $ conda config --set auto_activate_base false


Verify Anaconda installation
----------------------------
We recommend the use of the ``bash`` shell::

    $ /bin/bash -l

Make sure that ``~/anaconda3/bin/conda`` is in your ``PATH`` by doing::

    $ which conda

The Anaconda installer should have added conda configurations to the
``~/.bash_profile`` for you.  If ``conda`` is not found, try::

    $ source ~/.bash_profile

If ``activate`` is still not found, you might have to add
``export PATH=~/anaconda3/bin:$PATH`` to your ``~/.bash_profile`` using your
favorite text editor, and run the ``source`` command above again.

.. note:: Sometimes the Anaconda installer will install the software in
    ``~/anaconda3`` instead of simply ``~/anaconda``.  Just
    check in your home directory which one of the tow possibilities was used.

The code Anaconda adds to the .bash_profile will automatically activate
anaconda.  To activate or deactivate Anaconda manually::

    $ conda activate
    $ conda deactivate


.. _install_dragons:

Install DRAGONS
===============
With Anaconda installed and ready to go, now we can install DRAGONS and
the necessary dependencies.

Add conda-forge and the Gemini channel.  Those channels host the conda packages
that we will need.

::

    $ conda config --add channels conda-forge
    $ conda config --add channels http://astroconda.gemini.edu/public
    $ conda config --set channel_priority disabled

The next step is to create a virtual environment and install the DRAGONS
software and its dependencies in it.  The name of the environment can be
anything you like.  Here we use "dragons" as the name and we install
Python 3.10.

::

    $ conda create -n dragons python=3.10 dragons ds9

To use this environment, activate it::

    $ conda activate dragons

You will need to activate the environment whenever you start a new shell.
If you are planning to use it all the time, you might want to add the
command to your ``.bash_profile``, after the "conda init" block.

.. note::
    For Linux users only.

    As a side note, if you are going to use PyRAF regularly, for example to
    reduce Gemini data not yet supported in DRAGONS, you should install the
    ``iraf-all`` and ``pyraf-all`` conda packages as well.

    $ conda create -n geminiconda python=3.10 iraf-all pyraf-all ds9 dragons

    DRAGONS and the Recipe System do not need IRAF or PyRAF, however. See the
    Gemini website for information on how to configure IRAF (|geminiiraf_link|)

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
database.  See any of the tutorials to see the calibration manager in action.

On a new installation, you will need to configure ``ds9`` buffer
configurations::

    $ cd ~
    $ cp $CONDA_PREFIX/lib/python3.10/site-packages/gempy/numdisplay/imtoolrc ~/.imtoolrc
    $ vi .bash_profile   # or use your favorite editor

      Add this line to the .bash_profile:
        export IMTOOLRC=~/.imtoolrc

While not specific at all to DRAGONS, it is recommended to increase the
Operating System limit on the number of opened files.  We have seen an increase
in reports of the error "Too many open files."

In your .bash_profile, add the following line to overcome that OS limitation::

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

This will print the reduce help to the screen.

If you have Gemini FITS files available, you can test that the Recipe System
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
