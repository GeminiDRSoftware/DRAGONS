.. install.rst

.. |anaconda_link| raw:: html

    <a href="https://www.anaconda.com/distribution/#download-section" target="_blank">https://www.anaconda.com/distribution/#download-section</a>

.. |geminiiraf_link| raw:: html

   <a href="http://www.gemini.edu/node/11823" target="_blank">http://www.gemini.edu/node/11823</a>

.. _install:

************
Installation
************

The Recipe System is distributed as part of DRAGONS.  DRAGONS is available
as a conda package.  The installation instructions below will install all
the necessary dependencies.

The use of the ``bash`` shell is required by Anaconda.

Install Anaconda
================
If you already have Anaconda installed, you can skip this step and go to
the :ref:`Install DRAGONS <install_dragons>` section below.  If not, then your
first step is to get and install Anaconda.  You can download it at:

    |anaconda_link|

Choose the version of Python that suits your other Python needs.  DRAGONS is
compatible with both Python 2.7 and 3.6.  We recommend that you install the
Python 3 version of Anaconda, the specific Python version can be adjusted
later.

If you have downloaded the graphical installer, follow the graphical installer
instructions.  Install in your home directory.  It should be the default.

If you have downloaded the command-line installer, type the following in a
terminal, replacing the ``.sh`` file name to the name of the file you have
downloaded.  The ``/bin/bash -l`` line is not needed if you are already
using bash.  The command-line installer allows for more customization of the
installation.

::

    $ /bin/bash -l
    $ chmod a+x Anaconda3-2019.03-MacOSX-x86_64.sh
    $ ./Anaconda3-2019.03-MacOSX-x86_64.sh

(``$`` indicates the terminal prompt.)

.. note::  To prevent the Anaconda "base" environment from loading
   automatically, do::

   $ conda config --set auto_activate_base false


.. _install_dragons:

Install DRAGONS
===============

Anaconda requires the use of the bash shell.  ``tcsh`` or ``csh`` will not
work.  If you are using (t)csh, your first step is::

    $ /bin/bash -l

Make sure that ``~/anaconda3/bin/activate`` is in your ``PATH`` by doing::

    $ which activate

The Anaconda installer should have added conda configurations to the
``~/.bash_profile`` for you.  If ``activate`` is not found, try::

    $ source ~/.bash_profile

If ``activate`` is still not found, you might have to add
``export PATH=~/anaconda3/bin:$PATH`` to your ``~/.bash_profile`` using your
favorite text editor, and run the ``source`` command above again.

.. note:: Sometimes the Anaconda installer will install the software in
    ``~/anaconda2`` or ``~/anaconda3`` instead of simply ``~/anaconda``.  Just
    check in your home directory which one of the three possibilities was used.

The code Anaconda adds to the .bash_profile will automatically activate
anaconda.  To activate or deactivate Anaconda manually::

    $ conda activate
    $ conda deactivate

Now that Anaconda works, we add the needed astronomy software.  Add the
Astroconda channel and the Gemini channel.  Those channels host
the conda astronomy packages.

::

    $ conda config --add channels http://ssb.stsci.edu/astroconda
    $ conda config --add channels http://astroconda.gemini.edu/public

The next step is to create a virtual environment and install the DRAGONS
software and its dependencies in it.  The name of the environment can be
anything you like.  Here we use "dragons" as the name and we install
Python 3.6.

::

    $ conda create -n dragons python=3.6 stsci gemini

To use this environment, activate it::

    $ conda activate dragons

You will need to activate the environment whenever you start a new shell.
If you are planning to use it all the time, you might want to add the
command to your ``.bash_profile``, after the "conda init" block.

.. note::
    As a side note, if you are going to use PyRAF regularly, for example to
    reduce Gemini data not yet supported in DRAGONS, you should be installing
    Python 2.7 as well in a different environment, along with the ``iraf-all``
    and ``pyraf-all`` conda packages.  PyRAF is very slow under Python 3.

::

    $ conda create -n geminiconda python=2.7 iraf-all pyraf-all stsci gemini


DRAGONS and the Recipe System to not need IRAF, PyRAF or Python 2.7, though
all of DRAGONS is still compatible with Python 2.7.   See the Gemini
website for information on how to configure IRAF
(|geminiiraf_link|)


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

			--- reduce, v2.0.8 ---
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
