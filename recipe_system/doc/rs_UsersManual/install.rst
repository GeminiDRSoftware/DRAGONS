.. install:

.. include intro
.. include discuss

.. _install:

Installation
============
The Recipe System is distributed as part of DRAGONS.  DRAGONS is available
as a conda package.  The installation instructions below will install all
the necessary dependencies.

The use of the `bash` shell is required.  This is an Anaconda limitation.

Install Anaconda
----------------
If you already have Anaconnda installed, you can skip this step.  Go to
the ???KL ``Install geminiconda`` section below.

The first step is to get anaconda.  You can download it at:

    `<https://www.anaconda.com/download/>`_

Choose the version of Python that suits your other Python needs.  DRAGONS is
compatible with both Python 2.7 and 3.6.

If you have downloaded the graphical installer, follow the graphical installer
instructions.  Install in your home directory.  It should be the default.

If you have downloaded the command-line installer::

    /bin/bash -l
    chmod a+x Anaconda3-5.2.0-MacOSX-x86_64.sh
    ./Anaconda3-5.2.0-MacOSX-x86_64.sh

Of course, if you are already using ``bash`` you can forgo that line, and
adapt the Anaconda package installer name to what you downloaded.

Install geminiconda
-------------------
Anaconda requires the use of the bash shell.  ``tcsh`` or ``csh`` will not
work.  If you are using (t)csh, your first step is:

    /bin/bash -l

Make sure that ``~/anaconda/bin/activate`` is in your ``PATH``.  The easiest
way to ensure that is to add ``export PATH=~/anaconda/bin:$PATH`` to your
``.bash_profile``.  The Anaconda installer should have offered to add it for
you.

Activate anaconda::

    source ~/anaconda/bin/activate

Now add the Astroconda channel and the Gemini channel.  Those channels host
the necessary conda packages.

::

    conda config --add channels http://ssb.stsci.edu/astroconda
    conda config --add channels http://astroconda.gemini.edu/public

Then create the ``geminiconda`` environment and install the Gemini data
reduction software into it.

::

    Python 2.7:

    conda create -n geminiconda python=2.7 iraf-all pyraf-all stsci gemini

    Python 3.6

    conda create -n geminiconda python=3.6 iraf-all pyraf-all stsci gemini

If you are going to use PyRAF a regularly, we recommend installing Python 2.7
as PyRAF is very slow under Python 3.  Otherwise, install the Python 3
version.  If you know that you will not be using IRAF or PyRAF, you can
remove ``iraf-all`` and ``pyraf-all`` from command.

To use this environment, activate it::

    source activate geminiconda

You will need to activate the environment whenever you start a new shell.
If you are planning to use it all the time, you might want to add the
command to your ``.bash_profile``.

A note about IRAF.  If you have installed IRAF, you will need to configure it
if this is the first time.  With the ``geminiconda`` environment activated::

    cd ~
    mkdir iraf
    cd iraf
    mkiraf

At the ``mkiraf`` prompts choose ``xterm`` and re-initialize the ``uparm``
if asked.

IRAF is not needed for DRAGONS.  But since the Gemini IRAF suite is still
required for the reduction of some data, we add the information nonetheless.


.. _test:

Test the installation
---------------------

Start up the Python interpreter and import ``astrodata`` and the
``gemini_instruments`` packages::

   $ python
   >>> import astrodata
   >>> import gemini_instruments

If the imports are successful, i.e. no errors show up, exit Python (Ctrl-D).

Now test that ``reduce`` runs. There may be some delay as package modules
are compiled and loaded::

   $ reduce -h

or ::

   $ reduce --help

This will print the reduce help to the screen.

If you have Gemini fits files available, you can test that the Recipe System
is functioning as expected as follow::

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
