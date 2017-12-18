.. intro.rst

.. _intro:

************
Introduction
************

What is Astrodata?
==================
Astrodata is a tool to represent internally datasets stored on disks.
Astrodata provides uniform interfaces for working on datasets from different
instruments.  Once a dataset has been opened with Astrodata, the object
"knowns about itself".  Information like instrument, observation mode, and how
to access headers, is readily available through the uniform interfaces.  All
the details are coded inside the class associated with the instrument, that
class then provides the interfaces.  The appropriate class is selected
automatically when the file is opened and inspected by Astrodata.

Currently Astrodata implements a representation for Multi-Extension FITS (MEF)
files.  (Other representations can be added.)


.. _install:

Installing Astrodata
====================
The astrodata package has a few dependencies, ``astropy``, ``numpy`` and others.
  The best way to get everything you need is to install Anaconda, and the
``stsci`` and ``gemini`` stack from the AstroConda channel.

Astrodata itself is part of ``gemini_python``.  It is available from the
repository, as tar file, or as a conda package.  The bare ``astrodata`` package
does not do much by itself, it needs a companion instrument definitions
package.   For Gemini, this is ``gemini_instruments``, also included in
``gemini_python``.

Installing Anaconda and stacks from AstroConda
----------------------------------------------
This is required whether you are installing ``gemini_python`` from the
repository, the tar file or the conda package.

#. Install Anaconda.
    Go to https://www.continnum.io/downloads and install the latest 64-bit
    Anaconda, Python 2.7 or 3.x, it does not matter for the root installation.
    Since the Python world is moving away from 2.7, choosing 3.x is
    probably better.   The gemini_python software has not been fully tested
    under 3.x, so later we will set up an environment with 2.7.  It is expected
    that gemini_python will be confirmed Python 3 compatible in 2018.

#. Open a bash session.
    Anaconda requires bash.  If you are not familiar with bash, note that the
    shell configuration file is named ``.bash_profile``.  During the
    installation, a PATH setting has been added to your ``.bash_profile`` to
    add the Anaconda bin directory to the ``PATH``.

#. Activate Anaconda.
    Normal installation puts the software in ``~/anaconda/``.::

    $ source ~/anaconda/bin/activate

#. Configure the ``conda`` package manager to look in the AstroConda channel
    hosted by STScI.  This is a one-time step.  It affects current and future
    Anaconda installations belonging to the same user on the same machine.::

    $ conda config --add channels http:/ssb.stsci.edu/astroconda


#. Create an environment.
    To keep things clean, Anaconda offers virtual environments.  Each project
    can use it's own environment.  For example, if you do not want to modify
    the software packages needed for previous project, just create a new one
    for new project.

    Here we set up an environment where the ``gemini_python`` dependencies can
    be installed without affect the rest of the system when not using that
    virtual environement.  The new virtual environment is here named
    ``geminiconda``.
    ::

    $ conda create -n geminiconda python=2.7 stsci gemini

    If you are planning to use the ``recipe_system`` and Gemini data reduction
    pipeline, please note that there are still IRAF dependencies and you will
    need to install the IRAF-related conda packages.
    ::

    $ conda create -n geminiconda python=2.7 iraf-all pyraf-all stsci gemini

#. Activate your new virtual environment.
    ::

    $ source activate geminiconda


Conda installation (recommended)
--------------------------------
If the latest ``gemini_python`` package was present on the AstroConda channel
when you installed Anaconda and the gemini stack, then you are done.

To check for newer version::

    $ conda search gemini_python

    The * will show which version is installed if multiple are available.

To update to a newer version::

    $ conda update gemini_python


If gemini_python was not installed during the Anaconda and AstroConda stack
installation, install it::

    $ conda install gemini_python

Tarball installation
--------------------
If the latest ``gemini_python`` is not yet available as a conda package but
only has tarbal (`tar.gz`), it is still possible to install it in your
environment, or elsewhere if you do not want to mix things up.


(python setup.py install [--prefix=blah])  Waiting on response from James
(Slack-conda)


Using the latest software from the repository (expert)
------------------------------------------------------
The repository is available on github, on the Gemini Observatory Data
Reduction Software page, https://github.com/GeminiDRSoftware.   Either git
clone or download the content of ``gemini_python``.

Once you have the source code, remember to set your ``PYTHONPATH`` to include
the package's location.

Examples of shell configuration
-------------------------------

For bash users
++++++++++++++
Anaconda should have already added itself to the ``PATH`` during installation.
 If you want the ``geminiconda`` environment to load automatically, you can
 add ::

    source activate geminiconda

to your ``.bash_profile``.

For tcsh users
++++++++++++++
To use ``astrodata`` and ``gemini_python`` you will need to use ``bash``. If
you wish to continue using ``tcsh`` as your default, here are a few things you
can do to make, when you need it, the switch painless.

In your ``.cshrc``, add an alias to launch bash and source ``.bash_profile`` ::

    alias geminiconda "/bin/bash -l"

Then in your likely very bare ``.bash_profile``, add ::

    source activate geminiconda

The path to Anaconda should have already been set when you installed Anaconda,
something like ``export PATH="~/anaconda/bin:$PATH"``.

Working from tcsh, when you want to use ``astrodata``, type at the prompt ::

    > geminiconda

and the shell will switch to ``bash`` which will then automatically activate
the ``geminiconda`` environment.


Smoke test the Astrodata installation
-------------------------------------
From the configured bash shell::

    $ type python
    python is hashed (<home_path>/anaconda/envs/geminiconda/python)

    Make sure that python is indeed pointing to the Anaconda environment you
    have just set up.

::

    $ python
    >>> import astrodata
    >>> import gemini_instruments

    Expected result: Just a python prompt and no error messages.


Astrodata Support
=================
Astrodata is had not been officially released to the public yet.  It is an
internal project.  Gemini staff should contact members of the Science Users
Support Department.  Until public release, there is no external supports other
than for instrument builders.  Instrument teams should reach out to their
assigned Gemini contact person for data reduction.