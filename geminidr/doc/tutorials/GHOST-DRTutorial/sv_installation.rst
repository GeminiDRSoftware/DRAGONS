.. sv_installation.rst

.. _sv_installation:

****************************************
Special Installation Instructions for SV
****************************************

Install Anaconda
================
You can skip to the next section if you already have Anaconda
(or miniconda, etc) installed.

You can use the instructions in Section 2.1 on the link below.
**The current GHOST software requires DRAGONS 3.0, so once you are done with
the Anaconda installation, come back here to complete the DRAGONS and GHOSTDR
installation.**

Anaconda installation instructions (Section 2.1):

   `<https://dragons.readthedocs.io/projects/recipe-system-users-manual/en/v3.1.0/install.html>`_


Install DRAGONS
===============
The current version of GHOSTDR requires DRAGONS v3.0.  It is **incompatible**
with the latest 3.1 release.

If you already have an environment set up with any version of DRAGONS 3.0,
you can skip this step and go to the next section.  However, note that you
will modify that environment to add GHOSTDR and a new GeminiCalMgr, and an
extra dependency.

It is recommended that you instead create a new GHOST-specific environment.

::

  conda config --add channels conda-forge
  conda config --add channels http://astroconda.gemini.edu/public
  conda config --remove channels http://ssb.stsci.edu/astroconda

  conda create -n ghost-sv python=3.7 dragons=3.0.4 ds9
  conda activate ghost-sv

Install GHOSTDR and Dependencies
================================
Here we install the development software needed to reduce GHOST data.
We do not have conda packages for it yet.

::

  # Install GHOSTDR
  pip install git+https://github.com/GeminiDRSoftware/GHOSTDR.git@v1.1.0

  # Install the GHOST-compatible calibration manager
  curl -O https://raw.githubusercontent.com/GeminiDRSoftware/GHOSTDR/master/.jenkins/local_calibration_manager/GeminiCalMgr-1.0.2-py3-none-any.whl
  pip install --force-reinstall GeminiCalMgr-1.0.2-py3-none-any.whl

  # Install an extra dependency
  pip install pysynphot

Additional Software
===================
Because the data in this tutorial were obtained during commissioning, they
are identified as "engineering" data.  DRAGONS refuses to use such data, as
a safeguard.  To use the data anyway, we need to modify the program ID and
make the data *look* non-engineering.  We have a script to do that.  We will
use it later.

It is unclear at this time if this will be applicable to the SV data.

::

  cd <path>/ghost_tutorial/
  curl -O https://raw.githubusercontent.com/GeminiDRSoftware/GHOSTDR/master/scripts/fixprogid.py


About the state of the software
===============================

This is very much still under-development software.  A lot of clean up remains to
be done, to the logs, to the file naming conventions, to the code itself.  It
will produce scientifically correct products, as long as the input data is
in the expected format.

This manual has only one example, one target in standard resolution.  Depending
on the data, you might need to make slight modifications, eg. if the flats and
the arcs are taken with different read mode.  However, when
it comes to reducing two-target observations, or high resolution observations,
the steps are expected be the same. The file structure of the output will be
different, with an extra source or just one with no sky, etc.

The final science exposures (and the standard files) are NOT stacked.  The
software was **delivered without stacking**.  So if you obtain three red
exposures to increase the signal-to-noise, you will need to stack the final
calibrated files yourself.  The DRAGONS team will be looking into adding
stacking as we do the clean-up.  The generic ``stackFrames`` primitive appears
to be doing a reasonable job for now.

The software has not yet been tested on very faint stars.  Since having
enough flux in the slit view image is critical for the extraction, very faint
source hardly visible on the slit view images might be a problem.  We have not
yet determine how faint we can go.
