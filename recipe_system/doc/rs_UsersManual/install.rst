.. install:

.. include intro
.. include discuss

.. _install:

Installation
============

.. note:: While the original Recipe System, v1.0, was written in such a way as
   to prevent migration to Python 3.x, Recipe System v2.0 has been written with
   efforts to ensure compatibility with both Python 2.7.x and Python 3.x. Because
   of this, the Recipe System, and the larger *gemini_python* package, introduce a
   dependency on the ``future`` module (currently, v0.16.0). Users may need to
   install this package (see http://python-future.org).

.. todo:: This chapter will need updating with reference to Anaconda/astroconda, 
	  once package naming and org. is finalized.

In order to use ``reduce``, the ``Reduce`` class, and the Recipe System to 
process data, the **gemini_python** package must be installed properly. The 
*gemini_python* package provides all the components described herein, as well as
those components, such as ``astrodata``, referred to in the :ref:`Introduction <intro>`. 
In particular, though not the subject of this document, the *gemini_python* 
``astrodata`` package has several dependencies, such as ``astropy``, 
``scipy``, ``numpy``, and others.

All dependencies of ``gemini_python`` and ``astrodata`` are provided
by the Ureka package, and users are highly encouraged to install and use this
very useful package. It is an easy and, perhaps, best way to get everything you
need and then some. Ureka is available at http://ssb.stsci.edu/ureka/.

WARNING:  The Ureka installation script will not set up IRAF for you. You need
to do that yourself. Here's how::

   $ cd ~
   $ mkdir iraf
   $ cd iraf
   $ mkiraf
   -- creating a new uparm directory
   Terminal types: xgterm,xterm,gterm,vt640,vt100,etc.
   Enter terminal type: xgterm
   A new LOGIN.CL file has been created in the current directory.
   You may wish to review and edit this file to change the defaults.


Once a user has has retrieved the gemini_python package, available as a tarfile 
from the Gemini website (http://gemini.edu), and untarred only minor adjustments 
need to be made to the user environment in order to make astrodata importable and 
allow ``reduce`` to work properly.

.. _config:

Install
-------

Recommended Installation
------------------------

We recommend that users install the software in a location other than the standard 
python location for modules (the default ``site-packages``). This is recommended
because, in all likelihood, most users will not have write permission to the system
python ``site-packages`` directory. Here is how you install the software somewhere 
other than the system location::

   $ python setup.py install --prefix=/your/location

``/your/location`` must already exist.  This command will install executable
scripts in a ``bin`` subdirectory, the documentation in a ``share`` subdirectory,
and the modules in a ``lib/python2.7/site-packages`` subdirectory.  The modules
being installed are ``astrodata``, ``gemini_instruments``, ``geminidr``, 
``recipe_system``, and ``gempy``. In this manual, we will only use ``astrodata``.

Because you are not using the default location, you will need to add two paths to
your environment.  You might want to add the following to your .cshrc or
.bash_profile, or equivalent shell configuration script.

C shell(csh, tcsh)::

   setenv PATH /your/location/bin:${PATH}
   setenv PYTHONPATH /your/location/lib/python2.7/site-packages:${PYTHONPATH}

Bourne shells (sh, bash, ksh, ...) ::

   export PATH=/your/location/bin:${PATH}
   export PYTHONPATH=/your/location/lib/python2.7/site-packages:${PYTHONPATH}

If you added those lines to your shell configuration script, make sure your 
``source`` the file to activate the new setting.

For csh/tcsh::

   $ source ~/.cshrc
   $ rehash

For bash::

   $ source ~/.bash_profile

Installation under Ureka
++++++++++++++++++++++++

Assuming that you have installed Ureka and that you have write access to the Ureka
directory, this will install ``astrodata`` in the Ureka ``site-packages`` directory.

::

   $ python setup.py install

This will also add executables to the Ureka ``bin`` directory and documentation to
the Ureka ``share`` directory.

With this installation scheme, there is no need to add paths to your environment.
However, it is a lot more complicated to remove the Gemini software in case of
problems, or if you just want to clean it out after evaluation.

In tcsh, you will need to run ``rehash`` to pick the new executables written to
``bin``.

.. _test:

Test the installation
+++++++++++++++++++++

Start up the python interpreter and import astrodata and the gemini_instruments
packages::

   $ python
   >>> import astrodata
   >>> import gemini_instruments

Next, return to the command line and test that ``reduce`` runs. There may be some 
delay as package modules are compiled and loaded::

   $ reduce -h

or ::

   $ reduce --help

This will print the reduce help to the screen.

.. todo:: Update the following section for example "test_one". Currently,
   there is no defined recipe or primitive "test_one".

If you have Gemini fits files available, you can test that the Recipe System
is functioning as expected with a test recipe provided by the ``geminidr``
package::

  $ reduce --recipe test_one /path/to/gemini_data.fits

If all is well, you will see something like::

  Resetting logger for application: reduce
  Logging configured for application: reduce
                         --- reduce, v4890  ---
		Running under astrodata Version GP-X1
  All submitted files appear valid
  Starting Reduction on set #1 of 1

    Processing dataset(s):
	  gemini_data.fits

  ==============================================================================
  RECIPE: test_one
  ==============================================================================
   PRIMITIVE: showParameters
   -------------------------
   rtf = False
   suffix = '_scafaasled'
   otherTest = False
   logindent = 3
   logfile = 'reduce.log'
   reducecache = '.reducecache'
   storedcals = 'calibrations/storedcals'
   index = 1
   retrievedcals = 'calibrations/retrievedcals'
   cachedict = {'storedcals': 'calibrations/storedcals', 'retrievedcals': 
                'calibrations/retrievedcals', 'calibrations': 'calibrations', 
                'reducecache': '.reducecache'}
   loglevel = 'stdinfo'
   calurl_dict = {'CALMGR': 'http://fits/calmgr', 
                  'UPLOADPROCCAL': 'http://fits/upload_processed_cal', 
                  'QAMETRICURL': 'http://fits/qareport', 
                  'QAQUERYURL': 'http://fits/qaforgui', 
                  'LOCALCALMGR': 'http://localhost:%(httpport)d/calmgr/%(caltype)s'}
   logmode = 'standard'
   test = True
   writeInt = False
   calibrations = 'calibrations'
   .
  Wrote gemini_data.fits in output directory


  reduce completed successfully.

The URLs in the example above, i.e. ``http://fits/...`` are described in Sec. 
:ref:`fitsstore`, Chapter 5, Discussion.
