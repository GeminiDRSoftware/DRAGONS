.. include discuss

.. _intro:

************
Introduction
************

This document is version 2.0 (beta) of the Recipe System Users Manual.
Primarily, this document presents detailed information and discussion
regarding the Recipe System command line interface, ``reduce``, and the
programmatic interface on the underlying class, ``Reduce``.

Subsequent to this introduction, and in order to give users/readers a general
sense of the Recipe Systmem, this document provides a brief, high-level
overview of the Recipe System (Chapter 2, :ref:`overview`), which is then
followed by a detailed presentation of the above mentioned interfaces in
Chapter 3, :ref:`interfaces`.

It is expected that most users will be exposed to the Recipe System through
the ``reduce`` command line tool. The ``reduce`` application allows users to
invoke the Gemini Recipe System from the command line to perform complex data
processing and reduction on one or more astronomical datasets with a minimal
set of parameters when default processing is requested. As this document details,
``reduce`` provides a number of options and command line switches that allow
users to control the processing of their data.

This document will further describe usage of the Recipe System's application
programming interface (API). Details and information about the ``astrodata``
package, the Recipe System, and/or the data processing involved in data
reduction are beyond the scope of this document and will only be engaged when
directly pertinent to the operations of the Recipe System.

Installation
============

.. note:: While the original Recipe System, v1.0, was written in such a way as
   to prevent migration to Python 3.x, Recipe System v2.0 has been written with
   efforts to ensure compatibility with both Python 2.7.x and Python 3.x. Because
   of this, the Recipe System, and the larger gemini_python package introduce a
   dependency on the ``future`` module (currently, v0.16.0). Users may need to
   install this package (see http://python-future.org).

The ``astrodata`` package has several dependencies like ``numpy``, ``astropy``,
and others.

.. todo:: The following section will need updating with reference to
   Anaconda/astroconda, once package naming and org. is finalized.

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
++++++++++++++++++++++++

It is recommended to install the software in a location other than the standard 
python location for modules (the default ``site-packages``). This is also the 
only solution if you do not have write permission to the default ``site-packages``. 
Here is how you install the software somewhere other than the default location::

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
WARNING: While easier to install and configure, this will modify your Ureka
installation. ::

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
---------------------

Start up the python interpreter and import astrodata and the gemini_instruments
packages::

   $ python
   >>> import astrodata
   >>> import gemini_instruments

Next, return to the command line and test that ``reduce`` is reachable 
and runs. There may be some delay as package modules are byte compiled::

   $ reduce -h

or ::

   $ reduce --help

This will print the reduce help to the screen.

.. todo:: Update the following section for example "test_one". Currently,
   there is no defined recipe or primitive "test_one".

If users have Gemini fits files available, they can test that the Recipe System
is functioning as expected with a test recipe provided by the astrodata_Gemini
package::

  $ reduce --recipe test_one /path/to/gemini_data.fits

If all is well, users will see something like::

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

Users curious about the URLs in the example above, i.e. ``http://fits/...``, see
Sec. :ref:`fitsstore` in Chapter 5, Discussion.
