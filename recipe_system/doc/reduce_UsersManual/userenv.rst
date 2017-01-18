.. userenv:
.. include discuss

Installation
============
The ``astrodata`` package has several dependencies like ``numpy``, ``astropy``, 
and others. All dependencies of ``gemini_python`` and ``astrodata`` are provide by
the Ureka package, and users are highly encouraged to install and use this very
useful package. It is an easy and, perhaps, best way to get everything you need
and then some. Ureka is available at http://ssb.stsci.edu/ureka/.

WARNING:  The Ureka installation script will not set up IRAF for you. You need to do
that yourself. Here's how::

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

   $ python setup.py install --prefix=/your/favorite/location

``/your/favorite/location`` must already exist.  This command will install executable
scripts in a ``bin`` subdirectory, the documentation in a ``share`` subdirectory,
and the modules in a ``lib/python2.7/site-packages`` subdirectory.  The modules being
installed are ``astrodata``, ``astrodata_FITS``, ``astrodata_Gemini``, and ``gempy``.
In this manual, we will only use ``astrodata``.

Because you are not using the default location, you will need to add two paths to
your environment.  You might want to add the following to your .cshrc or 
.bash_profile, or equivalent shell configuration script.

C shell(csh, tcsh)::

   setenv PATH /your/favorite/location/bin:${PATH}
   setenv PYTHONPATH /your/favorite/location/lib/python2.7/site-packages:${PYTHONPATH}

Bourne shells (sh, bash, ksh, ...) ::

   export PATH=/your/favorite/location/bin:${PATH}
   export PYTHONPATH=/your/favorite/location/lib/python2.7/site-packages:${PYTHONPATH}

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

Start up the python interpreter and import astrodata::

   $ python
   >>> import astrodata

Next, return to the command line and test that ``reduce`` is reachable 
and runs. There may be some delay as package modules are byte compiled::

   $ reduce -h 

or ::

   $ reduce [--help]

This will print the reduce help to the screen.

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
