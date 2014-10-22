.. userenv:

User Environment
================

Once a user has has retrieved the gemini_python package, available as a tarfile 
from the Gemini website (http://gemini.edu), and untarred only minor adjustments need 
to be made to the user environment in order to make astrodata importable and 
allow ``reduce`` to work properly.

.. _config:

Installation
------------
Download the gemini_python X1 distribution (tar archive), place the tarfile as 
desired, and extract the archive::

  $ tar -xvf gemini_python_X1.tar.gz

Next, invoke the usual Distutils command for a standard python module installation::

  python setup.py install --prefix=/somewhere/

This will place executables in ``/somewhere/bin`` and the package modules in 
``/somewhere/lib/python2.7/site-packages/``.  

Users will then need to have ``/somewhere/bin`` in $PATH and
``/somewhere/lib/python2.7/site-packages`` in either $PYTHONPATH or add the 
site-packages to sys.path.

``reduce`` is made available on the command line once the installation is complete. 

Test the installation
---------------------

Start up the python interpreter and import astrodata::

   $ python
   >>> import astrodata

Next, return to the command line and test that ``reduce`` is reachable 
and runs. There may some delay as package modules are byte compiled::

   $ reduce -h [--help]

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
  Shutting down proxy servers ...
  ADCC is running externally. No proxies to close
  reduce exited     on status: 0

Exit status 0 indicates nominal operations.
