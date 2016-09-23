.. intro:

.. _intro:

************
Introduction
************

What is AstroData?
==================
The AstroData class is a tool to represent datasets stored in 
Multi-Extension FITS (MEF) files. It provides uniform interfaces for 
working on datasets from different instruments and their observational modes. 
Configuration packages are used to describe the specific data characteristics, 
layout, and to store type-specific implementations.  Once a MEF has been
opened with AstroData, the object it is assigned to knows essential information
about itself, like from which instrument this data comes from, how to access
the header information, etc.

Multi-extension FITS files are generalized as lists of header-data units 
(HDU), with key-value pairs populating headers, and pixel values populating 
data arrays. AstroData interprets a MEF as a single complex entity.  The 
individual "extensions" within the MEF are available with normal Python list 
("[]") syntax.

In order to identify types for the dataset and provide type-specific behavior, 
AstroData relies on configuration packages.  A configuration package (eg. 
``astrodata_Gemini``) contains definitions for all instruments and modes. A 
configuration package contains type definitions, meta-data functions, 
information lookup tables, and any other code or information needed to handle 
specific types of dataset.

.. _install:

Installing AstroData
====================

The ``astrodata`` package has several dependencies like ``numpy``, ``astropy``, and others.
The best way to get everything you need is to install Ureka, http://ssb.stsci.edu/ureka/.

Make that you launch Ureka after you have installed it.::

   $ ur_setup

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

Once this is done, install ``gemini_python``.  The ``astrodata`` package is currently
distributed as part of the ``gemini_python`` package.  The ``gemini_python`` package,
``gemini_python-X1.tar.gz``, can be obtained from the Gemini website:

  http://www.gemini.edu/sciops/data-and-results/processing-software

Recommended installation
------------------------

Note:  Before you do these steps, make sure Ureka has been launched.  A sure way
to check is to do::

   $ which python

Does it point to the Ureka version of Python?  If not, type ``ur_setup``.

It is recommended to install the software in a location other than the standard python
location for modules (the default ``site-packages``). This is also the only solution if 
you do not have write permission to the default ``site-packages``.  Here is how you 
install the software somewhere other than the default location::

   $ python setup.py install --prefix=/your/favorite/location

``/your/favorite/location`` must already exist.  This command will install executable
scripts in a ``bin`` subdirectory, the documentation in a ``share`` subdirectory,
and the modules in a ``lib/python2.7/site-packages`` subdirectory.  The modules being
installed are ``astrodata``, ``astrodata_FITS``, ``astrodata_Gemini``, and ``gempy``.
In this manual, we will not use ``gempy``.

Because you are not using the default location, you will need to add two paths to
your environment.  You might want to add the following to your .cshrc or .bash_profile,
or equivalent shell configuration script.

For tcsh::

   setenv PATH /your/favorite/location/bin:${PATH}
   setenv PYTHONPATH /your/favorite/location/lib/python2.7/site-packages:${PYTHONPATH}

For bash::

   export PATH=/your/favorite/location/bin:${PATH}
   export PYTHONPATH=/your/favorite/location/lib/python2.7/site-packages:${PYTHONPATH}

If you added those lines to your shell configuration script, make sure your ``source``
the file to activate the new setting.

For tcsh::

   $ source ~/.cshrc
   $ rehash

For bash::

   $ source ~/.bash_profile
   

Easier but more dangerous installation
--------------------------------------

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

In tcsh, you will need to run ``rehash`` to pick the new executables written to ``bin``.


Smoke test the installation
---------------------------

Just to make there is nothing obviously wrong with the installation and configuration,
we recommend that you run the following smoke tests::

   $ which typewalk
   
   Expected result: /your/favorite/location/bin/typewalk

::

   $ python
   >>> from astrodata import AstroData
   
   Expected result: Just a python prompt and no error messages.


AstroData Support
=================

This release of ``astrodata`` as part of ``gemini_python-X1`` is an early release of what 
we are working on.  It is not a fully supported product yet.  If you do have questions or 
feedback, please use the Gemini Helpdesk but keep in mind that the ticket will be addressed 
on a best-effort basis only.

