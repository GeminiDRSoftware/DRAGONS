.. intro:

.. _intro:

************
Introduction
************

What is AstroData?
==================
.. todo::
   Describe AstroData in a language that astronomer will comprehend.
   
.. note::
   For the TODO: use language a scientist will comprehend.  then either refer to the programmer's reference
   or add a section clearly advertised to programmers.  the idea is that we don't want to lose
   the scientists, but if some readers are more technically oriented we also want to make sure
   they get the info they are after.


.. _install:

Installing AstroData
====================

The ``astrodata`` package has several dependencies like ``numpy``, ``astropy``, and others.
The best way to get everything you need is to install Ureka, http://ssb.stsci.edu/ureka/.

Once this is done, install ``gemini_python``.  The ``astrodata`` package is currently
distributed as part of the ``gemini_python`` package.  The ``gemini_python`` package,
``gemini_python-X1.tar.gz``, can be obtained from the Gemini website.

Recommended installation
------------------------

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

   setenv PATH /your/favorite/location:${PATH}
   setenv PYTHONPATH /your/favorite/location:${PYTHONPATH}

For bash::

   export PATH=/your/favorite/location:${PATH}
   export PYTHONPATH=/your/favorite/location:${PYTHONPATH}

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

