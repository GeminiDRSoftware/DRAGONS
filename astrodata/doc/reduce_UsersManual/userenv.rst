.. userenv:

User Environment
================

Once `astrodata has been installed <http://gdpsg.wikis-internal.gemini.edu/index.php/InstallAstroData>`_, 
only minor adjustments need to be made to the user environment in order to make 
astrodata importable and allow ``reduce`` to work properly. Presumably, the user 
has retrieved the gemini_python package from the SVN repository describe in the link
above. 

.. _config:

Configuration
-------------

Let us start with checkout of the gemini_python trunk::

   $ svn co http://chara.hi.gemini.edu/svn/DRSoftware/gemini_python/trunk

Users need not leave the checkout name `trunk` in place and obviously can rename
it to be whatever they like. Whatever that path, add this path to the PYTHONPATH 
environment variable.

For example, ``gemini_python/trunk`` is checked out as described above.
In '.' the directory `trunk` is now present and populated::

   $ ls -l
   drwxr-xr-x  19 user  group     646 Aug 20 14:44 trunk/

Rename trunk to some friendlier name::

   $ mv trunk gemsoft
   $ ls -l
   drwxr-xr-x  19 user  group     646 Aug 20 14:44 gemsoft/

Set the environment to make astrodata `et al` importable::

   $ export PYTHONPATH=${PYTHONPATH}:/user/path/to/gemsoft

Add the path `astrodata/scripts` to the PATH environment variable::

   $ export PATH=${PATH}:/user/path/to/gemsoft/astrodata/scripts

`reduce` is made available on the command line from here. It should already
have the executable bit set from the repository, but if not, chmod ``reduce`` 
to an appropriate mask.

Test the installation
---------------------

Start up the python interpreter and import astrodata::

   $ python
   >>> import astrodata

Next, return to the command line and test that ``reduce`` is reachable 
and runs::

   $ reduce -h [--help]

This will print the reduce help to the screen.
