.. changes.rst

.. _changes:

***********
Change Logs
***********

2.1.2
=====

Bug Fixes
---------

**gempy.gemini**

* Ensure NIRI skyflats satisfy calibration association requirements

New Features
------------

**geminidr.core**

* Add ``remove_first`` parameter to removeFirstFrame primitive

2.1.1
=====

Bug Fixes
---------

**geminidr.core**

* Fix a crash when a section was used when stacking.

**gempy scripts**

* Add missing third party adpkg and drpkg support to utility scripts dataselect, showpars, typewalk, and showrecipes.

**gempy.library**

* fix to Jacobian calculation for non-affine transforms 

**recipe_system.adcc**

* Make adcc more robust to missing connection to fitsstore.


Compatibility
-------------

**gempy.gemini**

* Add compatibility with sigma_clip for astropy v3.1+
* Add IRAF compatibility keywords on GMOS mosaiced data.
* Add compatibility with astroquery 0.4.

**geminidr.core**

* Add compatibility with sigma_clip fro astropy v3.1+ 
  
**geminidr.gmos**

* Add IRAF compatibility recipe.


Documentation
-------------

* Various fixes to documentation and instruction manual following feedback from users.