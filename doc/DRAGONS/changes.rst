.. changes.rst

.. _changes:

***********
Change Logs
***********

3.0.2
=====

Bug Fixes
---------

**geminidr.core**

* Fix bug where manually-assigned sky frames in ``separateSky`` were added
  to the list of science frames by mistake.

* In fringeCorrect, ``do_cal=force`` has been reactivated.

**geminidr.gnirs**q

* Added missing support for YPHOT filter.

Interface Modifications
-----------------------
* Internal Gemini catalog server URL updated.

Documentation
------------

* Various fixes to the documentation affecting formatting, not the content.


3.0.1
=====

Bug Fixes
---------

**geminidr.core**

* Fix bug where ``section`` start/end comparison was made on string, not
  numeric, values.

**gempy.library.transform**

* Fix bug that caused longslit spectra to have incorrect WCS, offset from true
  slit location.


Interface Modifications
-----------------------

**geminidr.core**

* Expose ``min_snr`` parameter in ``findApertures``, make ``use_snr=False``
  the default, and estimate noise from pixel-to-pixel variations, regardless
  of its value.

Documentation
-------------

* Various fixes to the documentation.


3.0.0
=====

This release includes new support for GMOS longslit data.  Reduction of
GMOS longslit data is offered only quicklook mode.  It does not produce
science quality outputs, yet.

Bug Fixes
---------

**geminidr**

* In imaging mode, the science recipes now include a call to
  ``scaleByExposureTime`` before the stacking step.  It is now possible to stack
  frames with different exposure times.

**gemini_instruments.gemini**

* Fix the GCALLAMP tag for NIR data to include the QH lamp.

**geminidr.core**

* Remove incorrect logging in separateSky when object and/or sky files are specified.
* Improve algorithm for separating on-source and on-sky frames.
* Avoid upsampling OBJMASK from uint8 to uint16
* In near-IR imaging mode, frames that fail to be sky subtracted are removed
  from the main reduction stream to avoid contamination.  The reduction continues
  with the "good" frames.  If all frames fail the sky subtraction, then all
  frames will be passed to the next step of the reduction.

**geminidr.gemini**

* Fix to the calculation of the CC-band used in nighttime sky quality assessment.
* Fix to the calculation of the BG-band used in nighttime sky quality assessment.

**gempy.gemini**

* Ensure NIRI skyflats satisfy calibration association requirements

**gempy.numdisplay**

* Fix a Python 3 compatibility issue.


New Features
------------

**geminidr**

* Quicklook (``--ql`` mode) reduction support for GMOS longslit data.

**geminidr.core**

* Add ``remove_first`` parameter to removeFirstFrame primitive.
* Add ``match_radius`` parameter to adjustWCSToReference primitive.
* Add an IRAF compatibility primitive and recipe for Flamingos 2.

**astrodata and recipe_system**

* Provenance history stored with the data in tables named: PROVENANCE and
  PROVHISTORY.


Interface Modifications
-----------------------

**geminidr.core**

* ``biasCorrect``, ``darkCorrect``, ``flatCorrect``.  The ``do_bias``,
  ``do_dark``, and ``do_flat`` input parameters have been replaced with
  ``do_cal`` with more options than True or False.  Use ``showpars`` to
  inspect the options.


Compatibility
-------------

* Python 2 support has been dropped.  Starting with v3.0.0, DRAGONS requires
  Python 3.   All tests were run on Python 3.7, and this version of Python
  now serves as the minimal required version.
* Improved the F2 processed products backward compatibility with Gemini IRAF.


Documentation
-------------

* Fix various links in the documentation.
* Add examples and cross-reference to disco-stu usage documentation.
* New tutorial for the **quicklook** reduction of GMOS longslit data.



2.1.1
=====

Bug Fixes
---------

**geminidr.core**

* Fix a crash when a section was used when stacking.

**gempy scripts**

* Add missing third party adpkg and drpkg support to utility scripts dataselect, showpars, typewalk, and showrecipes.

**gempy.library**

* Fix to Jacobian calculation for non-affine transforms

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