.. changes.rst

.. _changes:

***********
Change Logs
***********

3.2.0
=====

This release includes support for GHOST data reduction and the new CCDs
installed in GMOS-S in late 2023.

New Features
------------

Full support for the reduction of GHOST data.
  This is based on the external GHOSTDR package, with improvements and
  changes to the names and scope of primitives to better align with the
  other instrument recipes.

Improvements
------------
**astrodata.wcs**

* Support for reading and writing log-linear wavelength axes to/from FITS.

* Support for reading and writing tabular wavelength information to/from FITS.

**geminidr**

* Creation of new ``skip_primitive`` parameter, e.g.,
  ``reduce -p skyCorrectFromSlit:skip_primitive=True`` which allows any
  primitive in a recipe to be skipped. Note that inconsiderate use of this
  may cause a recipe to crash because the inputs to the subsequent primitive
  in the recipe may be inappropriate.

* Creation of new ``write_outputs`` parameter, e.g.,
  ``reduce -p ADUToElectrons:write_outputs=True`` which will write to disk
  the outputs of the primitive.


Bug fixes
---------
**geminidr**

* Set default ``calculateSensitivity.bandpass`` parameter to 0.001 nm to
  better handle pure spectra in flux density units.

* Allow ``display`` to handle non-standard extention names, which did not
  work as intended.


**recipe_system**

* Set the ``engineering`` flag to False for all data stored in the local
  calibration database, to ensure that it can be retrieved.


3.1.1
=====

Improvements
------------

**astrodata.provenance**

* Renamed the ``PROVHISTORY`` table to ``HISTORY``, and changed wording in the
code from "provenance history" to simply "history".

**geminidr.core**

* Allow input files to ``shiftImages`` to recognize tabs or multiple
  whitespaces as the delimiter

**recipe_system.cal_service**

* Whitespace now allowed in directory paths (if quoted), e.g.,
  ``databases = "~/.my dragons/dragons.db"``

Bug Fixes
---------

**geminidr.gmos**

* Fix the QE model selection for the GMOS-S EEV CDDs.


3.1.0
=====

This release includes new science-approved support for GMOS longslit data,
along with new interactive tools to optimize the spectroscopic reduction.
The calibration service has also been refactored with significant changes that
the user need to be aware of.  Please read on.

New Features
------------

Science quality support for GMOS longslit spectroscopy, including nod-and-shuffle.
  Please refer to the tutorial, |GMOSLSTut|.  DRAGONS is now the official
  software for reducing GMOS longslit data in normal and nod-and-shuffle
  mode.

New browser-base interactive tools to support spectroscopy.
  The following primitives have an interactive mode that can be activated with
  the ``-p interactive=True`` flag:

  * normalizeFlat
  * determineWavelengthSolution
  * skyCorrectFromSlit
  * findApertures
  * traceApertures
  * calculateSensitivity

The GSAOI alignment and stacking is now done in DRAGONS.
  The package ``disco_stu`` is no longer needed.  The default GSAOI recipe
  will align and stack.  See the tutorial,  |GSAOIImgTut|

The bad pixel masks are now handled as the other calibration files.
  They are distributed through the archive instead of with the package.  They
  are also fully integrated into the calibration service.  See the various
  tutorials for details.

The calibration service has been through a large refactor.
  It is now possible to have the processed calibrations stored automatically
  (was a user step before), and it possible to serially search more than one
  database. See below for details on the new configuration file,
  :ref:`interface_3.1`.  For usage examples, see the various tutorials.

New imaging recipes.
   For Flamingos-2, GSAOI, NIRI:
       ``ultradeep``  See |F2ImgTut| for an example.
   GMOS:
       ``reduceSeparateCCDs`` and ``reduceSeparateCCDCentral`` (See |GMOSImgTut|
       for an example.

.. _interface_3.1:

Interface Modifications
-----------------------
**recipe_system**

* There has been many changes to the calibration service.  Most of them are
  internal but the one big change for the users is the configuration file.
  The configuration file now ``~/.dragons/dragonrc`` (was
  ~/.geminidr/rsys.cfg).  The syntax inside the file has changed a bit too.

  * New ``[interactive]`` section.  This is used to set the browser that the
    interactive tools will use.  Valid browsers: "safari", "chrome", "firefox"::

       [interactive]
       browser = safari
  * New format for the ``[calib]`` section.  The variable is now named
    ``databases``, plural, and multiple databases can be defined to be searched
    serially.  One database per line.  The name of the database can now be
    set by the user instead of being hardcoded to ``cal_manager.db``.  Two
    new flags can be set ``get`` and ``store`` to, respectively, "get"
    processed calibrations for that database, and "store" them to it. ::

      [calib]
      databases = /Users/someone/data/myprogramcal.db get store
                  https://archive.gemini.edu get


**geminidr**

* You must now ensure that the bad pixel masks (BPMs) can be found in a
  database.  The BPMs are no longer distributed with the software.  They are
  downloadable from the archive.  See the "Tips and Tricks" section of any
  tutorial (except Flamingos-2).


Improvements
------------

**geminidr**

* As mentioned above, the BPMs are now stored in archive.  Using the archive to
  distribute the BPMs will allow us to make new BPMs available rapidly, for
  example, when new bad columns appear in GMOS CCDs, after a catastrophic event
  like the amplifier 5 failure in January 2022, or when the CCDs are replaced.

* Several new or improved algorithms compared to 3.0.x.

Documentation
-------------
**geminidr**

* There has been some restructuring of the tutorials to better present
  multiple examples within a tutorial.

* Several new examples for Flamingos-2 and GMOS imaging tutorials.

* Several science quality examples for the new GMOS longslit spectroscopy
  support.

**astrodata**

* The three previously separated ``astrodata`` manuals, "Cheat Sheet",
  "User Manual", and "Programmer Manual" have been consolidated into one
  master document.  Please fix your link, |ADMaster|.


Compatibility
-------------
* DRAGONS v3.1 is compatible with Python 3.7 to 3.10.  The pre-release tests
  were done with Python 3.10.  Please note that DRAGONS v3.1.x will the last
  minor version to support Python v3.7.

* The conda package was built and tested against conda-forge dependencies.
  STScI has dropped support of the astroconda channel.  Make sure that you
  adjust your conda channels.

  If you already have a ``~/.condarc`` file, make sure that the channels are
  set as follows::

    channels:
      - http://astroconda.gemini.edu/public
      - https://conda.anaconda.org/conda-forge
      - defaults

  If you are installing conda for the first time, see the installation
  instructions here:  |RSUserInstall|

3.0.4
=====

Bug Fixes
---------

**geminidr.gmos**

* Allow ``maskFaultyAmp`` to work on astrodata objects with no mask.

* Fix ``maskFaultyAmp`` to work on central stamp ROI.

**geminidr.core**

* Adjust minimal dither separation for fringe frame creation.

**astrodata**

* Fix AstroData ``info()`` method to handle extensions with no pixels.  Required
  for upcoming GHOST data.

Improvements
------------

**geminidr.gmos**

* Update to the GMOS-S Hamamatsu 4x4 imaging illumination mask.

**geminidr.core**

* Improve behavior of ``addIllumMaskToDQ`` to cope with larger shifts due to
  recent GMOS misalignment.

* Add provenance for the flux calibration step.

**gemini_instruments.f2**

* Switched to using WAVELENG for central_wavelength for F2 to be better aligned
  with the instrument and observatory software.

**gempy**

* In ``dataselect``, make the disperser selection default to the "pretty"
  mode rather than requiring the full component ID.

Quality Assessment Pipeline
---------------------------

* Increase robustness of measureIQ for 2D spectra.

* Interface improvements to the QAP Specviewer.

* Fix missing ``maskFaultyAmp`` in some QAP recipes.

* Limit the number of aperture/spectra selected in GMOS LS QA recipes for
  performance reasons.


3.0.2 and 3.0.3
===============

Note that 3.0.2 was found to have one broken recipe, 3.0.3 fixes it.

Bug Fixes
---------

**geminidr.core**

* Continue without crashing when ``traceApertures`` cannot identify a
  starting location for a trace.

* Fix issues with assignment of on-source/sky frames when the user specifies
  specific frames.

* Fix bug where ``stackFrames`` crashed if using the ``statsec`` parameter
  when scaling or zero-offsetting.

* In fringeCorrect, ``do_cal=force`` has been reactivated.

* Better handling of infinites and NaN in the flat normalization.

**geminidr.gmos**

* Added new primitive to the recipes to mask amplifier 5 in GMOS-S data
  obtained since January 28, 2022.  GMOS-S amplifier 5 suffered a major
  failure and it is not usable.

* Ensure that the masks are used when calculating the statistics in
  scaleByIntensity.

**geminidr.gnirs**

* Added missing support for YPHOT filter.

**geminidr.f2***

* Support of the Flamingos 2 filters.

New Features
------------

** geminidr **

* Add ``wave_units`` and ``data_units`` parameters to ``write1DSpectra`` to
  configure the output

* Under-the-hood modification to distinguish data reduced in quicklook mode
  versus science mode.

Interface Modifications
-----------------------
* Internal Gemini catalog server URL updated.

Documentation
-------------

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