.. changes.rst

.. include:: symbols.txt

.. _changes:

***********
Change Logs
***********

4.1.0
=====

Static images such as illumination masks are now distributed as bzip2
files to reduce the size of the download.

Improvements
------------
**geminidr.core**

  * log-linear wavelength resampling is now supported

    Instead of the ``force_linear`` boolean parameter in the
    ```resampleToCommonFrame`` primitive, the ``output_wave_scale``
    parameter now accepts three values: ``linear``,  ``loglinear``,
    and ``reference``.  The first two force a linear or log-linear
    resampling along the wavelength axis, while ``reference`` maintains the
    wavelength sampling of the reference frame, but can only be used if
    ``trim_spectral=True`` since it is unsafe to extrapolate this solution.

  * Changes to ``stackFrames``

    The ``scale`` and ``zero`` parameters now work by a pairwise comparison
    of the overlap regions of the input frames and perform a least-squares
    minimization of the differences after applying an appropriate
    transformation, rather than simply scaling by the average value of the
    entire image (or ``statsec`` if provided).

    The previous behavior can be restored by setting the new parameter
    ``debug_old_normalization=True``.

  * Easier handling of incorrect solutions in the
    ``determineWavelengthSolution`` GUI

    If all identified lines are deleted in the GUI, the model will revert
    to the initial linear solution instead of maintaining the original (bad)
    solution.


Interface Modifications
-----------------------
**geminidr.core**

  * ``determineWavelengthSolution`` will now proceed even if no solution
    is found, leaving the initial linear solution in place.

  * The default parameters of ``fitTelluric`` have changed so as not to mask
    regions with significant intrinsic stellar absorption.

**geninidr.gnirs**

  * A non-linearity correction is now applied to GNIRS data taken at Gemini
    North with the original IR Detector Controller (between 2010 and summer
    2025). This follows the same form as the NIRI non-linearity correction.

  * Before fitting a smooth function in ``normalizeFlat``, the flat field is
    divided by a sawtooth pattern to remove the odd-even row effect seen in
    the data. This pattern is re-applied to the data after the normalization.


4.0.0
=====

This major release includes new support for near-infrared spectroscopic data.
Specifically, we are introducing support for GNIRS longslit data.

Many improvements and bug fixes have been included in this release.  Below
we list the most notable ones.

New Features
------------

Full support for the reduction of GNIRS longslit spectroscopy data.
  GNIRS longslit data reduction can now be performed in DRAGONS.  Full support
  from raw data to telluric and flux calibrated data is available. All GNIRS
  wavebands are supported, X, J, H, K, as well as L and M.  All three
  dispersers, 10 l/mm, 32 l/mm, and 111 l/mm are supported.

  The software offers algorithms and tools to help with the wavelength
  calibration.  Wavelength calibrations from arc lamp, OH and |O2| sky lines,
  and from telluric features are all supported.  The tutorial includes a
  guide to help you choose the best wavelength calibration method for your data.

  Algorithms and tools are includes to help with the measurment of the
  telluric model and the sensitivity function and then for the correction of
  the telluric features present in the data.



Improvements
------------
**geminidr.core**

  * Additional interpolation modes during resampling.

    Cubic and quintic polynomial interpolation are now available. The "order"
    parameter that was previously used to designate the order of spline
    interpolation has been replaced by a string parameter, "interpolant" that
    can take the value "nearest", "linear", "poly3", "poly5", "spline3", or
    "spline5".

  * Better ability to correct WCS

    ``standardizeWCS`` provides options for dealing with incorrect values in
    the FITS headers by constructing new WCS models from the telescope
    offsets and/or target and position angle information.  The option to
    control this is ``prepare:bad_wcs``


Interface Modifications
-----------------------

**geminidr.core**
**geminidr.ghost**

* Rename the ``order`` parameter to ``interpolant`` in the following primitives:

  * ``resampleToCommonFrame``
  * ``transferObjectMask``
  * ``distortionCorrect``
  * ``linearizeSpectra``
  * ``combineNodAndShuffleBeams``
  * ``mosaicDetectors``
  * ``shiftImages``
  * ``combineOrders``

**geminidr.core**

* Rename the ``threshold`` parameter in ``transferObjectMask`` to
  ``dq_threshold``, in line with other primitives.
* The ``force_linear`` boolean parameter of the spectroscopic
  ``resampleToCommonFrame`` primitive has been deprecated. Use
  ``output_wave_scale`` instead, with options ``linear`` and ``reference``
  corresponding to ``force_linear`` values of ``True`` and ``False``,
  respectively.
* The spectroscopic version of ``adjustWCSToReference`` now has an additional
  option, ``wcs``, which uses the absolute WCS information to align. This is
  equivalent to the old option "None", which was available as a fallback
  method. This is now the default fallback method, with "None" resulting in
  an exception if the primary method does not provide valid offsets.

**calibration database**

Any calibration database created with a version of DRAGONS prior to 4.0.0 will
not be compatible because v4.0 uses a new version of the archive code which
defines the underlying database schema (conda package ``fitsstorage``). You
will need to create a new database and ``caldb add`` your calibrations to it.


Bug fixes
---------

**geminidr.core**

* ``resampleToCommonFrame(trim_spectral=True)`` did not work as documented.
  It trimmed the spectral coverage to the extent of the reference (as
  ``trim_spatial`` works) instead of to the intersection of the spectral
  coverages of all inputs. This has been corrected.
* If not resampling the output spectrum, it is required to set
  ``trim_spectral=True`` to avoid errors from evaluating the wavelength
  solution and its inverse beyond its original limits.

----------------------------------------------------------

3.2.3
=====

Dependency Updates
------------------
**gempy.library**

* Ensure compatibility with SciPy v1.15 (matching.py)

Improvements
------------
**gempy**

* Improve interrupt handling and allow additional loggers to enhance the
  operation of GOATS.


3.2.2
=====

Bug Fixes
---------
**geminidr.ghost**

* Fix an issue where the GHOST reduction would fail if specific header
  values were not in the expected format.

Improvements
------------
* Reduce memory usage in ``flagCosmicRays`` and ``QECorrect`` primitives.

3.2.1
=====

Improvements
------------
**geminidr & gempy**

* Improved speed and success rate of wavelength calibration.

**geminidr.ghost**

* Added ``makeIRAFCompatible`` primitive to write order-combined GHOST
  spectra in a non-FITS-standard format that is compatible with IRAF.

Bug fixes
---------
**geminidr.ghost**

* Cause data validation to fail for echellograms without exactly 4 extensions.

* Fixed an occasional issue with bad pixels causing ``traceFibers`` to fail.

**geminidr.interactive**

* Fixed issues where certain values were not initialized correctly.

* Fixed stylesheet issues

New Features
------------

**gemini_instruments.gnirs**

* Preemptively added support for handling GNIRS data produced with the new
  detector controller software that will be installed in coming months.  An
  additional patch release will be issued once the gain, read noise, and other
  detector properties are known.

Documentation
-------------
* Several updates to the GHOST tutorials to fix errors and improve clarity.

Interface Modifications
-----------------------
**gemini_instruments.gnirs**

* New prims_motor_steps descriptor to support flat association with HR-IFU
and SciOps prism mechanism reproducibility workarounds adopted in Apr-2024.

**gemini_instruments.ghost**

* Change GHOST fast/low read mode from "fast" to "rapid".

3.2.0
=====

This release includes support for GHOST data reduction and the new CCDs
installed in GMOS-S in late 2023.

New Features
------------

Full support for the reduction of GHOST data.
  This is based on the external GHOSTDR package, with important improvements.
  Includes changes to the names and scope of primitives to better align with the
  other instrument recipes.

**Support for new GMOS-S CCDs installed in late 2023.**

Improvements
------------
**astrodata.wcs**

* Support for reading and writing log-linear wavelength axes to/from FITS.

* Support for reading and writing tabular wavelength information to/from FITS.

**astrodata.provenance**

* Renamed the ``PROVHISTORY`` table to ``HISTORY``, and changed wording in the
  code from "provenance history" to simply "history".

**astrodata.fits**

* Support reading ASCII tables when opening FITS files with astrodata

**geminidr.core**

* Creation of new ``skip_primitive`` parameter, e.g.,
  ``reduce -p skyCorrectFromSlit:skip_primitive=True`` which allows any
  primitive in a recipe to be skipped. Note that inconsiderate use of this
  may cause a recipe to crash because the inputs to the subsequent primitive
  in the recipe may be inappropriate.

* Creation of new ``write_outputs`` parameter, e.g.,
  ``reduce -p ADUToElectrons:write_outputs=True`` which will write to disk
  the outputs of the primitive.

* Allow input files to ``shiftImages`` to recognize tabs or multiple
  whitespaces as the delimiter

**geminidr.gsaoi**

* Modification to the `nostack` science recipe to not "store" the image but
  rather continue and detect sources in the images in anticipation of the likely
  stacking that will follow later.  The output images will have the
  `_sourcesDetected` suffix rather than the `_image` suffix.

**recipe_system.cal_service**

* Whitespace now allowed in directory paths (if quoted), e.g.,
  ``databases = "~/.my dragons/dragons.db"``


Bug fixes
---------
**geminidr.core**

* Set default ``calculateSensitivity.bandpass`` parameter to 0.001 nm to
  better handle pure spectra in flux density units.

* Allow ``display`` to handle non-standard extension names, which did not
  work as intended.

**geminidr.gmos**

* Fix the QE model selection for the GMOS-S EEV CDDs.

**recipe_system**

* Set the ``engineering`` flag to False for all data stored in the local
  calibration database, to ensure that it can be retrieved.

Compatibility
-------------
**geminidr.interactive**

* The interactive tools are now compatible with and require bokeh v3 and above.


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
