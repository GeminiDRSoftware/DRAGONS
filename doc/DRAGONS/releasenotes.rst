.. releasenotes.rst

.. include:: symbols.txt

.. _releasenotes:

*************
Release Notes
*************

V4.1.0
======

This release includes new support for GNIRS cross-dispersed data.  Full
support from raw data to telluric and flux calibrated data is available. All
GNIRS cross-dispersed configurations are supported.   This release builds
upon the GNIRS longslit support introduced in DRAGONS V4.0.0.

Improvements to various algorithms, especially wavelength calibration, have
been made.  Various bug fixes and documentation improvements are also included.

V4.0.0
======

This major release includes new support for near-infrared spectroscopic data.
Specifically, we are introducing support for GNIRS longslit data.

GNIRS longslit data reduction can now be performed in DRAGONS.  Full support
from raw data to telluric and flux calibrated data is available. All GNIRS
wavebands are supported, X, J, H, K, as well as L and M.  All three
dispersers, 10 l/mm, 32 l/mm, and 111 l/mm are supported.

The software offers algorithms and tools to help with the wavelength
calibration.  Wavelength calibrations from arc lamp, OH and |O2| sky lines,
and from telluric features are all supported.  The tutorial includes a
guide to help you choose the best wavelength calibration method for your data.

Algorithms and tools are includes to help with the measurement of the
telluric model and the sensitivity function and then for the correction of
the telluric features present in the data.


V3.2.3
======

This is a bug fix release that addresses a change in API in SciPy v1.15. The
update also includes new code to in support of GOATS.

V3.2.2
======

This is bug fix release in support of GHOST.  Unplanned changes in
header values led to GHOST reduction to fail in some cases.  This release
offers a more resilient implementation.

We also include memory usage optimizations to the ``flagCosmicRays`` and
``QECorrect`` primitives.

V3.2.1
======

Not compatible with Numpy v2.  Add "numpy<2" when you create a new conda
environment.

This patch release includes improvements to speed up and increase the success
rate of wavelength calibration and includes important fixes to the GHOST
data reduction software and documentation.

Preemptive support for the new GNIRS detector controller has been included,
thought not yet tested, also the gain, read noise and other similar values are
known to be incorrect for the new detector controller in this release.  A patch
release will be issued once the new values are known.  GNIRS data not using the
new controllers are not impacted.

V3.2.0
======

This release adds:

* Fully integrated support for GHOST
* Support for the new GMOS-S CCDs that were installed in late 2023

The release also includes a number of other improvements and bug fixes.
See the :ref:`changes` for all the details.

With this release, DRAGONS offers support for:

Science Quality reduction
   * GMOS imager
   * NIRI imager
   * GSAOI imager
   * F2 imager
   * GMOS longslit spectrograph (including nod-and-shuffle)
   * GHOST spectrograph

For imaging and GMOS longslit spectroscopy, this software should be used
instead of the Gemini IRAF package.

To install DRAGONS, please follow the installation guide provided in the
Recipe System User Manual:

  |RSUserInstall|.



V3.1.0
======

This new release includes many new features, new support, various improvements
and new tutorials.  See the :ref:`changes` for all the details.

This update of DRAGONS has four big changes over V3.0:

* New science quality reduction for GMOS longslit data, including nod-and-shuffle.
* The alignment and stacking of GSAOI data is now done within DRAGONS and the]
  use of ``disco_stu`` is no longer required.
* The calibration service has gone through a major refactoring.  Of direct
  impact to the users, DRAGONS can now automatically store processed
  calibrations and the configuration file is now ``~/.dragons/dragonsrc``
* Static Bad pixel masks (BPMs) are now handled as calibrations and are
  distributed via the archive rather than being packaged with the software
  allowing for faster BPM-update response.

With this release, DRAGONS offers support for:

Science Quality reduction
   * GMOS imager
   * NIRI imager
   * GSAOI imager
   * F2 imager
   * GMOS longslit spectrograph (including nod-and-shuffle)

For imaging and GMOS longslit spectroscopy, this software should be used
instead of the Gemini IRAF package.

To install DRAGONS, please follow the installation guide provided in the
Recipe System User Manual:

  |RSUserInstall|.


V3.0.4
======
This patch release includes several small fixes and improvements, many
related to the Quality Assessment Pipeline run internally at Gemini.
Provenance for flux calibration is now included.  The patch is recommended
to all but not critical for most.

V3.0.2 and V3.0.3
=================
Note that 3.0.2 was found to have one broken recipe, 3.0.3 fixes it.

This patch release improves the reduction of GMOS-S data obtained since the
event on January 28, 2022 that led to the failure of amplifier 5.  This patch
also adds support of the new Flamingos 2 filters and the filter wheel
reshuffling that occurred earlier this year.  Various other fixes and features
are also contained in this patch.  See the :ref:`change logs <changes>` for
details.

V3.0.1
======

This is a patch release that fixes bugs related to the ``section`` parameter of some
primitives and the WCS of longlist spectra.  There has been a change in the ``findApertures``
interface to better optimize the automatic detection of the source apertures.  See the
:ref:`change logs <changes>` for details.

V3.0.0
======
This new release includes several new features, new support, and several bug
fixes.  See the :ref:`changes` for details.

This major update of DRAGONS has two big changes over V2:

* New "quicklook" reduction for GMOS longslit data
* Python 3 compatibilty only.  Python 2 is no longer supported.

With this release, DRAGONS offers support for:

Science Quality reduction
   * GMOS imager
   * NIRI imager
   * GSAOI imager
   * F2 imager

Quicklook Quality reduction
   * GMOS longslit spectrograph


For imaging, this software should be used instead of the Gemini IRAF package.

**For GMOS longslit spectroscopy, use this package only for quicklook
purposes.**  Please continue to use Gemini IRAF for science quality reductions.
We are working on a science quality package for GMOS longslit but it is not
ready yet.  We believe that releasing what we have for quicklook inspection
will nevertheless be useful to our users.

Installation instructions can be found in the Recipe System User Manual at:

 |RSUserShow|


