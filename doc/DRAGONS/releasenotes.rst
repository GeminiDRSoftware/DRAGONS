.. releasenotes.rst

.. _releasenotes:

*************
Release Notes
*************

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


