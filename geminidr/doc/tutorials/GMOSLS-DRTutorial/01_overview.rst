.. overview.rst

.. _overview:

********
Overview
********

This is a collection of tutorials for the reduction of GMOS longslit
spectroscopic data with DRAGONS.

.. warning::

   DRAGONS v3.0.0 is **NOT** approved for science quality reduction of
   GMOS Longslit data.  This version of DRAGONS should be used on GMOS
   Longslit data only for quicklook reduction and inspection.  Please continue
   to use the Gemini IRAF to produce your science quality products for GMOS
   Longslit while we work on providing that service on DRAGONS in a future
   release.

GMOS is an imager and spectrograph offering longslit spectroscopy,
multi-object spectroscopy (MOS), and integral field spectroscopy.  This
tutorial focuses on the longslit spectroscopy.   For a tutorial on the
reduction of GMOS imaging data, see `GMOS Imaging Data Reduction Tutorial <http://GMOSImg-DRTutorial.readthedocs.io/en/release-3.0.0>`_.

Here are tutorials that you, the reader, can run and experiment with.  This
document comes with a downloadable data package that contains all the data
you need to run the examples presented.  Instructions on where to get that
package and how to set things up are given in :ref:`datasetup`.

The GMOS longslit tutorial series for now contains only one scientific example,
the reduction of an observation with multiple sources in the field.  This can
be done in two different ways:

* From the terminal using the command line. (:ref:`Example 1-A <multisource_cmdline>`)
* From Python using the DRAGONS classes and functions. (:ref:`Example 1-B <multisource_api>`)

We plan to add additional examples in the future.

See the `DRAGONS Installation Instructions <https://dragons.readthedocs.io/projects/recipe-system-users-manual/en/release-3.0.0/install.html` to
install the software if you have not already.