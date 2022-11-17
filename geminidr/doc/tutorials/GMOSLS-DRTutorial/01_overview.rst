.. overview.rst

.. _overview:

********
Overview
********

This is a tutorial for the reduction of GMOS longslit
spectroscopic data with DRAGONS.

GMOS is an imager and spectrograph offering longslit spectroscopy,
multi-object spectroscopy (MOS), and integral field spectroscopy.  This
tutorial focuses on the longslit spectroscopy.   For a tutorial on the
reduction of GMOS imaging data, see `GMOS Imaging Data Reduction Tutorial <http://GMOSImg-DRTutorial.readthedocs.io/en/v3.0.3>`_.

Here is a tutorial that you, the reader, can run and experiment with.  This
document comes with a downloadable data package that contains all the data
you need to run the example presented.  Instructions on where to get that
package and how to set things up are given in :ref:`datasetup`.

The GMOS longslit tutorial series for now contains one scientific example,
the reduction of an observation of a single stellar source with dither in both
wavelength and spatial direction.

The reduction can be done in two different ways:

* From the terminal using the command line.
* From Python using the DRAGONS classes and functions.

We show how to run the same reduction using both methods.

* Dithered point source
    * :ref:`dithered_cmdline`
    * :ref:`dithered_api`

More examples will be added in the future.

See the |RSUserInstall| to install the software if you have not already.