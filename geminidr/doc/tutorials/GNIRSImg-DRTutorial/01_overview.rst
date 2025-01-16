.. overview.rst

.. _overview:

********
Overview
********

This is a collection of tutorials for the reduction GNIRS keyhole imaging
data with DRAGONS.

GNIRS is a spectrograph.  It uses a keyhole to image a small area of the sky
and acquire targets.  The use of the GNIRS keyhole is not recommended for
imaging but it has been used in that capacity in the past when there were no
other options immediately available.   The quality of the data, and therefore
of the reduction is somewhat unpredictable.  The quality of the reduction
critically depends on the quality and the availability of the calibration
frames, the flat fields in particular.

In here are tutorials that you, the reader, can run and experiment with.  This
document comes with a downloadable data package that contains all the data
you need to run the examples presented.  Instructions on where to get that
package and how to set things up are given in :ref:`datasetup`.

Given the limited usefulness and general usage of this keyhole imaging mode,
we provide here one example, a dithered observation with two point sources.  In all cases, the reduction can be done
in two different ways:

* From the terminal using the command line.
* From Python using the DRAGONS classes and functions.

We show how to run the same reduction using both methods.

* :ref:`twostars_example`
