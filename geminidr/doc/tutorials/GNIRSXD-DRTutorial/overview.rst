.. overview.rst

.. _overview:

********
Overview
********

This is a tutorial for the reduction of GNIRS cross-dispersed (XD)
spectroscopic data with DRAGONS.

GNIRS is a near-IR spectrograph offering longslit spectroscopy, cross-dispersed
spectroscopy, and intergral field spectroscopy (aka IFU).  This tutorial
focuses on the cross-dispersed spectroscopy.

For longslit spectroscopy, see |GNIRSLSTut|.

Note that the acquisition keyhole can be used as an imager when no other
near-IR imager is available on the telescope and the science is time sensitive.
For a tutorial on the reduction of GNIRS keyhole imaging data, see
|GNIRSImgTut|.

In this tutorial, you will be able to run the examples yourself and experiment
with the process.  The tutorial comes with downloadable data packages that
contains all the data you need to run the examples presented.  Instructions
on where to get that package and how to set things up are given in
:ref:`datasetup`.

The GNIRS cross-dispersed tutorial series contains two scientific
examples covering a Short Blue 32 l/mm configuration and a Short Blue
111 l/mm configuration with multiple central wavelengts observations.

The reduction can be done in two different ways:

* From the terminal using the command line.
* From Python using the DRAGONS classes and functions.

We show how to run the same reduction using both methods.

The tutorials are:

* :ref:`XD with Short-Blue + 32 l/mm grating  <gnirsxd_SXD32mm_example>`
* :ref:`XD with Short-Blue + 111 l/mm grating <gnirsxd_SXD111mm_example>`

See the |RSUserInstall| to install the software if you have not already.

.. tip:: If you are using the tutorials to guide you through the reduction
          of your own data and you encounter a problem, please review the
          :ref:`tips_and_tricks` and :ref:`issues_and_limitations` sections.
          They may contain the solution to your problem.

          Also, please use the :ref:`gnirsxd_wavecal_guide` to help you choose
          the best way to wavelength calibrate your data.


