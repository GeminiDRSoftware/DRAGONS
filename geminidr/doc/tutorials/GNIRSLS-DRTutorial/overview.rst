.. overview.rst

.. _overview:

********
Overview
********

This is a tutorial for the reduction of GNIRS longslit spectroscopic data
with DRAGONS.

GNIRS is a near-IR spectrograph offering longslit spectroscopy, cross-dispersed
spectroscopy, and intergral field spectroscopy (aka IFU).  This tutorial
focuses on the longslit spectroscopy.

Note that the acquisition keyhole can be used as an imager when no other
near-IR imager is available on the telescope and the science is time sensitive.
For a tutorial on the reduction of GNIRS keyhole imaging data, see
|GNIRSImgTut|.

In this tutorial, you will be able to run the examples yourself and experiment
with the process.  The tutorial comes with a downloadable data package that
contains all the data you need to run the examples presented.  Instructions
on where to get that package and how to set things up are given in
:ref:`datasetup`.

The GNIRS longslit tutorial series contains at time a couple scientific
examples covering non-thermal and thermal bands.

The reduction can be done in two different ways:

* From the terminal using the command line.
* From Python using the DRAGONS classes and functions.

We show how to run the same reduction using both methods.

The tutorials are:

* :ref:`K-band 32 l/mm grating <gnirsls_Kband32mm_example>`
* :ref:`J-band 111 l/mm grating <gnirsls_Jband111mm_example>`
* :ref:`L-band 10 l/mm grating <gnirsls_Lband10mm_example>`
* :ref:`K-band beyond 2.3 microns 111 l/mm grating <gnirsls_Kband11mm_red_example>`

See the |RSUserInstall| to install the softwae if you have not already.