.. tutorial:

.. _tutorial:

********
Tutorial
********

(write the points below into real text)

* this is just a quick intro.  you will be referred to sections within this manual for
the more detailed explanation and more information.
* the intended audience are new users of astrodata, but it can obviously serve as a 
quick refresher for standard operations.
* the data used in this tutorial and throughout the manual is packaged in 
``gemini_python_datapkg-X1.tar.gz``. The archive file can be downloaded from the Gemini
website.
* in the data package, the data used in this manual is stored in the directory
``data_for_ad_user_manual``.
* to run this tutorial, you will need to install ``gemini_python``.  See the 
:ref:`Installing Astrodata <install>` section in the :ref:`Introduction` <intro>`.


Open and Access MEF Files
=========================

* for the first few sections we will use the interactive Python shell
* cd to playground
* paths are assuming you are in playground and getting data from data_for_ad_user_manual
* start python


.. highlight:: python
   :linenothreshold: 1

::

   >>> from astrodata import AstroData
   >>> ad = AstroData('....fits')


Display a MEF File
==================

Operate on MEF Files
====================

Create and Update MEF Files
===========================

Writing a Python Function using AstroData
=========================================



.. todo::
   Create a tutorial that shows how to do typical things an astronomer might want do.

