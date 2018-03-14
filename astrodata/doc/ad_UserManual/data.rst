.. data.rst

.. _data:

**********
Pixel Data
**********

**Try it yourself**

If you wish to follow along and try the commands yourself, download the
data package ``dragons_datapkg-v1.0``, go to the ``playground`` directory
and launch python.

::

    $ cd <path>/dragons_datapkg-v1.0/playground
    $ python

Then import core astrodata and the Gemini astrodata configurations. ::

    >>> import astrodata
    >>> import gemini_instruments


Operate on Pixel Data
=====================
The pixel data are stored in the ``AstroData`` object as a list of
``NDAstroData`` objects.  The ``NDAstroData`` is subclass of Astropy's
``NDData`` class which combines in one "package" the pixel values, the
variance, and the data quality plane or mask.   The data can be retrieved
as standard NumPy ``ndarray``.

In the sections below, we will present several typical examples of data
manipulation.  But first let's start with a quick example on how to access
the pixel data. ::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')

    >>> the_data = ad[1].data
    >>> type(the_data)
    <type 'numpy.ndarray'>

    >>> # Loop through the extensions
    >>> for ext in ad:
    ...     the_data = ext.data
    ...     print the_data.sum()
    ...

In this example, we first access the pixels for the second extensions.
Remember that in Python, list are zero-indexed, hence we access the second
extension as ``ad[1]``.   The ``.data`` attribute contains a numpy ``ndarray``.
In the for-loop, for each extension, we get the data and use the numpy
``.sum()`` method to sum the pixel values.   Anything that can be done
with a ``ndarray`` can be done on ``AstroData`` pixel data.


Arithmetic on AstroData Objects
===============================
``AstroData`` objects support basic in-place arithmetics with these methods:

+----------------+-------------+
| addition       | .add()      |
+----------------+-------------+
| subtraction    | .subtract() |
+----------------+-------------+
| multiplication | .multiply() |
+----------------+-------------+
| division       | .divide()   |
+----------------+-------------+

Normal, not in-place, arithmetics is also possible using the standard
operators, ``+``, ``-``, ``*``, and ``/``.

The big advantage of using ``AstroData`` to do arithmetics is that the
variance and mask, if present, will be propagated through to the output
``AstroData`` object.  We will explore the variance propagation in the next
section and mask usage later in this chapter.

Simple operations
-----------------
Here are a few examples of arithmetics on ``AstroData`` objects.::

    >>> ad = astrodata.open('../playdata/N20170521S0925_forStack.fits')

    >>> # Addition
    >>> ad.add(50.)
    >>> ad = ad + 50.
    >>> ad += 50.

    >>> # Subtraction
    >>> ad.subtract(50.)
    >>> ad = ad - 50.
    >>> ad -= 50.

    >>> # Multiplication (Using a descriptor)
    >>> ad.multiply(ad.exposure_time())
    >>> ad = ad * ad.exposure_time()
    >>> ad *= ad.exposure_time()

    >>> # Division (Using a descriptor)
    >>> ad.divide(ad.exposure_time())
    >>> ad = ad / ad.exposure_time()
    >>> ad /= ad.exposure_time()

When the syntax ``adout = adin + 1`` is used, the output variable is a copy
of the original.  In the examples above we reassign the result back onto the
original.  The two other forms, ``ad.add()`` and ``ad +=`` are in-place
operations.

When a descriptor returns a list because the value changes for each
extension, a for-loop is needed::

    >>> for (ext, gain) in zip(ad, ad.gain()):
    ...     ext.multiply(gain)
    ...

If you want to do the above but on a new object, leaving the original unchanged,
use ``deepcopy`` first. ::

    >>> from copy import deepcopy
    >>> adcopy = deepcopy(ad)
    >>> for (ext, gain) in zip(adcopy, adcopy.gain()):
    ...     ext.multiply(gain)
    ...


Operator Precedence
-------------------
The ``AstroData`` arithmetics methods can be stringed together but beware that
there is no operator precedence when that is done.  For arithmetics that
involve more than one operation, it is probably safer to use the normal
Python operator syntax.  Here is a little example to illustrate the difference.

::

    >>> ad.add(5).mult(10).sub(5)

    >>> # means:  ad = ((ad + 5) * 10) - 5
    >>> # NOT: ad = ad + (5 * 10) - 5

This is because the methods modify the object in-place, one operation after
the other from left to right.  This also means that the original is modified.

This example applies the expected operator precedence::

    >>> ad = ad + ad * 3 - 40.
    >>> # means: ad = ad + (ad * 3) - 40.

If you need a copy, leaving the original untouched, which is sometimes useful
you can use ``deepcopy`` or just use the normal operator and assign to a new
variable.::

    >>> adnew = ad + ad * 3 - 40.


Variance
========
When doing arithmetic on an ``AstroData`` object, if a variance is present
it will be propagated appropriately to the output no matter which syntax
you use (the methods or the Python operators).

Adding a Variance Plane
-----------------------
In this example, we will add the poisson noise to an ``AstroData`` dataset.
The data is still in ADU, therefore the poisson noise as variance is
``signal / gain``.   We want to set the variance for each of the pixel
extensions.

::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')

    >>> for (extension, gain) in zip(ad, ad.gain()):
    ...    extension.variance = extension.data / gain
    ...

Check ``ad.info()``, you will see a variance plane for each of the four
extensions.

Automatic Variance Propagation
------------------------------
As mentioned before, if present, the variance plane will be propagated to the
resulting ``AstroData`` object when doing arithmetics.  The variance
calculation assumes that the data are not correlated.

Let's look into an example.

::

    #     output = x * x
    # var_output = var * x^2 + var * x^2
    >>> ad = astrodata.open('../playdata/N20170609S0154_varAdded.fits')

    >>> ad[1].data[50,50]
    56.160931
    >>> ad[1].variance[50,50]
    96.356529
    >>> adout = ad * ad
    >>> adout[1].data[50,50]
    3154.05
    >>> adout[1].variance[50,50]
    607826.62


Display
=======

Displaying
----------

Retrieving cursor position
--------------------------

Useful tools from the Numpy and SciPy Modules
=============================================

ndarray
-------

Simple Numpy Statistics
-----------------------

Clipped Statistics
------------------

Filters with SciPy
------------------

Many other tools
----------------

Using the Astrodata Data Quality Plane
======================================

.. todo::
   Write examples that use the mask.  Eg. put mask plan in numpy mask and
   do statistics.

Manipulate Data Sections
========================

Basic Statistics on Section
---------------------------

Example - Overscan Subtraction with Trimming
--------------------------------------------

Data Cubes
==========

Plot Data
=========
