.. data.rst

.. _pixel-data:

**********
Pixel Data
**********

**Try it yourself**

Download the data package (:ref:`datapkg`) if you wish to follow along and run the
examples.  Then ::

    $ cd <path>/ad_usermanual/playground
    $ python

Then import core astrodata and the Gemini astrodata configurations. ::

    >>> import astrodata
    >>> import gemini_instruments


Operate on Pixel Data
=====================
The pixel data are stored in the ``AstroData`` object as a list of
``NDAstroData`` objects.  The ``NDAstroData`` is a subclass of Astropy's
``NDData`` class which combines in one "package" the pixel values, the
variance, and the data quality plane or mask (as well as associated meta-data).
The data can be retrieved as a standard NumPy ``ndarray``.

In the sections below, we will present several typical examples of data
manipulation.  But first let's start with a quick example on how to access
the pixel data. ::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')

    >>> the_data = ad[1].data
    >>> type(the_data)
    <class 'numpy.ndarray'>

    >>> # Loop through the extensions
    >>> for ext in ad:
    ...     the_data = ext.data
    ...     print(the_data.sum())
    333071030
    335104458
    333170484
    333055206

In this example, we first access the pixels for the second extensions.
Remember that in Python, list are zero-indexed, hence we access the second
extension as ``ad[1]``.   The ``.data`` attribute contains a NumPy ``ndarray``.
In the for-loop, for each extension, we get the data and use the NumPy
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

If you want to do the above but on a new object, leaving the original unchanged,
use ``deepcopy`` first. ::

    >>> from copy import deepcopy
    >>> adcopy = deepcopy(ad)
    >>> for (ext, gain) in zip(adcopy, adcopy.gain()):
    ...     ext.multiply(gain)


Operator Precedence
-------------------
The ``AstroData`` arithmetics methods can be stringed together but beware that
there is no operator precedence when that is done.  For arithmetics that
involve more than one operation, it is probably safer to use the normal
Python operator syntax.  Here is a little example to illustrate the difference.

::

    >>> ad.add(5).multiply(10).subtract(5)

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

Check ``ad.info()``, you will see a variance plane for each of the four
extensions.

Automatic Variance Propagation
------------------------------
As mentioned before, if present, the variance plane will be propagated to the
resulting ``AstroData`` object when doing arithmetics.  The variance
calculation assumes that the data are not correlated.

Let's look into an example.

::

    >>> #     output = x * x
    >>> # var_output = var * x^2 + var * x^2
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

Data Quality Plane
==================
The NDData ``mask`` stores the data quality plane.  The simplest form is a
True/False array of the same size at the pixel array.  In Astrodata we favor
a bit array that allows for additional information about why the pixel is being
masked.   For example at Gemini here is our bit mapping for bad pixels.

+---------------+-------+
| Meaning       | Value |
+===============+=======+
| Bad pixel     |   1   |
+---------------+-------+
| Non Linear    |   2   |
+---------------+-------+
| Saturated     |   4   |
+---------------+-------+
| Cosmic Ray    |   8   |
+---------------+-------+
| No Data       |  16   |
+---------------+-------+
| Overlap       |  32   |
+---------------+-------+
| Unilluminated |  64   |
+---------------+-------+

(These definitions are located in ``geminidr.gemini.lookups.DQ_definitions``.)

So a pixel marked 10 in the mask, would be a "non-linear" "cosmic ray".  The
``AstroData`` masks are propagated with bitwise-OR operation.  For example,
let's say that we are stacking frames. A pixel is set as bad (value 1)
in one frame, saturated in another (value 4), and fine in all the other
the frames (value 0).  The mask of the resulting stack will be assigned
a value of 5 for that pixel.

These bitmasks will work like any other NumPy True/False mask.  There is a
usage example below using the mask.

The mask can be accessed as follow::

    >>> ad = astrodata.open('../playdata/N20170609S0154_varAdded.fits')
    >>> ad.info()

    >>> ad[2].mask


Display
=======
Since the data is stored in the ``AstroData`` object as a NumPy ``ndarray``
any tool that works on ``ndarray`` can be used.  To display to DS9 there
is the ``imexam`` package.  The ``numdisplay`` package is still available for
now but it is no longer supported by STScI.  We will show
how to use ``imexam`` to display and read the cursor position.  Read the
documentation on that tool to learn more about what else it has
to offer.

Displaying with imexam
----------------------

Here is an example how to display pixel data to DS9 with ``imexam``.  You must
start ``ds9`` before running this example.

::

    >>> import imexam
    >>> ad = astrodata.open('../playdata/N20170521S0925_forStack.fits')

    # Connect to the DS9 window (should already be opened.)
    >>> ds9 = imexam.connect(list(imexam.list_active_ds9())[0])

    >>> ds9.view(ad[0].data)

    # To scale "a la IRAF"
    >>> ds9.view(ad[0].data)
    >>> ds9.scale('zscale')

    # To set the mininum and maximum scale values
    >>> ds9.view(ad[0].data)
    >>> ds9.scale('limits 0 2000')


Retrieving cursor position with imexam
--------------------------------------

The function ``readcursor()`` can be used to retrieve cursor
position in pixel coordinates.  Note that it will **not** respond to
mouse clicks, **only** keyboard entries are acknowledged.

When invoked, ``readcursor()`` will stop the flow of the program and wait
for the user to put the cursor on top of the image and type a key.  A
tuple with three values will be returned:  the x and
y coordinates **in 0-based system**, and the value of the key the user
hit.

::

    >>> import imexam
    >>> ad = astrodata.open('../playdata/N20170521S0925_forStack.fits')

    # Connect to the DS9 window (should already be opened.)
    # and display
    >>> ds9 = imexam.connect(list(imexam.list_active_ds9())[0])
    >>> ds9.view(ad[0].data)
    >>> ds9.scale('zscale')


    >>> cursor_coo = ds9.readcursor()
    >>> print(cursor_coo)

    # To extract only the x,y coordinates
    >>> (xcoo, ycoo) = cursor_coo[:2]
    >>> print(xcoo, ycoo)

    # If you are also interested in the keystroke
    >>> keystroke = cursor_coo[2]
    >>> print('You pressed this key: %s' % keystroke)


Useful tools from the NumPy, SciPy, and Astropy Packages
========================================================
Like for the Display section, this section is not really specific to
Astrodata but is rather a quick show-and-tell of a few things that can
be done on the pixels with the big scientific packages NumPy, SciPy,
and Astropy.

Those three packages are very large and rich.  They have their own
extensive documentation and it is highly recommend for the users to learn about what
they have to offer.  It might save you from re-inventing the wheel.

The pixels, the variance, and the mask are stored as NumPy ``ndarray``'s.
Let us go through some basic examples, just to get a feel for how the
data in an ``AstroData`` object can be manipulated.

ndarray
-------
The data are contained in NumPy ``ndarray`` objects.  Any tools that works
on an ``ndarray`` can be used with Astrodata.

::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')

    >>> data = ad[0].data

    >>> # Shape of the array.  (equivalent to NAXIS2, NAXIS1)
    >>> data.shape
    (2112, 288)

    >>> # Value of a pixel at "IRAF" or DS9 coordinates (100, 50)
    >>> data[49,99]
    455

    >>> # Data type
    >>> data.dtype
    dtype('uint16')

The two most important thing to remember for users coming from the IRAF
world or the Fortran world are that the array has the y-axis in the first
index, the x-axis in the second, and that the array indices are zero-indexed,
not one-indexed.  The examples above illustrate those two critical
differences.

It is sometimes useful to know the data type of the values stored in the
array.  Here, the file is a raw dataset, fresh off the telescope.  No
operations has been done on the pixels yet.  The data type of Gemini raw
datasets is always "Unsigned integer (0 to 65535)", ``uint16``.

.. warning::
    Beware that doing arithmetic on ``uint16`` can lead to unexpected
    results.  This is a NumPy behavior.  If the result of an operation
    is higher than the range allowed by ``uint16``, the output value will
    be "wrong".  The data type will not be modified to accommodate the large
    value.  A workaround, and a safety net, is to multiply the array by
    ``1.0`` to force the conversion to a ``float64``. ::

        >>> a = np.array([65535], dtype='uint16')
        >>> a + a
        array([65534], dtype=uint16)
        >>> 1.0*a + a
        array([ 131070.])



Simple Numpy Statistics
-----------------------
A lot of functions and methods are available in NumPy to probe the array,
too many to cover here, but here are a couple examples.

::

    >>> import numpy as np

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    >>> data = ad[0].data

    >>> data.mean()
    >>> np.average(data)
    >>> np.median(data)

Note how ``mean()`` is called differently from the other two. ``mean()``
is a ``ndarray`` method, the others are NumPy functions. The implementation
details are clearly well beyond the scope of this manual, but when looking
for the tool you need, keep in mind that there are two sets of functions to
look into. Duplications like ``.mean()`` and ``np.average()`` can happen,
but they are not the norm. The readers are strongly encouraged to refer to
the NumPy documentation to find the tool they need.


Clipped Statistics
------------------
It is common in astronomy to apply clipping to the statistics, a clipped
average, for example.   The NumPy ``ma`` module can be used to create masks
of the values to reject.  In the examples below, we calculated the clipped
average of the first pixel extension with a rejection threshold set to
+/- 3 times the standard deviation.

Before Astropy, it was possible to do something like that with only
NumPy tools, like in this example::

    >>> import numpy as np

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    >>> data = ad[0].data

    >>> stddev = data.std()
    >>> mean = data.mean()

    >>> clipped_mean = np.ma.masked_outside(data, mean-3*stddev, mean+3*stddev).mean()

There is no iteration in that example.  It is a straight one-time clipping.

For something more robust, there is an Astropy function that can help, in
particular by adding an iterative process to the calculation.  Here is
how it is done::

    >>> import numpy as np
    >>> from astropy.stats import sigma_clip

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    >>> data = ad[0].data

    >>> clipped_mean = np.ma.mean(sigma_clip(data, sigma=3))


Filters with SciPy
------------------
Another common operation is the filtering of an image, for example convolving
with a gaussian filter.  The SciPy module ``ndimage.filters`` offers
several functions for image processing.  See the SciPy documentation for
more information.

The example below applies a gaussian filter to the pixel array.

::

    >>> from scipy.ndimage import filters
    >>> import imexam

    >>> ad = astrodata.open('../playdata/N20170521S0925_forStack.fits')
    >>> data = ad[0].data

    >>> # We need to prepare an array of the same size and shape as
    >>> # the data array.  The result will be put in there.
    >>> convolved_data = np.zeros(data.size).reshape(data.shape)

    >>> # We now apply the convolution filter.
    >>> sigma = 10.
    >>> filters.gaussian_filter(data, sigma, output=convolved_data)

    >>> # Let's visually compare the convolved image with the original
    >>> ds9 = imexam.connect(list(imexam.list_active_ds9())[0])
    >>> ds9.view(data)
    >>> ds9.scale('zscale')
    >>> ds9.frame(2)
    >>> ds9.view(convolved_data)
    >>> ds9.scale('zscale')
    >>> ds9.blink()
    >>> # When you are convinced it's been convolved, stop the blinking.
    >>> ds9.blink(blink=False)

Note that there is an Astropy way to do this convolution, with tools in
``astropy.convolution`` package.  Beware that for this particular kernel
we have found that the Astropy ``convolve`` function is extremely slow
compared to the SciPy solution.
This is because the SciPy function is optimized for a Gaussian convolution
while the generic ``convolve`` function in Astropy can take in any kernel.
Being able to take in any kernel is a very powerful feature, but the cost
is time.  The lesson here is do your research, and find the best tool for
your needs.


Many other tools
----------------
There are many, many other tools available out there.  Here are the links to
the three big projects we have featured in this section.

* NumPy: `www.numpy.org <http://www.numpy.org>`_
* SciPy: `www.scipy.org <http://www.scipy.org>`_
* Astropy:  `www.astropy.org <http://www.astropy.org>`_

Using the Astrodata Data Quality Plane
======================================
Let us look at an example where the use of the Astrodata mask is
necessary to get correct statistics.  A GMOS imaging frame has large sections
of unilluminated pixels; the edges are not illuminated and there are two
bands between the three CCDs that represent the physical gap between the
CCDs.  Let us have a look at the pixels to have a better sense of the
data::

    >>> ad = astrodata.open('../playdata/N20170521S0925_forStack.fits')
    >>> import imexam
    >>> ds9 = imexam.connect(list(imexam.list_active_ds9())[0])

    >>> ds9.view(ad[0].data)
    >>> ds9.scale('zscale')

See how the right and left portions of the frame are not exposed to the sky,
and the 45 degree angle cuts of the four corners.  The chip gaps too.
If we wanted to do statistics on the whole frames, we certainly would not want
to include those unilluminated areas.  We would want to mask them out.

Let us have a look at the mask associated with that image::

    >>> ds9.view(ad[0].mask)
    >>> ds9.scale('zscale')

The bad sections are all white (pixel value > 0).  There are even some
illuminated pixels that have been marked as bad for a reason or another.

Let us use that mask to reject the pixels with no or bad information and
do calculations only on the good pixels.  For the sake of simplicity we will
just do an average.  This is just illustrative.  We show various ways to
accomplish the task; choose the one that best suits your need or that you
find most readable.

::

    >>> import numpy as np

    >>> # For clarity...
    >>> data = ad[0].data
    >>> mask = ad[0].mask

    >>> # Reject all flagged pixels and calculate the mean
    >>> np.mean(data[mask == 0])
    >>> np.ma.masked_array(data, mask).mean()

    >>> # Reject only the pixels flagged "no_data" (bit 16)
    >>> np.mean(data[(mask & 16) == 0])
    >>> np.ma.masked_array(data, mask & 16).mean()
    >>> np.ma.masked_where(mask & 16, data).mean()

The "long" form with ``np.ma.masked_*`` is useful if you are planning to do
more than one operation on the masked array.  For example::

    >>> clean_data = np.ma.masked_array(data, mask)
    >>> clean_data.mean()
    >>> np.ma.median(clean_data)
    >>> clean_data.max()


Manipulate Data Sections
========================
So far we have shown examples using the entire data array.  It is possible
to work on sections of that array.  If you are already familiar with
Python, you probably already know how to do most if not all of what is in
this section.  For readers new to Python, and especially those coming
from IRAF, there are a few things that are worth explaining.

When indexing a NumPy ``ndarray``, the left most number refers to the
highest dimension's axis.  For example, in a 2D array, the IRAF section
are in (x-axis, y-axis) format, while in Python they are in
(y-axis, x-axis) format.  Also important to remember is that the ``ndarray``
is 0-indexed, rather than 1-indexed like in Fortran or IRAF.

Putting it all together, a pixel position (x,y) = (50,75) in IRAF or from
the cursor on a DS9 frame, is accessed in Python as ``data[74,49]``.
Similarly, the IRAF section [10:20, 30:40] translate in Python to
[9:20, 29:40].  Also remember that when slicing in Python, the upper limit
of the slice is not included in the slice.  This is why here we request
20 and 40 rather 19 and 39.

Let's put it in action.

Basic Statistics on Section
---------------------------
In this example, we do simple statistics on a section of the image.

::

    >>> import numpy as np

    >>> ad = astrodata.open('../playdata/N20170521S0925_forStack.fits')
    >>> data = ad[0].data

    >>> # Get statistics for a 25x25 pixel-wide box centered on pixel
    >>> # (50,75)  (DS9 frame coordinate)
    >>> xc = 49
    >>> yc = 74
    >>> buffer = 25
    >>> (xlow, xhigh) = (xc - buffer//2, xc + buffer//2 + 1)
    >>> (ylow, yhigh) = (yc - buffer//2, yc + buffer//2 + 1)
    >>> # The section is [62:87, 37:62]
    >>> stamp = data[ylow:yhigh, xlow:xhigh]
    >>> mean = stamp.mean()
    >>> median = np.median(stamp)
    >>> stddev = stamp.std()
    >>> minimum = stamp.min()
    >>> maximum = stamp.max()

    >>> print(' Mean   Median  Stddev  Min   Max\n \
    ... %.2f  %.2f   %.2f    %.2f  %.2f' % \
    ... (mean, median, stddev, minimum, maximum))

Have you noticed that the median is calculated with a function rather
than a method?  This is simply because the ``ndarray`` object does not
have a method to calculate the median.

Example - Overscan Subtraction with Trimming
--------------------------------------------
Several concepts from previous sections and chapters are used in this
example.  The Descriptors are used to retrieve the overscan section and
the data section information from the headers.  Statistics are done on the
NumPy ``ndarray`` representing the pixel data.  Astrodata arithmetics is
used to subtract the overscan level.  Finally, the overscan section is
trimmed off and the modified ``AstroData`` object is written to a new file
on disk.

To make the example more complete, and to show that when the pixel data
array is trimmed, the variance (and mask) arrays are also trimmed, let us
add a variance plane to our raw data frame.

::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')

    >>> for (extension, gain) in zip(ad, ad.gain()):
    ...    extension.variance = extension.data / gain
    ...

    >>> # Here is how the data structure looks like before the trimming.
    >>> ad.info()
    Filename: ../playdata/N20170609S0154.fits
    Tags: ACQUISITION GEMINI GMOS IMAGE NORTH RAW SIDEREAL UNPREPARED

    Pixels Extensions
    Index  Content                  Type              Dimensions     Format
    [ 0]   science                  NDAstroData       (2112, 288)    uint16
              .variance             ndarray           (2112, 288)    float64
    [ 1]   science                  NDAstroData       (2112, 288)    uint16
              .variance             ndarray           (2112, 288)    float64
    [ 2]   science                  NDAstroData       (2112, 288)    uint16
              .variance             ndarray           (2112, 288)    float64
    [ 3]   science                  NDAstroData       (2112, 288)    uint16
              .variance             ndarray           (2112, 288)    float64

    >>> # Let's operate on the first extension.
    >>> #
    >>> # The section descriptors return the section in a Python format
    >>> # ready to use, 0-indexed.
    >>> oversec = ad[0].overscan_section()
    >>> datasec = ad[0].data_section()

    >>> # Measure the overscan level
    >>> mean_overscan = ad[0].data[oversec.y1: oversec.y2, oversec.x1: oversec.x2].mean()

    >>> # Subtract the overscan level.  The variance will be propagated.
    >>> ad[0].subtract(mean_overscan)

    >>> # Trim the data to remove the overscan section and keep only
    >>> # the data section.  Note that the WCS will be automatically
    >>> # adjusted when the trimming is done.
    >>> #
    >>> # Here we work on the NDAstroData object to have the variance
    >>> # trimmed automatically to the same size as the science array.
    >>> # To reassign the cropped NDAstroData, we use the reset() method.
    >>> ad[0].reset(ad[0].nddata[datasec.y1:datasec.y2, datasec.x1:datasec.x2])

    >>> # Now look at the dimensions of the first extension, science
    >>> # and variance.  That extension is smaller than the others.
    >>> ad.info()
    Filename: ../playdata/N20170609S0154.fits
    Tags: ACQUISITION GEMINI GMOS IMAGE NORTH RAW SIDEREAL UNPREPARED

    Pixels Extensions
    Index  Content                  Type              Dimensions     Format
    [ 0]   science                  NDAstroData       (2112, 256)    float64
              .variance             ndarray           (2112, 256)    float64
    [ 1]   science                  NDAstroData       (2112, 288)    uint16
              .variance             ndarray           (2112, 288)    float64
    [ 2]   science                  NDAstroData       (2112, 288)    uint16
              .variance             ndarray           (2112, 288)    float64
    [ 3]   science                  NDAstroData       (2112, 288)    uint16
              .variance             ndarray           (2112, 288)    float64

    >>> # We can write this to a new file
    >>> ad.write('partly_overscan_corrected.fits')

A new feature presented in this example is the ability to work on the
``NDAstroData`` object directly.  This is particularly useful when cropping
the science pixel array as one will want the variance and the mask arrays
cropped exactly the same way.  Taking a section of the ``NDAstroData``
object (ad[0].nddata[y1:y2, x1:x2]), instead of just the ``.data`` array,
does all that for us.

To reassign the cropped ``NDAstroData`` to the extension one uses the
``.reset()`` method as shown in the example.

Of course to do the overscan correction correctly and completely, one would
loop over all four extensions.  But that's the only difference.

Data Cubes
==========
Reduced Integral Field Unit (IFU) data is commonly represented as a cube,
a three-dimensional array.  The ``data`` component of an ``AstroData``
object extension can be such a cube, and it can be manipulated and explored
with NumPy, AstroPy, SciPy, imexam, like we did already in this section
with 2D arrays.  We can use matplotlib to plot the 1D spectra represented
in the third dimension.

In Gemini IFU cubes, the first axis is the X-axis, the second, the Y-axis,
and the wavelength is in the third axis.  Remember that in a ``ndarray``
that order is reversed (wlen, y, x).

In the example below we "collapse" the cube along the wavelenth axis to
create a "white light" image and display it.  Then we plot a 1D spectrum
from a given (x,y) position.

::

    >>> import imexam
    >>> import matplotlib.pyplot as plt

    >>> ds9 = imexam.connect(list(imexam.list_active_ds9())[0])

    >>> adcube = astrodata.open('../playdata/gmosifu_cube.fits')
    >>> adcube.info()

    >>> # Sum along the wavelength axis to create a "white light" image
    >>> summed_image = adcube[0].data.sum(axis=0)
    >>> ds9.view(summed_image)
    >>> ds9.scale('minmax')

    >>> # Plot a 1-D spectrum from the spatial position (14,25).
    >>> plt.plot(adcube[0].data[:,24,13])
    >>> plt.show()   # might be needed, depends on matplotlibrc interactive setting


Now that is nice but it would be nicer if we could plot the x-axis in units
of Angstroms instead of pixels.  We use the AstroData's WCS handler, which is
based on ``gwcs.wcs.WCS`` to get the necessary information.  A particularity
of ``gwcs.wcs.WCS`` is that it refers to the axes in the "natural" way,
(x, y, wlen) contrary to Python's (wlen, y, x). It truly requires you to pay
attention.

::

    >>> import matplotlib.pyplot as plt

    >>> adcube = astrodata.open('../playdata/gmosifu_cube.fits')

    # We get the wavelength axis in Angstroms at the position we want to
    # extract, x=13, y=24.
    # The wcs call returns a 3-element list, the third element ([2]) contains
    # the wavelength values for each pixel along the wavelength axis.

    >>> length_wlen_axis = adcube[0].shape[0]   # (wlen, y, x)
    >>> wavelengths = adcube[0].wcs(13, 24, range(length_wlen_axis))[2] # (x, y, wlen)

    # We get the intensity along that axis
    >>> intensity = adcube[0].data[:, 24, 13]   # (wlen, y, x)

    # We plot
    >>> plt.clf()
    >>> plt.plot(wavelengths, intensity)
    >>> plt.show()


Plot Data
=========
The main plotting package in Python is ``matplotlib``.  We have used it in the
previous section on data cubes to plot a spectrum.  There is also the project
called ``imexam`` which provides astronomy-specific tools for the
exploration and measurement of data.  We have also used that package above to
display images to DS9.

In this section we absolutely do not aim at covering all the features of
either package but rather to give a few examples that can get the readers
started in their exploration of the data and of the visualization packages.

Refer to the projects web pages for full documentation.

* Matplotlib: `https://matplotlib.org <https://matplotlib.org/>`_
* imexam: `https://github.com/spacetelescope/imexam <https://github.com/spacetelescope/imexam>`_

Matplotlib
----------
With Matplotlib you have full control on your plot.  You do have to do a bit
for work to get it perfect though.  However it can produce publication
quality plots.  Here we just scratch the surface of Matplotlib.

::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from astropy import wcs

    >>> ad_image = astrodata.open('../playdata/N20170521S0925_forStack.fits')
    >>> ad_spectrum = astrodata.open('../playdata/estgsS20080220S0078.fits')

    >>> # Line plot from image.  Row #1044 (y-coordinate)
    >>> line_index = 1043
    >>> line = ad_image[0].data[line_index, :]
    >>> plt.clf()
    >>> plt.plot(line)
    >>> plt.show()

    >>> # Column plot from image, averaging across 11 pixels around colum #327
    >>> col_index = 326
    >>> width = 5
    >>> xlow = col_index - width
    >>> xhigh = col_index + width + 1
    >>> thick_column = ad_image[0].data[:, xlow:xhigh]
    >>> plt.clf()
    >>> plt.plot(thick_column.mean(axis=1))  # mean along the width.
    >>> plt.show()
    >>> plt.ylim(0, 50)     # Set the y-axis range
    >>> plt.plot(thick_column.mean(axis=1))
    >>> plt.show()

    >>> # Contour plot for a section of an image.
    >>> center = (1646, 2355)
    >>> width = 15
    >>> xrange = (center[1]-width//2, center[1] + width//2 + 1)
    >>> yrange = (center[0]-width//2, center[0] + width//2 + 1)
    >>> blob = ad_image[0].data[yrange[0]:yrange[1], xrange[0]:xrange[1]]
    >>> plt.clf()
    >>> plt.imshow(blob, cmap='gray', origin='lower')
    >>> plt.contour(blob)
    >>> plt.show()

    >>> # Spectrum in pixels
    >>> plt.clf()
    >>> plt.plot(ad_spectrum[0].data)
    >>> plt.show()

    >>> # Spectrum in Angstroms
    >>> spec_wcs = wcs.WCS(ad_spectrum[0].hdr)
    >>> pixcoords = np.array(range(ad_spectrum[0].data.shape[0]))
    >>> wlen = spec_wcs.wcs_pix2world(pixcoords, 0)[0]
    >>> plt.clf()
    >>> plt.plot(wlen, ad_spectrum[0].data)
    >>> plt.show()


imexam
------
For those who have used IRAF, ``imexam`` is a well-known tool.  The Python
``imexam`` reproduces many of of the features of its IRAF predecesor, the interactive mode of
course, but it also offers programmatic tools.  One can even control DS9
from Python.  As for Matplotlib, here we really just scratch the surface of
what ``imexam`` has to offer.

::

    >>> import imexam
    >>> from imexam.imexamine import Imexamine

    >>> ad_image = astrodata.open('../playdata/N20170521S0925_forStack.fits')

    # Display the image
    >>> ds9 = imexam.connect(list(imexam.list_active_ds9())[0])
    >>> ds9.view(ad_image[0].data)
    >>> ds9.scale('zscale')

    # Run in interactive mode.  Try the various commands.
    >>> ds9.imexam()

    # Use the programmatic interface
    # First initialize an Imexamine object.
    >>> plot = Imexamine()

    # Line plot from image.  Row #1044 (y-coordinate)
    >>> line_index = 1043
    >>> plot.plot_line(0, line_index, ad_image[0].data)

    # Column plot from image, averaging across 11 pixels around colum #327
    # There is no setting for this, so we have to do something similar
    # to what we did with matplotlib.
    >>> col_index = 326
    >>> width = 5
    >>> xlow = col_index - width
    >>> xhigh = col_index + width + 1
    >>> thick_column = ad_image[0].data[:, xlow:xhigh]
    >>> mean_column = thick_column.mean(axis=1)
    >>> plot.plot_column(0, 0, np.expand_dims(mean_column, 1))

    >>> # Contour plot for a section of an image.
    >>> center = (1646, 2355)  # in python coordinates
    >>> width = 15
    >>> plot.contour_pars['ncolumns'][0] = width
    >>> plot.contour_pars['nlines'][0] = width
    >>> plot.contour(center[1], center[0], ad_image[0].data)
