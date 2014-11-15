.. data:

.. _data:

**********
Pixel Data
**********

**Try it yourself**


If you wish to follow along and try the commands yourself, download
the data package, go to the ``playground`` directory and copy over
the necessary files.

::

   cd <path>/gemini_python_datapkg-X1/playground
   cp ../data_for_ad_user_manual/N20110313S0188.fits .
   cp ../data_for_ad_user_manual/gmosifu_cube.fits .
   cp ../data_for_ad_user_manual/estgsS20080220S0078.fits .

Then launch the Python shell::

   python


.. highlight:: python
   :linenothreshold: 5


Operate on the Pixel Data
=========================
The pixel data are stored in a numpy ``ndarray``.  Therefore anything 
that can be done with numpy on a ``ndarray`` can be done on the pixel 
data stored in the AstroData object.  Examples include arithmetic, 
statistics, display, plotting, etc.  Please refer to numpy documentation 
for details on what it offers.  In this chapter, we will present some typical 
examples.

But first, here is how to acess the data array stored in an AstroData 
object. ::

   from astrodata import AstroData
   
   ad = AstroData('N20110313S0188.fits')
   
   the_data = ad['SCI',2].data
   
   # Loop through the extensions. 
   for extension in ad['SCI']:
      the_data = extension.data
      print the_data.sum()


An extension's ``data`` attribute returns a numpy ``ndarray``.  An extension
must be specified by name (Line 5) or position ID, or extracted from the 
main AstroData object (Line 8).  Also, remember that PHUs do not have any
pixel data, so none of what is discussed in this chapter applies to the PHU.

The AstroData pixel data can be manipulated like any other ``ndarray``.  For
example, on Line 10, we calculate the sum of the pixel values with the 
``ndarray`` ``sum()`` method.


Arithmetic on AstroData Objects
===============================
AstroData supports basic arithmetic directly:

+----------------+---------+
| addition       | .add()  |
+----------------+---------+
| subtraction    | .sub()  |
+----------------+---------+
| multiplication | .mult() |
+----------------+---------+
| division       | .div()  |
+----------------+---------+

The big advantage of using the AstroData 
implementation of those operator is that if the AstroData object has variance 
and data quality planes, those will be calculated and propagated to the 
output appropriately. ::

   from astrodata import AstroData
   
   ad = AstroData('N20110313S0188.fits')
   
   # addition
   #   ad = ad + 5.
   ad.add(5.)
   
   # subtraction
   #   ad = ad - 5.
   ad.sub(5.)
   
   # multiplication.  Using descriptor as operand.
   #   ad = ad * gain
   ad.mult(ad.gain())
   
   # division. Using descriptor as operand.
   #   ad = ad / gain
   ad.div(ad.gain())
   
When using AstroData arithmetic, all the science frames (``EXTNAME='SCI'``) 
are operated on.  The modifications are **done in-place**, the AstroData
object is modified.

The AstroData arithmetic methods can be stringed together.  Note that because
the calculations are done "in-place", **operator precedence cannot be 
respected**. For example, 

::

   ad.add(5).mult(10).sub(5)
   
   # means:  ad = ((ad + 5) * 10) - 5
   # not: ad = ad + (5 * 10) - 5
   
The AstroData data arithmetic method modify the data "in-place".  This means 
that the data values are modified and the original values are no more.  If 
you need to keep the original values unmodified, for example, you will need 
them later, use ``deepcopy`` to make a separate copy on which you can work. 

Let us say that we want to calculate ``y = x*10 + x`` where ``x`` is the 
pixel values.  We must use ``deepcopy`` here since after ``ad.mult(10)`` the
values in ``ad`` will have been modified and cannot be used for the ``+ x``
part of the equation.

Let us follow a pixel through the math.

The WRONG way to calculate ``x*10 + x``::

   from astrodata import AstroData
   
   ad = AstroData('N20110313S0188.fits')
   
   value_before = ad['SCI',1].data[50,50]
   expected_value_after = value_before*10 + value_before
   
   ad.mult(10).add(ad)
   bad_value_after = ad['SCI',1].data[50,50]
   
   print expected_value_after, bad_value_after
   
   ad.close()
   
   # The result of the arithmetic above is y = (x*10) + (x*10)

The CORRECT way to calculate ``x*10 + x``::

   from astrodata import AstroData
   from copy import deepcopy

   ad = AstroData('N20110313S0188.fits')
   adcopy = deepcopy(ad)

   value_before = ad['SCI',1].data[50,50]
   expected_value_after = value_before*10 + value_before
  
   ad.add(adcopy.mult(10))
   
   good_value_after = ad['SCI',1].data[50,50]
   print expected_value_after, good_value_after
   
   ad.close()
   adcopy.close()

   
As one can see, for complex equation, using the AstroData arithmetic method
can get fairly confusing.  Operator overload would solve this situation but
it has not been implemented yet mostly due to a lack of resources.  Therefore, 
we recommend the use numpy for really complex equation since operator overload 
is implemented in numpy and the operator precedence is respected.  The 
downside is that if you need the variance plane propagated correctly, you will 
have to do the math yourself. 

Here is the ``y = x*10 + x`` operation again, but this time numpy is used on
the ``ndarray`` returned by ``.data``.  Like before, we follow a pixel through
the math.

::

   from astrodata import AstroData
   
   ad = AstroData('N20110313S0188.fits')
      
   value_before = ad['SCI',1].data[50,50]
   expected_value_after = value_before*10 + value_before
   
   for extension in ad['SCI']:
       data_array = extension.data
       data_array = data_array*10 + data_array
       extension.data = data_array
       
   value_after = ad['SCI',1].data[50,50]
   print expected_value_after, value_after
   
   ad.close()

Variance
========

The AstroData arithmetic methods can propagate the variance planes, if any are
present.  The variance extensions must be named ``VAR`` to be recognized as 
such.

The initial variance from read noise and poisson noise normally needs to be
calculated by the programmer; the raw data normally contains only science 
extensions.

Adding variance extensions
--------------------------

For the sake of simplicity, only the poisson noise is considered in this
example.

::

   from astrodata import AstroData
   
   ad = AstroData('N20110313S0188.fits')
   ad.info()
   
   for extension in ad['SCI']:
      variance = extension.data / extension.gain().as_pytype()
      variance_header = extension.header
      variance_extension = AstroData(data=variance, header=variance_header)
      variance_extension.rename_ext('VAR')
      ad.append(variance_extension)
   
   ad.info()
   
   # Let's save a copy of this dataset.
   ad.write('N188_with_var.fits')
   ad.close()

On Line 6, the loop through all the science extension is launched.  The 
Poisson noise will be calculated for each science extension and stored in a 
new extension named 'VAR'.  The extension version informs on the association
between the 'SCI' and the 'VAR' extensions, eg. ['VAR', 1] is the variance
for ['SCI', 1].

For each science extension, the variance is calculated from the pixel data 
and the gain obtained from the Descriptor ``.gain`` (Line 7).  Note the use
of ``as_pytype()`` on the Descriptor.  Since ``extension.data`` is a 
``ndarray`` not a standard Python type, the DescriptorValue does not know
how it is expected to behave, requiring the use of ``as_pytype()`` which
converts the DescriptorValue to a Python float.

On Line 8, we simply copy the header for the science extension and use that
as the header for the new variance extension (Line 9).

The new variance extension is renamed to 'VAR' on Line 10 -- it was 'SCI' since
we copied the header -- and append the extension to the AstroData object
(Line 11).

Finally, we write that AstroData object to a new MEF on disk.  We will use
that MEF in the next examples.

For reference, the AstroData object before the variance planes are added looks
like this::
   
   Filename: N20110313S0188.fits
       Type: AstroData
       Mode: readonly
   
   AD No.    Name          Type      MEF No.  Cards    Dimensions   Format   
           hdulist       HDUList
           phu           PrimaryHDU    0       179
           phu.header    Header
   [0]     ('SCI', 1)    ImageHDU      1        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [1]     ('SCI', 2)    ImageHDU      2        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [2]     ('SCI', 3)    ImageHDU      3        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray


After the variance planes are added, the structure looks like this::

   Filename: N20110313S0188.fits
       Type: AstroData
       Mode: readonly
   
   AD No.    Name          Type      MEF No.  Cards    Dimensions   Format   
           hdulist       HDUList
           phu           PrimaryHDU    0       179
           phu.header    Header
   [0]     ('SCI', 1)    ImageHDU      1        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [1]     ('SCI', 2)    ImageHDU      2        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [2]     ('SCI', 3)    ImageHDU      3        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [3]     ('VAR', 1)    ImageHDU      4        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [4]     ('VAR', 2)    ImageHDU      5        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray
   [5]     ('VAR', 3)    ImageHDU      6        37    (2304, 1056)  float32
              .header    Header
              .data      ndarray



Automatic variance propagation
------------------------------

As mentioned before, if the AstroData arithmetic methods are used, the 
variance will be propagated automatically.  A simple ``ad.mult()`` suffices
to multiply the science pixels and calculate the resulting variance, for all
extensions.

Let us follow a science pixel and a variance pixel through the AstroData 
arithmetic. ::

   #     output = x * x
   # var_output = var * x^2 + var * x^2
   
   from astrodata import Astrodata
   
   ad = AstroData('N188_with_var.fits')
   
   value_before = ad['SCI',1].data[50,50]
   variance_before = ad['VAR',1].data[50,50]  
   expected_value_after = value_before + value_before
   expected_variance_after = 2 * (variance_before * value_before * value_before)
   
   ad.mult(ad)
   
   value_after = ad['SCI',1].data[50,50]
   variance_after = ad['VAR',1].data[50,50]
   print expected_value_after, value_after
   print expected_variance_after, variance_after
   
   ad.close()

So all it took to multiply the science extensions by themselves and propagate
the variance accordingly was ``ad.mult(ad)`` (Line 13).

Manual propagation with numpy
-----------------------------

To do the same thing as ``ad.mult(ad)``, but by operating directly on the 
numpy arrays of each extension::

   from astrodata import AstroData
   
   ad = AstroData('N188_with_var.fits')

   # This loop is the equivalent of ``ad.mult(ad)``
   for i in range(1,ad.count_exts('SCI')+1):
       d = ad['SCI',i].data
       v = ad['VAR',i].data
       data = d*d
       variance = v * d*d + v * d*d
       ad['SCI',i].data = data
       ad['VAR',i].data = variance
   
   ad.close()


Display
=======
Displaying ``ndarray`` arrays from Python is straighforward with the 
``numdisplay`` module.  The module also has a function to read the position 
the cursor, which can be useful when developing an interactive tool.

The ``numdisplay`` module is a module of the ``stsci.tools`` package
distributed in Ureka.

Displaying
----------

To display the pixel data of an AstroData extension, the 
``numdisplay.display`` function is used. A display tool, like DS9 or ximtool, 
must also be running.

::

   from astrodata import AstroData
   from stsci.numdisplay import display
   
   ad = AstroData('N20110313S0188.fits')
   
   display(ad['SCI',1].data)
   
   # To scale "a la IRAF"
   display(ad['SCI',1].data, zscale=True)
   
   # To set the minimum and maximum values to display
   display(ad['SCI',1].data, z1=700, z2=2000)

``numdisplay.display`` accepts various arguments. See ``help(display)`` to
get more information.  The examples on Line 6, 9, and 12, are probably the
most common, especially for users coming from IRAF.

Retrieving cursor position
--------------------------

The function ``numdisplay.readcursor`` can be used to retrieve cursor position.
Note that it will **not** respond to mouse clicks, **only** keyboard entries
are acknowledged.

When invoked, ``readcursor()`` will stop the flow of the program and wait for
the user to put the cursor on top of the image and type a key.  A **string** 
with four space-separated values are going to be returned: the x and y 
coordinates, a frame reference number, and the value of the key the user hit.

::

   # here we assume that the previous example has just been run.
   
   from stsci.numdisplay import readcursor

   # User instructions: Put cursor on image, type a key.
   cursor_coo = readcursor()
   print cursor_coo
   
   # To extract only the x,y coordinates:
   (xcoo, ycoo) = cursor_coo.split()[:2]
   print xcoo, ycoo
   
   # If you are also interested in the keystoke:
   (xcoo, ycoo, junk, keystroke) = cursor_coo.split()
   print 'You pressed this key: "%s"' % keystroke
  

Useful tools from the Numpy and SciPy Modules
=============================================

Like the Display section, this section is not really specific to AstroData,
but is rather an introduction to numpy and scipy, and to using those modules
on ``ndarray`` objects.  Since AstroData pixel data is stored in that format,
it is believe important to show a few examples to steer new users in the 
right direction.
 
The ``numpy`` and ``scipy`` modules offer a multitude of functions and tools. 
They both have their own documentation.  Here we simply highlighting a few 
functions that could be used for common things an astronomer might want to do.  
The idea is to get the reader started in her exploration of ``numpy`` and 
``scipy``.

ndarray
-------

::

   from astrodata import AstroData
   import numpy as np
   
   ad = AstroData('N20110313S0188.fits')
   data = ad['SCI',2].data
   
   # Shape of array, (NAXIS2, NAXIS1)
   data.shape

   # Value of pixel with IRAF coordinates (100, 50)
   data[49,99]
   
   # Data type
   data.dtype

The two most important thing to remember for users coming from the IRAF world 
are that the array has the y-axis in the first index, the x-axis in the second
(Line 8, 11), and that the array indices are zero-indexed, not one-indexed 
(Line 11).

Sometimes it is useful to know the type of the values stored in the array,
eg. integer, float, double-precision, etc., this information is obtained with
``dtype`` (Line 14).


Simple numpy statistics
-----------------------

A lot of functions and methods are available in numpy to probe the array, too
many to cover here, but here are a couple examples.

::
   
   data.mean()
   np.average(data)
   np.median(data)

Note how ``mean()`` is called differently from the other two.  ``mean()`` is
a ``ndarray`` method, the others are numpy functions.  The implementation 
details are clearly well beyond the scope of this manual, but when looking
for the tool you need, keep in mind that there are two sets of functions to
look into.  Duplications like ``.mean()`` and ``np.average()`` can happen,
but they are not the norm.  The readers are strongly encouraged to refer to the
numpy documentation to find the tool they need.

Clipped statistics
------------------

It is common in astronomy to apply clipping to the statistics, a clipped
average, for example.

The numpy ``ma`` module can be used to create masks of the values to reject.

In the example below, we calculate a clipped mean with rejection at 
+/- 3 times the standard deviation.

::

   import numpy.ma as ma
      
   stddev = data.std()
   mean = data.mean()
   
   clipped_mean = ma.masked_outside(data, mean-3*stddev, mean+3*stddev).mean()
   

On Line 6, ``ma.masked_outside`` identify all values falling outside the 
allowed range, and creates a ``MaskedArray`` which is a combination of the 
data array and a True/False mask, with True identifying the values to reject.
The really convenient thing is that the method ``mean`` will take this 
information and calculate the average ignoring all values tagged for rejection.

WARNING: The ``numpy.clip`` function is not the equivalent of the mask
solution.  As explain in the ``numpy.clip`` documentation, the function 
*replaces* the extreme values with the minimum and maximum, it *does not
reject* the extreme values.  


Filters with scipy
------------------

Another common operation is the filtering of an image, for example convolving
with a gaussian filter.  The scipy module ``ndimage.filters`` offers several
such functions for image processing.  See the scipy documentation for more
details.

The example below applies a gaussian filter to a pixel array.

:: 

   import scipy.ndimage.filters as filters
   from stsci.numdisplay import display
   
   convolved_data = np.zeros(data.size).reshape(data.shape)
   sigma = 10.
   filters.gaussian_filter(data, sigma, output=convolved_data)
   
   # visually compare the convolve image with the original.
   display(data, zscale=True)
   display(convolved_data, zscale=True, frame=2)
   
   # To put the convolved data back in the AstroData object
   ad['SCI',2].data = convolved_data

On Line 4, a numpy array of the same size and shape as the input data is 
created and filled with zeros.  This receives the output convolved image 
produced by the function ``gaussian_filter`` (Line 6).  

On Line 13, the input data is replaced with the convolved data.  Remember
that one will need to use ``write`` to make that change effective on disk.


Many other tools
----------------

The world of ``numpy``, ``scipy``, and the new ``astropy`` is rich and vast.
The reader should refer to those packages' documentation to learn more.


Using the AstroData Data Quality Plane
======================================

.. todo::
   Write examples that use the DQ plane.  Eg. transform DQ plane in a numpy 
   mask and do statistics.


Manipulate Data Sections
========================

Sections of the data array can be accessed and processed.  It is important to
note here that when indexing a numpy array, the left most number refers to the
highest dimension's axis (eg. in IRAF sections are in (x,y) format, in Python
they are in (y,x) format). Also important is to remember that the numpy arrays
are 0-indexed, not 1-indexed like in Fortran or IRAF.  For example, in a 2-D 
numpy array, the pixel position (x,y) = (50,75) would be accessed as 
data[74,49].

Basic statistics on section
---------------------------

::

   from astrodata import AstroData
   import numpy as np
   
   ad = AstroData('N20110313S0188.fits')
   data = ad['SCI',2].data
   
   # Get statistics for a 25x25 pixel-wide box centered on pixel 50,75.
   mean = data[62:87,37:62].mean()
   median = np.median(data[62:87,37:62])
   stddev = data[62:87,37:62].std()
   minimum = data[62:87,37:62].min()
   maximum = data[62:87,37:62].max()
   
   print "Mean    Median Stddev  Min    Max\n", \
      mean, median, stddev, minimum, maximum

There is one odd thing that the reader should notice.  On Line 9, ``median`` 
is not being called like the others.  This is because there is no ``median``
method associated with a ``ndarray``.  But there is a numpy function, so that
is what is used.

Example - Overscan subtraction
------------------------------

Several concepts from previous chapters are used in this example.
The Descriptors are used to retrieve overscan section and data section 
information from the headers.  Numpy statistics is done on the pixel data 
sections.  AstroData arithmetics is used to subtract the overscan level. 
Finally, the overscan section is trimmed off and the AstroData is written to 
a new file.

To make the example more complete, we use the file created in the Variance
section.  We will propagate the variance and trim the variance planes too.

::

   from astrodata import AstroData
   
   ad = AstroData('N188_with_var.fits')

   # Get EXTVER-keyed dictionary for the overscan section and the data 
   # section.
   oversec_dict = ad.overscan_section().collapse_by_extver()
   datasec_dict = ad.data_section().collapse_by_extver()
   
   # Loop through the science extensions. 
   for ext in ad['SCI']:
      #
      extver = ext.extver()
      #
      (x1, x2, y1, y2) = oversec_dict[extver]
      (dx1, dx2, dy1, dy2) = datasec_dict[extver]
      #
      # Measure the overscan level
      mean_overscan = ext.data[y1:y2,x1:x2].mean() 
      # 
      # Append variance and subtract overscan. Variance is propagated.
      ext.append(ad['VAR',extver])
      ext.sub(mean_overscan)
      #
      # Trim the data to remove the overscan section and keep only
      # the data section.
      ext['SCI',extver].data = ext['SCI',extver].data[dy1:dy2,dx1:dx2]
      ext['VAR',extver].data = ext['VAR',extver].data[dy1:dy2,dx1:dx2]

   ad.write('N188_overscanTrimmed.fits')

The dataset loaded in Line 3 has three 'SCI' extensions and three 'VAR' 
extensions.  

On Lines 7 and 8, the descriptors ``overscan_section`` and ``data_section`` 
are used to retrieve the region information relating to the location
of the overscan section and data section in each extension.  The return
values for each extension is a list of four zero-indexed indices identifying 
the x and y axes lower and upper range for the section.

Since the overscan and data sections are the same for a given extension 
version regardless of the extension name (ie. 'SCI' and 'VAR' have 
identical size and structure), the section information is keyed by extension
version only with the help of the ``collapse_by_extver()`` method (Line 7
and 8).

The content of oversec_dict is ::

   1: [0, 32, 0, 2304], 2: [0, 32, 0, 2304], 3: [1024, 1056, 0, 2304]}

The overscan section and the data section could have been obtained in the 
loop for each extension.  It is however more efficient to retrieve the
descriptor information once than three times in the loop, when it is 
possible to do so.

Moving on to the loop starting on Line 11.  For convenience, the extension
version of the current extension is stored in a variable on Line 13.  It
will be used to access the section dictionaries from Line 7 and 8, and to
associate a variance extension to the current extension.

The boundaries of the overscan section and of the data section for the
current extension are assigned to convenient variables on Lines 15 and 16.
The values are zero-indexed to match the ``ndarray``.

The overscan is here simply calculated as the average within the overscan
section (Line 19).

To simplying our lifes, we want to be using the automatic variance propagation
offered by the AstroData arithmetic utility.  The extension obtained from the
``in`` on Line 11 is an AstroData object with only the 'SCI' extension.  No
variance propagation will occur if there are no 'VAR' extension in that
AstroData object.  Therefore, on Line 22, we append the 'VAR' extension of the
same version as the 'SCI' extension.  When the subtraction is done on Line 23,
the operation affects both the 'SCI' extension and the newly appended 'VAR'
extension.

Now that the overscan level has been subtracted, that section needs to be 
trimmed off.  Only the data section is kept.  This is done on Lines 27 and 28.
Note that after the 'VAR' extension is appended, there are two extensions in
the AstroData object, and in order to work with them, they must now be 
specified.  Contrast that with the calls on Lines 13 and 19; then there were
still only one extension, the 'SCI' extension, and selection was implicit.

A final but important reminder.  All of the work was done on the extensions
extracted from the original AstroData object, ``ad``.  Since the extensions 
are not true copies but rather references to the original source, all the 
modifications to ``ext`` are in reality modifications to ``ad``.  So, now, all
we have to do is write the modified ``ad`` to disk with another name (Line 31).


Data Cubes
==========

Reduced Integral Field Unit (IFU) data is commonly stored in a cube, a 
three-dimensional array.  The ``data`` component of an AstroData object can
be such a cube, and can be manipulated and explored with numpy, plotted with
matplotlib (the next section shows a few more matplotlib examples.)

In the MEF file, the x and y axes are in the first and second dimension, and 
the third dimension is for the wavelength axis. In the ``ndarray``, that order 
is reversed.  The dimensions of the ``ndarray`` are (wavelength, y, x).
   
::

   from astrodata import AstroData
   import numpy as np
   from stsci.numdisplay import display
   import matplotlib.pyplot as plt
   
   adcube = AstroData('gmosifu_cube.fits')
   adcube.info()
   
   # The shape of the cube, (wavelength, y, x).   
   adcube.data.shape
   
   # Sum along the wavelength axis to create a "white light" image.
   sum_image = adcube.data.sum(axis=0)
   display(sum_image, zscale=True)
   
   # Plot a 1-D spectrum from pixel position (7,30)
   plt.plot(adcube.data[:,29,6])
   plt.show()
   
   # Plot the same thing but with wavelength along the x axis of the plot.
   crval3 = adcube.get_key_value('CRVAL3')
   cdelt3 = adcube.get_key_value('CDELT3')
   spec_pixel_length = adcube.data[:,29,6].size
   wavelength = crval3 + np.arange(spec_pixel_length)*cdelt3
   plt.clf()
   plt.plot(wavelength, adcube.data[:,29,6])
   plt.show()

On Line 13, the cube is "collapsed" in the wavelength dimension to produce a
"white light" 2-D image.

On Line 17, a vector through the cube at pixel position (7,30) is retrieved
and plotted with matplotlib.  The pixel values are plotted against the 
zero-indexed pixel position along the wavelength axis.  The ``show()`` 
statement (Line 18) might or might not be needed depending on your 
``interactive`` setting in the ``matplotlibrc`` file.  

To plot the spectrum with physical units on the plot's x-axis one needs the
WCS information from the header (Lines 21 to 24).  The statement ``plt.clf()``
simply clears the figure.

People familiar with astropy might prefer to use the ``wcs`` module to 
convert pixels to wavelenths (Lines 21 to 24).


Plot Data
=========
In Python, the main tool to create plots is ``matplotlib``.  We have used it 
in the previous section on data cubes.  Here we do not aimed at covering all 
of ``matplotlib``; the reader should refer to that package's documentation.  
Rather we will give a few examples that might be of use for quick inspection 
of the data.

::

   from astrodata import AstroData
   import matplotlib.pyplot as plt
   from matplotlib.colors import LogNorm
   
   adimg = AstroData('N20110313S0188.fits')
   adspec = AstroData('estgsS20080220S0078.fits')
   
   # Line plot from image.  Row #1044.
   line_index = 1043
   line = adimg['SCI',2].data[line_index, :]
   plt.clf()
   plt.plot(line)
   plt.show()
   
   # Column plot from image, averaging across 11 pixels around column #327.
   col_index = 326
   width = 5
   col_section = adimg['SCI',2].data[:,col_index-width:col_index+width+1]
   column = col_section.mean(axis=1)
   plt.clf()
   plt.plot(column)
   plt.show()
   
   # Contour plot for section
   galaxy = adimg['SCI',2].data[1045:1085,695:735]
   plt.clf()
   plt.imshow(galaxy, cmap='binary', norm=LogNorm())
   plt.contour(galaxy)
   plt.axis('equal')
   plt.show()
   
   # Spectrum in pixel
   plt.clf()
   plt.plot(adspec['SCI',1].data)
   plt.show()
   
   # Spectrum in wavelength (CRPIX1 = 1)
   crpix1 = adspec['SCI',1].get_key_value('CRPIX1')
   crval1 = adspec['SCI',1].get_key_value('CRVAL1')
   cdelt1 = adspec['SCI',1].get_key_value('CDELT1')
   length = adspec['SCI',1].get_key_value('NAXIS1')
   wavelengths = crval1 + (np.arange(length)-crpix1+1)*cdelt1
   plt.clf()
   plt.plot(wavelengths, adspec['SCI',1].data)
   plt.show()

