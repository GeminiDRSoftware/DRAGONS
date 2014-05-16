.. data:

.. _data:

**********
Pixel Data
**********

Operate on the Pixel Data
=========================
The pixel data is stored as a numpy ndarray.  This means that anything that can be done with numpy on a ndarray
can be done on the pixel data stored in the AstroData object.  Examples include arithmetic, statistics, display,
plotting, etc.  Please refer to numpy documentation for details on what it offers.  In this chapter, we will 
present some typical examples.

But first, here's how one accesses the data array stored in an AstroData object. ::

  from astrodata import AstroData
  
  ad = AstroData('N20110313S0188.fits')
  
  # The PHU does not have any pixel data.  Only the extensions can have pixel data.
  the_data = ad['SCI',2].data
  
  # or to loop through the extensions.  Here we just print the sum of all the pixels
  # for each extension.
  for extension in ad['SCI']:
     the_data = extension.data
     print the_data.sum()
  
  ad.close()

Arithmetic on AstroData Objects
===============================
AstroData supports basic arithmetic directly: addition, subtraction, multiplication, division.
The big advantage of using the AstroData implementation of those operator is that if the
AstroData object has variance and data quality planes, those will be calculated and propagated
to the output appropriately. ::

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
   
When using the AstroData arithmetic, all the science (EXTNAME='SCI') frames are
operated on.

The AstroData arithmetic methods can be stringed together.  Note that because
the calculations are done "in-place", **operator precedence cannot be respected**. ::

   ad.add(5).mult(10).sub(5)
   # means:  ad = ((ad + 5) * 10) - 5
   # not: ad = ad + (5 * 10) - 5
   
   ad.close()
   
The AstroData data arithmetic method modify the data "in-place".  This means that the data
values are modified and the original values are no more.  If you need to keep the original
values unmodified, for example, you will need them later, use ``deepcopy`` to make a separate
copy on which you can work. ::

   from astrodata import AstroData
   from copy import deepcopy
   
   ad = AstroData('N20110313S0188.fits')

   # To do:  x = x*10 + x
   #  One must use deepcopy because after the mult(10), 'ad' has been modified
   #  and it is that modified version that will be use in add(ad)

   # Let's follow a pixel through the math
   
   value_before = ad['SCI',1].data[50,50]
   expected_value_after = value_before*10 + value_before
   
   ad.mult(10).add(ad)
   bad_value_after = ad['SCI',1].data[50,50]
   
   print expected_value_after, bad_value_after
   
   # The result of the arithmetic above is x = (x*10) + (x*10)
   
   # To do the right thing, one can use ``deepcopy``
   # First let's reload a fresh ad.
   ad = AstroData('N20110313S0188.fits')
   adcopy = deepcopy(ad)
   
   ad.add(adcopy.mult(10))
   
   good_value_after = ad['SCI',1].data[50,50]
   print expected_mean_after, good_mean_after

   ad.close()
   adcopy.close()

   
As one can see, for complex equation, using the AstroData arithmetic method
can get fairly confusing.  Operator overload would solve this situation but
it has not been implemented yet.  Therefore, we recommend to use numpy for
really complex equation since operator overload is implemented and the operator
precedence is respected.  The downside is that if you need the variance plane
propagate correctly, you will have to do the math yourself. ::

   from astrodata import AstroData
   
   ad = AstroData('N20110313S0188.fits')
   
   # Let's do 'x = x*10 + x' again but this time we operate directly on
   # the numpy ndarray return by '.data'.  We will follow a pixel through
   # the math like before.
   
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

Here we demonstrate the variance propagation when using AstroData arithmetic methods.
First let us create and append variance planes to our file.  We will just add the poisson
noise and ignore read noise for the purpose of this example. ::

  from astrodata import AstroData
  from copy import deepcopy
  
  ad = AstroData('N20110313S0188.fits')
  ad.info()
  
  for extension in ad['SCI']:
      variance = extension.data / extension.gain().as_pytype()
      variance_header = extension.header
      variance_extension = AstroData(data=variance, header=variance_header)
      variance_extension.rename_ext('VAR')
      ad.append(variance_extension)

  ad.info()
  
  # Let's just save a copy of this ad for later use.
  advar = deepcopy(ad)

Now let us follow a science pixel and a variance pixel through the AstroData arithmetic. ::

  #     output = x * x
  # var_output = var * x^2 + var * x^2
  
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
the variance accordingly was ``ad.mult(ad)``.

To do the same thing operating directly on the numpy array::

   # Let's recall the ad with the variance planes we created earlier
   ad = deepcopy(advar)
   
   for i in range(1,ad.count_exts('SCI')+1):
       d = ad['SCI',i].data
       v = ad['VAR',i].data
       data = d*d
       variance = v * d*d + v * d*d
       ad['SCI',i].data = data
       ad['VAR',i].data = variance

   print ad['VAR',1].data[50,50]


Display
=======
Displaying ``numpy`` arrays from Python is straighforward with the ``numdisplay`` module.
The module also has a function to read the position the cursor, which can be useful when
developing an interactive task.

Start a display tool, like DS9 or ximtool. Then try the commands below.::

  from astrodata import AstroData
  from stsci.numdisplay import display
  from stsci.numdisplay import readcursor
  
  ad = AstroData('N20110313S0188.fits')
  
  display(ad['SCI',1].data)
  
  # To scale "a la IRAF"
  display(ad['SCI',1].data, zscale=True)
  
  # To set the minimum and maximum values to display
  display(ad['SCI',1].data, z1=700, z2=10000)

If you need to retrieve cursor position inputs, the numdisplay.readcursor function can help.
It does not respond to mouse clicks, but it does respond to keyboard entries.::

  # Invoke readcursor() and put the cursor on top of the image.
  # Type any key.
  # cursor_coo will contain the x, y positions and in the last column the key that was typed.
  cursor_coo = readcursor()
  print cursor_coo
  
  # If you just want to extract the x,y coordinates:
  (xcoo, ycoo) = cursor_coo.split()[:2]
  print xcoo, ycoo
  
  # If you are also interested in the keystoke:
  (xcoo, ycoo, junk, keystroke) = cursor_coo.split()
  print 'You pressed this key: "%s"' % keystroke
  

Useful tools from the Numpy and SciPy Modules
=============================================
The ``numpy`` and ``scipy`` modules offer a multitude of functions and tools.  They
both have their own documentation.  Here we simply highlight a few functions that 
could be used for common things an astronomer might want to do.  The idea is to
get the reader started in her exploration of ``numpy`` and ``scipy``.

::

  from astrodata import AstroData
  import numpy as np
  import numpy.ma as ma
  import scipy.ndimage.filters as filters
  from stsci.numdisplay import display
  
  ad = AstroData('N20110313S0188.fits')
  data = ad['SCI',2].data
  
  # The shape of the ndarray stored in data is given by .shape
  # The first number is NAXIS2, the second number is NAXIS1.
  data.shape
  
  # Calculate the mean and median of the entire array.
  # Note how the way mean and median are called differently.
  data.mean()
  np.median(data)
  
  # If the desired operation is a clipped mean, ie. rejecting
  # values before calculating the mean, the numpy.ma module
  # can be used to mask the data.  Let's try a clipped mean
  # at -3 and +3 times the standard deviation
    
  # ma.masked_outside() with mask out anything outside +/- 3*stddev of the mean.
  # mask_extreme contains the "mask" returned by masked_outside()
  stddev = data.std()
  mean = data.mean()
  mask_extremes = ma.masked_outside(data, mean-3*stddev, mean+3*stddev).mask
  
  # ma.array() applies the mask to data.
  # The compressed() method converts the masked data into a ndarray on
  # which we can run .mean().
  clipped_mean = ma.array(data, mask=mask_extremes).compressed().mean()
  
  # Another common image operation is the filtering of an image.
  # To gaussian filter an image, use scipy.ndimage.filters.gaussian_filter.
  # The filters module offers several other functions for image processing, 
  # see help(filters)
  conv_data = np.zeros(data.size).reshape(data.shape)
  sigma = 10.
  filters.gaussian_filter(data, sigma, output=conv_data)
  display(data, zscale=True)
  display(conv_data, zscale=True)
  
  # If you wanted to put this convoled data back in the AstroData
  # object you would do:
  ad['SCI',2].data = conv_data

The world of ``numpy``, ``scipy``, and the new ``astropy`` is rich and vast.
The reader should refer to those packages' documentation to learn more.

Using the AstroData Data Quality Plane
======================================
.. todo::
   Write examples that use the DQ plane.  Eg. transform DQ plane in a numpy mask
   and do statistics.


Manipulate Data Sections
========================
Sections of the data array can be accessed and processed.  It is important to
note here that when indexing a numpy array, the left most number refers to the
highest dimension's axis (eg. in IRAF sections are in (x,y) format, in Python
they are in (y,x) format). Also important is to remember that the numpy arrays
are 0-indexed, not 1-indexed like in Fortran or IRAF.  For example, in a 2-D 
numpy array, the pixel position (x,y) = (50,75) would be accessed as data[74,49].

Here are some examples using data sections.::

  from astrodata import AstroData
  import numpy as np
  
  ad = AstroData('N20110313S0188.fits')
  data = ad['SCI',2].data
  
  # Let's get statistics for a 25x25 pixel-wide box centered on pixel 50,75.
  mean = data[62:87,37:62].mean()
  median = np.median(data[62:87,37:62])
  stddev = data[62:87,37:62].std()
  minimum = data[62:87,37:62].min()
  maximum = data[62:87,37:62].max()
  print "Mean      Median Stddev       Min    Max\n", mean, median, stddev, minimum, maximum

Now let us apply our knownledge so far to do a quick overscan subtraction.
In this example, we make use of Descriptors, astrodata arithmetic
functions, data sections, numpy 0-based arrays, and numpy statistics function mean().::

  # Get the (EXTNAME,EXTVER)-keyed dictionary for the overscan section and
  # the data section.
  oversec_descriptor = ad.overscan_section().as_dict()
  datasec_descriptor = ad.data_section().as_dict()
  
  # Loop through the extensions. 
  for ext in ad['SCI']:
      extnamever = (ext.extname(),ext.extver())
      (x1, x2, y1, y2) = oversec_descriptor[extnamever]
      (dx1, dx2, dy1, dy2) = datasec_descriptor[extnamever] 
        
      # Measure and subtract the overscan level
      mean_overscan = ad[extnamever].data[y1:y2,x1:x2].mean()
      ad[extnamever].sub(mean_overscan)
      
      # Trim the data to remove the overscan section and keep only
      # the data section.
      ad[extnamever].data = ad[extnamever].data[dy1:dy2,dx1:dx2]


Work on Data Cubes
==================

.. todo::
   write some intro to the data cube section and example
   
::

  from astrodata import AstroData
  from stsci.numdisplay import display
  from pylab import *
  
  adcube = AstroData('gmosifu_cube.fits')
  adcube.info()
  
  # The pixel data is a 3-dimensional numpy array with wavelength is axis 0, and 
  # x,y positions in axis 2 and 1, respectively.  (In the FITS file, wavelength
  # is in axis 3, and x, y are in axis 1 and 2, respectively.)
  adcube.data.shape
  
  # To sum along the wavelength axis
  sum_image = adcube.data.sum(axis=0)
  display(sum_image, zscale=True)
  
  # To plot a 1-D representation of the wavelength axis at pixel position (7,30)
  plot(adcube.data[:,29,6])
  show()
  
  # To plot the same thing using the wavelength values for the x axis of the plot
  # one needs to use the WCS to calculate the pixel to wavelength conversion.
  crval3 = adcube.get_key_value('CRVAL3')
  cdelt3 = adcube.get_key_value('CDELT3')
  spec_length = adcube.data[:,29,6].size
  wavelength = crval3 + arange(spec_length)*cdelt3
  plot(wavelength, adcube.data[:,29,6])
  show()

Plot Data
=========
In Python, the main tool to create plots is ``matplotlib``.  We have used it in the 
previous section on data cubes.  Here we do not aimed at covering all of ``matplotlib``;
the reader should refer to that package's documentation.  Rather we will give a few
examples that might be of use for quick inspection of the data.

::

  from astrodata import AstroData
  from pylab import *
  
  adimg = AstroData('N20110313S0188.fits')
  adspec = AstroData('estgsS20080220S0078.fits')
  
  # Line plot from image.  Row #1044.
  line_index = 1043
  line = adimg['SCI',2].data[line_index, :]
  plot(line)
  show()
  
  # Column plot from image, averaging across 11 pixels around column #327.
  col_index = 326
  width = 5
  col_section = adimg['SCI',2].data[:,col_index-width:col_index+width+1]
  column = col_section.mean(axis=1)
  plot(column)
  show()
  
  # Contour plot for section
  galaxy = adimg['SCI',2].data[1045:1085,695:735]
  contour(galaxy)
  axis('equal')
  show()
  
  # Spectrum in pixel
  plot(adspec['SCI',1].data)
  show()
  
  # Spectrum in wavelength (CRPIX1 = 1)
  crpix1 = adspec['SCI',1].get_key_value('CRPIX1')
  crval1 = adspec['SCI',1].get_key_value('CRVAL1')
  cdelt1 = adspec['SCI',1].get_key_value('CDELT1')
  length = adspec['SCI',1].get_key_value('NAXIS1')
  wavelengths = crval1 + (arange(length)-crpix1+1)*cdelt1
  plot(wavelengths, adspec['SCI',1].data)
  show()
  
