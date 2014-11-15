.. headers:

.. _headers:

************
FITS Headers
************

**Try it yourself**


If you wish to follow along and try the commands yourself, download
the data package, go to the ``playground`` directory and copy over
the necessary file.

::

   cd <path>/gemini_python_datapkg-X1/playground
   cp ../data_for_ad_user_manual/N20111124S0203.fits .

Then launch the Python shell::

   python


.. highlight:: python
   :linenothreshold: 5


AstroData Descriptors
=====================
AstroData Descriptors provide a "header keyword-to-concept" mapping that 
allows one to access header information in a consistent manner, regardless 
of which instrument the dataset is from.  Like for the AstroDataTypes, the 
mapping is coded in a configuration package that is provided by the 
observatory or the user.  

For example, if one were interested to know the filter used for an 
observation, normally one would need to know which specific keyword or set of 
keywords to look at for that instrument.  However, once the concept of 
"filter" is coded in a Descriptor, one now only needs to call the 
``filtername`` Descriptor to retrieve the information.

The Descriptors are closely associated with AstroDataTypes.  The AstroDataType
of the AstroData object will tell the Descriptor system which piece of the
configuration package code to call to retrieve the information requested.

.. note::
   If the Descriptors have not been configured for a dataset's specific 
   AstroDataType, or if the AstroDataType for the data had not been defined, 
   in the configuration package, the Descriptors will not work.  In that case, 
   it is nevertheless possible to access AstroData header information directly
   with the pyfits interface.  This is also shown later in this chapter.

To get the list of descriptors available for an AstroData object::

   from astrodata import AstroData
   
   ad = AstroData('N20111124S0203.fits')
   ad.descriptors
   sorted(ad.descriptors.keys())

The ``descriptors`` property returns a dictionary with the name of the
descriptors available for this AstroData object as the keys.  The really
interesting information is in the keys.  On Line 5, we retrieve the
sorted list of available Descriptors.

Most Descriptor names are readily understood, but one can get a short 
description of what the Descriptor refers to by calling the Python help 
function, for example::

   help(ad.airmass)
  
Descriptors associated with standard FITS keywords are defined in the 
``ADCONFIG_FITS`` package found in ``astrodata_FITS``.  All the Descriptors 
associated with other concepts used by the Gemini software are found in the 
``ADCONFIG_Gemini`` package, part of ``astrodata_Gemini``.

A user reducing Gemini data or coding for existing Gemini data only need to 
make sure that ``astrodata_FITS`` and ``astrodata_Gemini`` have been 
installed.  A user coding for a new Gemini instrument, or for another 
observatory, will need to write the configuration code for the new Descriptors 
and AstrodataTypes.  That is an advanced topic not covered by this manual.

Accessing Headers
=================

Accessing headers with Descriptors
----------------------------------

Whenever possible the Descriptors should be used to get information from the 
headers.  This allows for maximum re-use of the code as it will work on any 
datasets with an AstroDataTypes. Here are a few examples using Descriptors::

   from astrodata import AstroData
   import numpy as np
   
   ad = AstroData('N20111124S0203.fits')
   
   #--- print a value
   print 'The airmass is : ',ad.airmass()
   
   #--- use a value to control the flow
   if ad.exposure_time() < 240.:
      print 'This is a short exposure'
   else:
      print 'This is a long exposure'
   
   #--- multiply all extensions by their respective gain
   print 'The average before: ', np.average(ad['SCI',2].data)
   print 'The gain for [SCI,2]', ad['SCI',2].gain()
   
   ad.mult(ad.gain())
   
   print 'The average after: ', np.average(ad['SCI',2].data)
   
   #--- do arithmetics
   fwhm_pixel = 3.5
   fwhm_arcsec = fwhm_pixel * ad.pixel_scale()
   
   print 'The FWHM in arcsec is: ', fwhm_arcsec
   
The Descriptors are returned as DescriptorValue objects. In many types of 
statements the DescriptorValue will be automatically converted to the 
appropriate Python type based on context.  All the Descriptor calls above
do that convertion.  For example, on Line 25, the DescriptorValue returned
by ``pixel_scale()`` is converted to a Python float for the multiplication.

The AstroData arithmetics used on Line 19 will be discussed in more details
in a later chapter.  Essentially, the ``gain()`` Descriptor returns a 
DescriptorValue with gain information for each of the 6 extensions.  Then
AstroData's ``mult`` method multiplies each pixel data extensions with the
appropriate gain value for that extension.  No looping necessary, AstroData
and Descriptors are taking care of it.

When the automatic convertion to a Python cannot be determine from context
the programmer must use the method ``as_pytype()``. ::

   ad.pixel_scale()
   ad.pixel_scale().as_pytype()

The first line returns a DescriptorValue, the second line returns a float.


Accessing headers directly
--------------------------

Not all the header content has been mapped with Descriptors, nor should it.  
The header content is nevertheless accessible.  With direct access, there 
are no DescriptorValue involved and the type returned matches what is stored 
in the header.

One important thing to keep in mind is that the PHU and the extension headers 
are accessed differently. The method ``phu_get_key_value`` accesses the PHU
header; the method ``get_key_value`` accesses the header of the specified
extension.

Here are some direct access examples::

   from astrodata import AstroData
   
   ad = AstroData('N20111124S0203.fits')
   
   # Get keyword value from the PHU
   aofold_position = ad.phu_get_key_value('AOFOLD')
   
   # Get keyword value from a specific extension
   print ad['SCI',1].get_key_value('NAXIS2')
   
   # Get keyword value for all SCI extensions
   for extension in ad['SCI']:
      naxis2 = extension.get_key_value('NAXIS2')
      print naxis2


Whole headers
-------------
Entire headers can be retrieve as PyFITS Header object.

::

   # Get the header for the PHU
   phuhdr = ad.phu.header
   
   # Get the header for extension SCI, 1
   exthdr = ad['SCI',1].header

In the interactive Python shell, listing the header contents to screen can be
done as follow::
   
   # For the PHU
   ad.phu.header
   
   # For a specific extension:
   ad['SCI',2].header
   
   # For all the extensions:  (PHU excluded)
   ad.headers


EXTNAME and EXTVER
------------------
MEF files have the concept of naming and versioning extensions.  The header 
keywords storing the name and version are ``EXTNAME`` and ``EXTVER``.  AstroData uses
that concept extensively.  In fact, even if a MEF on disk does not have 
``EXTNAME`` and ``EXTVER`` defined, for example Gemini raw datasets, upon
opening the file AstroData will assign names and versions to each extension.
The default behavior is to assign all extension a ``EXTNAME`` of ``SCI``
and then version them sequential from 1 to the number of extension present.

The name and version of an extension is obtained this way::

   name = ad[1].extname()
   version = ad[1].extver()
   print name, version



Updating and Adding Headers
===========================

Header cards can be updated or added to the headers.  As for the simple access 
to the headers, there are methods to work on the PHU and different methods to
work on the extensions.

The methods to update and add headers mirror the access methods.  The method
``phu_set_key_value()`` modifies the PHU.  The method ``set_key_value()`` 
modifies the extension headers.

The inputs to the ``phu_set_key_value`` and ``set_key_value`` methods are
*keyword*, *value*, *comment*.  The comment is optional. 

::

   from astrodata import AstroData
   
   ad = AstroData('N20111124S0203.fits')
   
   # Add a header card to the PHU
   ad.phu_set_key_value('MYTEST', 99, 'Some meaningless keyword')
   
   # Modify a header card in the second extension
   ad[1].set_key_value('GAIN',5.)
   
   # The extension can also be specified by name and version.
   ad['SCI',2].set_key_value('GAIN', 10.)
   
   # In a loop 
   for extension in ad['SCI']:
      extension.set_key_value('TEST',9, 'This is a test.')

The ``set_key_value`` method works only on an extension, it will not work
on the whole AstroData object. For example, the following **will not** 
work::

   # DOES NOT WORK.  An extension must be specified.
   ad.set_key_value('TEST', 8, 'This test does not work')

EXTNAME and EXTVER
------------------

The name and version of an extension are so critical to AstroData that, like
for access, the editing of ``EXTNAME`` and ``EXTVER`` is done through a special
method.

The name and version of an extension can be set or reset manually with the 
``rename_ext`` method::

   ad['SCI',1].rename_ext('VAR',4)

Be careful with this function.  Having two extensions with the same name and
version in an AstroData data object, or a MEF files for that matter, can lead
to strange problems.


Adding Descriptors for New Instruments [Advanced Topic]
=======================================================

.. todo::
   Primer on Descriptor definitions for new instrument.

.. note::
   refer to Descriptors document for complete instructions
