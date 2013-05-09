.. headers:

************
FITS Headers
************

AstroData Descriptors
=====================
AstroData Descriptors provide a "header keyword-to-concept" mapping that allows one to 
access header information in a consistent manner, regardless of which instrument the 
dataset is from.  The mapping is coded in a configuration package that is provided 
by the observatory or the user.

For example, if one were interested to know the filter used for an observation, normally
one would need to know which specific keyword or set of keywords to look at.  Once the
concept of "filter" is coded in a Descriptor, one now only needs to call the ``filtername``
Descriptor.

To get the list of descriptors available for an AstroData object::

  from astrodata import AstroData
  
  ad = AstroData('N20111124S0203.fits')
  ad.all_descriptor_names()

Most Descriptor names are readily understood, but one can get a short description of
what the Descriptor refers to by call the Python help function, for example::

  help(ad.airmass)
  
Descriptors associated with standard FITS keywords are available from the ``ADCONFIG_FITS`` package
distributed in ``astrodata_FITS``.  All the Descriptors associated with other concepts used by
the Gemini software are found in the ``ADCONFIG_Gemini`` package, part of ``astrodata_Gemini``.

As a user reducing Gemini data or coding for existing Gemini data, all you need to do is make 
sure that astrodata_FITS and astrodata_Gemini have been installed.  If you are coding for a new
Gemini instrument, or for another observatory, Descriptors and AstrodataTypes will need to be
coded.  That's a more advanced topic addressed elsewhere. (KL?? ref to last section of this page) 

Accessing Headers
=================

Whenever possible the Descriptors should be used to get information from the headers.  This 
allows for maximum re-use of the code as it will work on any datasets with an AstroDataTypes.
Here are a few examples using Descriptors::

  from astrodata import AstroData
  from copy import deepcopy
  
  ad = AstroData('N20111124S0203.fits')
  adcopy = deepcopy(ad)
  
  print 'The airmass is : ',ad.airmass()
  
  if ad.exposure_time() < 240.:
    print 'This is a short exposure'
    
  # This call will multiply the pixel values in all three science extensions
  # by their respective gain.  There's no need to loop through the science
  # extension explicitly.
  adcopy.mult(adcopy.gain())
  
  fhwm_arcsec = 3.5 * ad.pixel_scale()


Of course not all the header content has been mapped with Descriptors.  Here is how
to get the value of a specific header keyword::

  from astrodata import AstroData
  
  ad = AstroData('N20111124S0203.fits')
  
  # Get keyword value from the PHU
  aofold_position = ad.phu_get_key_value('AOFOLD')
  
  # Get keyword value from a specific extension
  naxis2 = ad.ext_get_key_value(('SCI',1), 'NAXIS2')
  
  # Get keyword value from an extension when there's only one extension
  # This happens, for example, when looping through multiple extensions.
  for extension in ad['SCI']:
     naxis2 = extension.get_key_value('NAXIS2')
     print naxis2

Multi-extension FITS files, MEF, have this concept of naming and versioning the extensions.
The header keywords controlling name and version are ``EXTNAME`` and ``EXTVER``.  AstroData
uses that concept extensively.  See ??? for information on the typical structure of AstroData
objects.  The name and version of an extension is obtained this way::

  name = ad[1].extname()
  version = ad[1].extver()
  print name, version
  
To get a whole header from an AstroData object, one would do::

  # Get the header for the PHU as a pyfits Header object 
  phuhdr = ad.phu.header
  
  # Get the header for extension SCI, 1 as a pyfits Header object
  exthdr = ad['SCI',1].header
  
  # print the header content in the interactive shell
  # For a specific extension:
  ad['SCI',2].header
  # For all the extensions:  (PHU excluded)
  ad.get_headers()
  
  ad.close()

Updating and Adding Headers
===========================

Header cards can be updated or added to header.  As for the access to the headers, the PHU
have their own methods, different from the extension, but essentially doing the same thing.
To write to a PHU use the ``phu_set_key_value()`` method.  To write to the header of an 
extension, use the ``ext_set_key_values()``.  The difference is that one has to specify the
extension ID in the latter case. ::

  from astrodata import AstroData
  
  ad = AstroData('N20111124S0203.fits')
  
  # Add a header card to the PHU
  #  The arguments are *keyword*, *value*, *comment*.  The comment is optional.
  ad.phu_set_key_value('MYTEST', 99, 'Some meaningless keyword')
  
  # Modify a header card in the second extension
  #  The arguments are *extension*, *keyword*, *value*, *comment*.  The comment 
  #  is optional.  If a comment already exists, it will be left untouched.
  ad.ext_set_key_value(1,'GAIN',5.)
  
  # The extension can also be specified by name and version.
  ad.ext_set_key_value(('SCI',2), 'GAIN', 10.)
  
  # A utility method also exists for use in astrodata objects that contain
  # only one extension.  This is particularly useful when looping through
  # the extensions.  There's no need to specify the extension number since 
  # there's only one.  The arguments are *keyword*, *value*, *comment*, with
  # comment being optional.
  for extension in ad['SCI']:
      extension.set_key_value('TEST',9, 'This is a test.')
  
The name and version of an extension can be set or reset manually with the 
``rename_ext`` method::

  ad['SCI',1].rename_ext('VAR',4)

Be careful with this function.  Having two extensions with the same name and
version in an AstroData data object, or a MEF files for that matter, can lead
to strange problems.


Adding Descriptors Definitions for New Instruments
==================================================

(refer to Emma's document.)
