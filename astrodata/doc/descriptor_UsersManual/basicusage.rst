.. basicusage:

.. _Basic_Descriptor_Usage:

**********************
Basic Descriptor Usage
**********************

The command ``typewalk -l`` lists all the descriptors that are defined. As of
the date of this document there are 73 descriptors defined (:ref:`Appendix A
<Appendix_typewalk>`). 

The following commands show an example of how to use descriptors. They can be 
entered at an interactive Python prompt (e.g., ``python``, ``ipython``,
``pyraf``)::

  >>> from astrodata import AstroData
  # Load the fits file into AstroData
  >>> ad = AstroData("N20091027S0137.fits")
  # Count the number of pixel data extensions in the AstroData object
  >>> ad.count_exts()
  3
  # Count the number of science extensions in the AstroData object
  >>> ad.count_exts(extname="SCI")
  3
  # Get the airmass value using the airmass descriptor
  >>> airmass = ad.airmass()
  >>> print airmass
  1.327
  # Get the instrument name using the instrument descriptor
  >>> print "My instrument is %s" % ad.instrument()
  My instrument is GMOS-N
  # Get the gain value for each pixel data extension
  >>> for ext in ad:
  ...     print ext.gain()
  ... 
  2.1
  2.337
  2.3
  >>> print ad.gain()
  {('SCI', 2): 2.3370000000000002, ('SCI', 1): 2.1000000000000001, 
  ('SCI', 3): 2.2999999999999998}

In the examples above, the airmass and instrument apply to the dataset as a
whole i.e., the keywords themselves exist only in the Primary Header Unit
(PHU) and so only one value is returned. However, the gain applies specifically
to the pixel data extensions within the dataset and so for this AstroData
object, since there are three pixel data extensions, three values are returned
in the form of a Python dictionary, where the key of the dictionary is the
("``EXTNAME``", ``EXTVER``) tuple.

For those descriptors that describe a concept applying specifically to the
pixel data extensions within a dataset i.e., those that access keywords in the
headers of the pixel data extensions, a value for every pixel data extension in
the AstroData object is returned by default::

  >>> new_ad = AstroData("stgsS20100223S0042.fits")
  # Count the number of pixel data extensions in the AstroData object
  >>> new_ad.count_exts()
  103
  # Count the number of science extensions in the AstroData object
  >>> new_ad.count_exts(extname="SCI")
  34
  # Get the data section value for each pixel data extension in the form of a
  # dictionary 
  >>> print new_ad.data_section()
  {('SCI', 29): [0, 3108, 0, 22], ('DQ', 5): [0, 3108, 0, 21], 
  ('DQ', 9): [0, 3108, 0, 21], ('SCI', 15): [0, 3108, 0, 22], 
  ('SCI', 24): [0, 3108, 0, 21], ('VAR', 8): [0, 3108, 0, 21], 
  ...

Descriptors can also be used to return values relating to a subset of pixel
data extensions, i.e., those associated with a particular ``EXTNAME``::

  >>> new_ad.count_exts(extname="DQ")
  34
  # Get the data section value for the data quality extensions only
  >>> print new_ad["DQ"].data_section()
  {('DQ', 11): [0, 3108, 0, 21], ('DQ', 6): [0, 3108, 0, 22], 
  ('DQ', 29): [0, 3108, 0, 22], ('DQ', 9): [0, 3108, 0, 21], 
  ('DQ', 4): [0, 3108, 0, 22], ('DQ', 19): [0, 3108, 0, 21], 
  ...

Note that it is quicker to obtain the values of a descriptor in the form of a
dictionary than it is to obtain the values of a descriptor for each extension
separately.
