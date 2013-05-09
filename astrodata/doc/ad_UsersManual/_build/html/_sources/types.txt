.. types:

**************
AstroDataTypes
**************

What are AstroDataTypes
=======================

(explain what they are. data type & data processing status. classification based on headers only. 
explain how to install the Gemini types.)

Using AstroDataTypes
====================

There are two ways to check the AstroDataTypes of a dataset::

  from astrodata import AstroData
  
  ad = AstroData('N20111124S0203.fits')
  
  if ad.is_type('GMOS_IMAGING'):
      # do special steps for GMOS_IMAGING type data
  
  if 'GMOS_IMAGING' in ad.types:
      # do special steps for GMOS_IMAGING type data

The attribute ``ad.types`` returns a list of all the AstroDataTypes associated with the dataset.  
It can be useful when interactively exploring the various types associated with a dataset, or
when there's a need to write all the types to the screen or to a file, for logging purposes, for example.
Use at your discretion based on your need.

"Data Types" are referred to as *Typology* in the AstroDataTypes code.  "Data Processing Status" are
referred to as *Status*.  There are two additional attributes that might be useful if those two
concepts need to be addressed separately:  ``ad.typesStatus`` and ``ad.typesTypology``.  They
are used exactly the same way as ``ad.types``.


??? ad.refresh_types()


Creating New AstroDataTypes
===========================

(refer to programmer's manual, but give some idea of what needs to be done
and the basic principles)
