.. types:

.. _types:

**************
AstroDataTypes
**************

What are AstroDataTypes
=======================

.. todo::
   write a straightforward explanation of what AstroDataTypes are.

.. note::
   For the TODO: explain what they are. data type & data processing status. 
   classification based on headers only.  explain how to install the Gemini types.)

Using AstroDataTypes
====================

There are two ways to check the AstroDataTypes of a dataset::

  from astrodata import AstroData
  
  ad = AstroData('N20111124S0203.fits')
  
  if 'GMOS_IMAGING' in ad.types:
      # do special steps for GMOS_IMAGING type data

The attribute ``ad.types`` returns a list of all the AstroDataTypes associated with the dataset.  
It can be useful when interactively exploring the various types associated with a dataset, or
when there's a need to write all the types to the screen or to a file, for logging purposes, for example.

.. warning::
   We are in the process of improving the AstroData API.  Currently, 
   "Data Types" are referred to as *Typology* in the AstroDataTypes code.  "Data Processing Status" 
   are referred to as *Status*.  There are two additional attributes that might be useful if those 
   two concepts need to be addressed separately:  ``ad.typesStatus`` and ``ad.typesTypology``.  
   They are used exactly the same way as ``ad.types``.
   
   WARNING:  *Do not use typesStatus or typesTypology*, we getting rid of them and sorting the mess.
   
If a primitive changes the AstroDataTypes, it is necessary to let the system know about it.
The method ``refresh_types()`` takes care of that.  This is used mostly when the processing
status needs to be changed, for example once the raw data has been standardized, it's processing
status becomes "PREPARED".  This is an important processing status other primitives will check.::

   ad.refresh_types()


Creating New AstroDataTypes
===========================

.. todo::
   Primer on creating new AstroDataTypes.

.. note::
   refer to programmer's manual, but give some idea of what needs to be done
   and the basic principles
