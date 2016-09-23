.. structure:

.. _structure:

*********************
MEF Structure Mapping
*********************

.. warning::
   Structures are not fully implemented in astrodata.

.. todo::
   The structure section needs to be written, but since the functionality
   is not available, other than the SCI,VAR,DQ model, this is not a priority.

File Structure Definitions
==========================

Gemini Data Structure
=====================

Raw Gemini data are Multi-Extension FITS that contains only the scientific
pixel data.  The extensions are not named (EXTNAME) or versioned (EXTVER).

When the raw data is "prepared", the science extensions are named "SCI"
(ie. EXTNAME keyword set to "SCI") and versioned with an index (ie.
EXTVER keyword set to 1 ... N, N being the number of extensions the MEF
contains).  

As early as possible, the variance plane and the data quality plane will
be calculated and added.  Those planes are named "VAR" and "DQ", and 
receive an EXTVER value corresponding to their respective "SCI" extension.

For spectroscopic observation, a Mask Definition File, a FITS table, is attached to the
data and the extension is named "MDF", with version number 1.

Those version names are implied when an AstroData object is created.



Using Structures
================

Adding New Structure
====================
