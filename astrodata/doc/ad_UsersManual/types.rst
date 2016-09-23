.. types:

.. _types:

**************
AstroDataTypes
**************


What are AstroDataTypes
=======================

AstroDataTypes are AstroData's way to know about itself.  When a file is 
opened with AstroData, the headers are inspected, data identification rules 
are applied, and all applicable AstroDataTypes are assigned.  From that point 
on, the AstroData object "knows" whether it is a GMOS image, a NIRI spectrum, 
an IFU from GMOS or NIFS. This embedded knowledge is critical to the header 
keyword mapping done by the Descriptors (see 
:ref:`Section 4 - FITS Headers <headers>`), for examples.  
The RecipeSystem also depends heavily on the AstroDataType feature.

Examples of AstroDataTypes are: GMOS_IMAGE, SIDEREAL, GMOS_IFU_FLAT, NIRI_CAL,
NIRI_SPECT, GEMINI_SOUTH, etc.

The AstroDataTypes can also refer to data processing status, eg. RAW.  This 
feature is not used as much yet.

The types are obviously observatory and instrument dependent.  The 
identification rules do need to be coded for AstroData to assign 
AstroDataTypes.  This has been done for most if not all Gemini data.  The 
Gemini Types are included in the ``astrodata_Gemini`` package that is 
installed along with ``astrodata`` when ``gemini_python`` is installed.   
Keeping the instrument rules and configuration separate from ``astrodata`` 
keeps it generic, and allows other packages to be easily added, for example, 
one might want to add a third-party package for CFHT instruments. 



Using AstroDataTypes
====================

**Try it yourself**


If you wish to follow along and try the commands yourself, download
the data package, go to the ``playground`` directory and copy over
the necessary files.

::

   cd <path>/gemini_python_datapkg-X1/playground
   cp ../data_for_ad_user_manual/N20111124S0203.fits .

Then launch the Python shell::

   python


.. highlight:: python
   :linenothreshold: 5

The attribute ``types`` is a common way to check the AstroDataType and make
logic decisions based to on the type.

::

   from astrodata import AstroData
   
   ad = AstroData('N20111124S0203.fits')
   
   if 'GMOS_IMAGE' in ad.types:
      print "I am a GMOS Image."
   else:
      print "I am these types instead: ", ad.types

The attribute ``types`` is a list of all the AstroDataTypes 
associated with the dataset.  Other than for controlling the flow of the
program, it can be useful when interactively exploring the various types 
associated with a dataset, or when there's a need to write all the types 
to the screen or to a file, for logging purposes, for example.

If you have a set of Gemini datasets on disk and you wish to know which 
AstroDataTypes are associated with them, use the shell tool ``typewalk``
in that directory and you will be served a complete list of types for each
datasets in that directory.  Try it in the ``data_for_ad_user_manual`` 
directory.  Open another shell (not the interactive Python shell) and ::

   cd <path>/gemini_python_datapkg-X1/data_for_ad_user_manual
   typewalk

You will get::

   directory: /data/giraf/gemini_python_datapkg-X1/data_for_ad_user_manual
        estgsS20080220S0078.fits .......... (GEMINI) (GEMINI_SOUTH) (GMOS) 
        ................................... (GMOS_LS) (GMOS_S) (GMOS_SPECT) (LS) 
        ................................... (PREPARED) (SIDEREAL) (SPECT) 
        gmosifu_cube.fits ................. (GEMINI) (GEMINI_SOUTH) (GMOS) 
        ................................... (GMOS_IFU) (GMOS_IFU_BLUE) 
        ................................... (GMOS_IFU_RED) (GMOS_IFU_TWO) 
        ................................... (GMOS_S) (GMOS_SPECT) (IFU) 
        ................................... (PREPARED) (SIDEREAL) (SPECT) 
        N20110313S0188.fits ............... (GEMINI) (GEMINI_NORTH) (GMOS) 
        ................................... (GMOS_IMAGE) (GMOS_N) (GMOS_RAW) 
        ................................... (IMAGE) (RAW) (SIDEREAL) (UNPREPARED) 
        N20110316S0321.fits ............... (CAL) (GEMINI) (GEMINI_NORTH) (GMOS) 
        ................................... (GMOS_CAL) (GMOS_IMAGE) 
        ................................... (GMOS_IMAGE_FLAT) 
        ................................... (GMOS_IMAGE_TWILIGHT) (GMOS_N) 
        ................................... (GMOS_RAW) (IMAGE) (RAW) (SIDEREAL) 
        ................................... (UNPREPARED) 
        N20111124S0203.fits ............... (CAL) (GEMINI) (GEMINI_NORTH) (GMOS) 
        ................................... (GMOS_CAL) (GMOS_IFU) (GMOS_IFU_FLAT) 
        ................................... (GMOS_IFU_RED) (GMOS_N) (GMOS_RAW) 
        ................................... (GMOS_SPECT) (IFU) (RAW) (SIDEREAL) 
        ................................... (SPECT) (UNPREPARED) 


The attribute ``types`` returns all the processing status flags as well as
the types proper.  Those two concepts can be separated.  The method ``type()``
returns only types, no status; the method ``status()`` returns only processing
status, no types.

::

   ad.type()
   ad.status()

The ``type()`` statement returns::

   ['GMOS_IFU', 'GMOS_IFU_RED', 'IFU', 'GEMINI_NORTH', 'GMOS_N', 
   'GMOS_IFU_FLAT', 'GMOS_CAL', 'GEMINI', 'SIDEREAL', 'GMOS_SPECT', 
   'CAL', 'GMOS', 'SPECT']

and the ``status()`` statement returns::

   ['GMOS_RAW', 'UNPREPARED', 'RAW']


If code applies modifications to the AstroData object that result in changes to 
the AstroDataTypes, it is necessary to let the system know about it.  The 
method ``refresh_types()`` rescan the AstroData headers and reapply the
identification rules.  This type refreshing is used mostly when the processing
status needs to be changed, for example once the raw data has been 
standardized, it's processing status becomes "PREPARED".::

   ad.refresh_types()


Creating New AstroDataTypes [Advanced Topic]
============================================

.. todo::
   Primer on creating new AstroDataTypes.

.. note::
   refer to programmer's manual, but give some idea of what needs to be done
   and the basic principles
