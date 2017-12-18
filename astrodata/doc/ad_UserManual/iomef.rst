.. iomef.rst

.. _iomef:

************************************************************
Input and Output Operations and Extension Manipulation - MEF
************************************************************

AstroData is not intended to be Multi-Extension FITS (MEF) centric.  The core
is independent of the file format.  At Gemini, our data model uses MEF.
Therefore we have implemented a FITS handler that maps a MEF to the
internal AstroData representation.

In this chapter, we present examples that will help the reader understand how
to access the information stored in a MEF with the AstroData object and
understand that mapping.

**Try it yourself**

Download the "playground" data if you wish to follow along and run the
examples.  Then ::

    $ cd <path>/gemini_python_datapkg-v2.0/playground
    $ python

Imports
=======
Before doing anything, you need to import AstroData and the Gemini instrument
configuration package.

::

    import astrodata
    import gemini_instruments


Open and access existing dataset
================================

Read in the dataset
-------------------
The file on disk is loaded into the AstroData class associated with the
instrument the data is from. This association is done automatically based on
header content.

::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    >>> type(ad)
    gemini_instruments.gmos.adclass.AstroDataGmos

From now on, ``ad`` knows it is GMOS data.  It knows how to access it's headers
and when using the Recipe System (``recipe_system``), it will trigger the
selection of the GMOS primitives and recipes.

The original path and filename are stored in the object. If you were to write
the AstroData object to disk without specifying anything, those path and
filename would be used. ::

    >>> ad.path
    '../playdata/N20170609S0154.fits'
    >>> ad.filename
    'N20170609S0154.fits'


Accessing the content of a MEF file
-----------------------------------
Accessing pixel data, headers, tables will be covered in details in following
chapters.  Here we just introduce the basic content interfaces.

For details on the AstroData structure, please refer to the previous chapter.

AstroData uses NDData as the core of its structure.  Each FITS extension
becomes a NDData object and is added to a list.

To access pixel data, the list index and the `.data` attribute are used.  That
returns a ``numpy.ndarray``. The list of NDData is zero-indexed.  *Extension
number 1 in a MEF is index 0 in an AstroData object*. ::

    >>> ad = astrodata.open('../playdata/N20170609S0154_varAdded.fits')
    >>> data = ad[0].data
    >>> type(data)
    numpy.ndarray
    >>> data.shape
    (2112, 256)

Remember that in Python the y-axis is the first number in a ndarray.

The variance and data quality planes, the VAR and DQ planes in Gemini MEF
files, are represented by the ``variance`` and ``mask`` attributes,
respectively.  They are not their own "extension", they don't have their
own index in the list, unlike in a MEF.  They are attached to the data pixel,
packaged together by the NDData object. ::

    >>> var = ad[0].variance
    >>> dq = ad[0].mask

Tables in the MEF will also be loaded into the AstroData object.  If a table
is associated with a specific science extension through the EXTVER header, that
table will be packaged within the same AstroData extension as the pixel data.
The AstroData extension is the NDData object plus any table or other pixel
array.  If the table is not associated with a specific extension and applies
globally, it will be added to the AstroData object as a global addition.  No
indexing will be required to access it.

The tables are stored internally as astropy ``Table`` objects. ::

    >>> ad[0].OBJCAT
    <Table length=6>
    NUMBER X_IMAGE Y_IMAGE ... REF_MAG_ERR PROFILE_FWHM PROFILE_EE50
    int32  float32 float32 ...   float32     float32      float32
    ------ ------- ------- ... ----------- ------------ ------------
         1 283.461 55.4393 ...     0.16895       -999.0       -999.0
    ...
    >>> type(ad[0].OBJCAT)
    astropy.table.table.Table

    >>> refcat = ad.REFCAT
    >>> type(refcat)
    astropy.table.table.Table


Headers are stored as Python dictionaries.  Headers associated with extensions
are stored with the corresponding AstroData extension.  The MEF Primary Header
Unit (PHU) is stored "globally".  Note that when slicing an AstroData object,
for example copying over just the first extension, the PHU will follow.
Headers can be accessed directly or for some predefined concepts, Descriptors
can be used.  See the chapters on headers for details.

Using Descriptors::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    >>> ad.filter_name()
    'open1-6&g_G0301'
    >>> ad.filter_name(pretty=True)
    'g'

Using directy header access::

    >>> ad.phu['FILTER1']
    'open1-6'
    >>> ad.phu['FILTER2']
    'g_G0301'

Accessing the extension headers::

    >>> ad.hdr['CCDSEC']
    ['[1:512,1:4224]',
     '[513:1024,1:4224]',
     '[1025:1536,1:4224]',
     '[1537:2048,1:4224]']
    >>> ad[0].hdr['CCDSEC']
    '[1:512,1:4224]'

    With descriptors:
    >>> ad.array_section(pretty=True)
    ['[1:512,1:4224]',
     '[513:1024,1:4224]',
     '[1025:1536,1:4224]',
     '[1537:2048,1:4224]']


Modify Existing MEF Files
=========================
Before you start modify the structure of an AstroData object, you should be
familiar with it.  Please make sure that you have read the previous chapter
on Structure.

Appending an extension
----------------------
In this section, we take an extension from one AstroData object and append it
to another.  Because we are mapping a FITS file, the EXTVER keyword gets
automatically updated to the next available value to ensure that when the
AstroData object is written back to disk as MEF, it will be coherent.

Here is an example appending a whole AstroData extension, with pixel data,
variance, mask and tables.

::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    >>> advar = astrodata.open('../playdata/N20170609S0154_varAdded.fits')

    >>> ad.info()
    Filename: ../playdata/N20170609S0154.fits
    Tags: ACQUISITION GEMINI GMOS IMAGE NORTH RAW SIDEREAL UNPREPARED
    Pixels Extensions
    Index  Content                  Type              Dimensions     Format
    [ 0]   science                  NDDataRef         (2112, 288)    uint16
    [ 1]   science                  NDDataRef         (2112, 288)    uint16
    [ 2]   science                  NDDataRef         (2112, 288)    uint16
    [ 3]   science                  NDDataRef         (2112, 288)    uint16

    >>> ad.append(advar[3])
    >>> ad.info()
    Filename: ../playdata/N20170609S0154.fits
    Tags: ACQUISITION GEMINI GMOS IMAGE NORTH RAW SIDEREAL UNPREPARED
    Pixels Extensions
    Index  Content                  Type              Dimensions     Format
    [ 0]   science                  NDDataRef         (2112, 288)    uint16
    [ 1]   science                  NDDataRef         (2112, 288)    uint16
    [ 2]   science                  NDDataRef         (2112, 288)    uint16
    [ 3]   science                  NDDataRef         (2112, 288)    uint16
    [ 4]   science                  NDDataRef         (2112, 256)    float32
              .variance             ndarray           (2112, 256)    float32
              .mask                 ndarray           (2112, 256)    int16
              .OBJCAT               Table             (5, 43)        n/a
              .OBJMASK              ndarray           (2112, 256)    uint8

    >>> ad[4].hdr['EXTVER']
    5
    >>> advar[3].hdr['EXTVER']
    4

As you can see above, the fourth extension of ``advar``, along with everything
it contains was appended at the end of the first AstroData object.  Also, note
that the EXTVER of the extension while in ``advar`` was 4, but once appended
to ``ad``, it had to be changed to the next available integer, 5, numbers 1 to
4 being already used by ``ad``'s own extensions.

And here we are appending only the pixel data, leaving behind the other
associated data.  The header associated with that data does follow however.

::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    >>> advar = astrodata.open('../playdata/N20170609S0154_varAdded.fits')

    >>> ad.append(advar[3].data)
    >>> ad.info()
    Filename: ../playdata/N20170609S0154.fits
    Tags: ACQUISITION GEMINI GMOS IMAGE NORTH RAW SIDEREAL UNPREPARED
    Pixels Extensions
    Index  Content                  Type              Dimensions     Format
    [ 0]   science                  NDDataRef         (2112, 288)    uint16
    [ 1]   science                  NDDataRef         (2112, 288)    uint16
    [ 2]   science                  NDDataRef         (2112, 288)    uint16
    [ 3]   science                  NDDataRef         (2112, 288)    uint16
    [ 4]   science                  NDDataRef         (2112, 256)    float32

Notice how a new extension was created but ``variance``, ``mask``, the OBJCAT
table and OBJMASK image were not copied over.  Only the science pixel data was
copied over.

(Please note, there is no "inserting" of extension implemented.)

Removing an extension
---------------------



Writing back to disk
====================

Writing to a new file
---------------------

Updating the existing file on disk
----------------------------------


Create New MEF Files
====================

Create New Copy of MEF Files
----------------------------

Basic example
^^^^^^^^^^^^^

Needing true copies in memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create New MEF Files from Scratch
---------------------------------




