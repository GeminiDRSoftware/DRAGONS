.. cheatsheet

.. _cheatsheet::

*********************
Astrodata Cheat Sheet
*********************

Imports
=======

Import Astrodata and the Gemini instruments configurations.

::

    import astrodata
    import gemini_instruments

Basic read and write operations
===============================

Open a file::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')

Get path and filename::

    >>> ad.path
    '../playdata/N20170609S0154.fits'
    >>> ad.filename
    N20170609S0154.fits

Write to a new file::

    >>> ad.write(fileobj='new154.fits')
    >>> ad.filename
    N20170609S0154.fits

Overwrite the file::

    >>> adnew = astrodata.open('new154.fits')
    >>> adnew.filename
    new154.fits
    >>> adnew.write(clobber=True)

Object structure
================

Description
-----------
The astrodata object is assigned by "tags" that describe the type of data it contains.
The tags are drawn from rules defined in ``gemini_instruments`` and are based on header
information.

When mapping a FITS file, each pixel extension is loaded as a NDData object.  The list
is zero-based.  So FITS extension 1 becomes element 0 of the astrodata object.

In the file below, each astrodata "extension" contains the pixel data, then an error
plane (``.uncertainty``) and a bad pixel mask plane (``.mask``) (VAR and DQ in Gemini
reduced FITS data.) Tables can be attached to an extension, like OBJCAT, or to the astrodata
object globally, like REFCAT. (OBJCAT is a catalogue of the sources detected in the image,
REFCAT is a reference catalog for the area covered by the whole file.)  If other 2D data
needs to be associated with an extension this can also be done, like here with OBJMASK,
a 2D mask matching the sources in the image.

::

    ad = astrodata.open('../playdata/N20170609S0154_varAdded.fits')
    ad.info()
    Filename: ../playdata/N20170609S0154_varAdded.fits
    Tags: ACQUISITION GEMINI GMOS IMAGE NORTH OVERSCAN_SUBTRACTED OVERSCAN_TRIMMED
        PREPARED SIDEREAL
    Pixels Extensions
    Index  Content                  Type              Dimensions     Format
    [ 0]   science                  NDDataRef         (2112, 256)    float32
              .uncertainty          StdDevUncertainty (2112, 256)    float32
              .mask                 ndarray           (2112, 256)    int16
              .OBJCAT               Table             (6, 43)        n/a
              .OBJMASK              ndarray           (2112, 256)    uint8
    [ 1]   science                  NDDataRef         (2112, 256)    float32
              .uncertainty          StdDevUncertainty (2112, 256)    float32
              .mask                 ndarray           (2112, 256)    int16
              .OBJCAT               Table             (8, 43)        n/a
              .OBJMASK              ndarray           (2112, 256)    uint8
    [ 2]   science                  NDDataRef         (2112, 256)    float32
              .uncertainty          StdDevUncertainty (2112, 256)    float32
              .mask                 ndarray           (2112, 256)    int16
              .OBJCAT               Table             (7, 43)        n/a
              .OBJMASK              ndarray           (2112, 256)    uint8
    [ 3]   science                  NDDataRef         (2112, 256)    float32
              .uncertainty          StdDevUncertainty (2112, 256)    float32
              .mask                 ndarray           (2112, 256)    int16
              .OBJCAT               Table             (5, 43)        n/a
              .OBJMASK              ndarray           (2112, 256)    uint8
    Other Extensions
                   Type        Dimensions
    .REFCAT        Table       (245, 16)



Modifying the structure
-----------------------
Append an extension

Delete an extension

Attach a table to an extension

Attach a table to the astrodata object


Astrodata tags
==============

::

    >>> ad = astrodata.open('../playdata/N20170521S0925_forStack.fits')
    >>> ad.tags
    {'GEMINI',
     'GMOS',
     'IMAGE',
     'NORTH',
     'OVERSCAN_SUBTRACTED',
     'OVERSCAN_TRIMMED',
     'PREPARED',
     'PROCESSED_SCIENCE',
     'SIDEREAL'}
    >>> type(ad.tags)
    set

    >>> {'IMAGE', 'PREPARED'}.issubset(ad.tags)
    True
    >>> 'PREPARED' in ad.tags
    True


Headers
=======

Descriptors
-----------

Full headers
------------

All headers, PHU plus pixel extensions.

Primary Header Unit

Direct access to header keywords
--------------------------------

::

    >>> ad = astrodata.open('../playdata/N20170521S0925_forStack.fits')

Primary Header Unit
*******************
Get value from PHU::

    >>> ad.phu.EXPTIME
    440.0

Set PHU keyword, with or without comment::

    >>> ad.phu.NEWKEY = 50.
    >>> ad.phu.ANOTHER = (30. 'Some comment')

Delete PHU keyword::

    >>> del ad.phu.NEWKEY


Pixel extension header
**********************

Table header
************


Pixel data
==========

Arithmetics
-----------

Other pixel data operations
---------------------------
(e.g average with numpy, using the mask, take data - do something - put it back(?) )

Tables
======

Create new astrodata object
===========================
