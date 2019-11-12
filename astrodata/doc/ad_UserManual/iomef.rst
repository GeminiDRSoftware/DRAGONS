.. iomef.rst

.. include:: references.txt

.. _iomef:

************************************************************
Input and Output Operations and Extension Manipulation - MEF
************************************************************

|AstroData| is not intended to be Multi-Extension FITS (MEF) centric. The core
is independent of the file format. At Gemini, our data model uses MEF.
Therefore we have implemented a FITS handler that maps a MEF to the
internal |AstroData| representation. A different handler can be implemented
for a different file format.

In this chapter, we present examples that will help the reader understand how
to access the information stored in a MEF with the |AstroData| object and
understand that mapping.

**Try it yourself**

Download the data package (:ref:`datapkg`) if you wish to follow along and run the
examples.  Then ::

    $ cd <path>/ad_usermanual/playground
    $ python


Imports
=======

Before doing anything, you need to import |AstroData| and the Gemini instrument
configuration package |gemini_instruments|.

::

    >>> import astrodata
    >>> import gemini_instruments


Open and access existing dataset
================================

Read in the dataset
-------------------

The file on disk is loaded into the |AstroData| class associated with the
instrument the data is from. This association is done automatically based on
header content.

::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    >>> type(ad)
    <class 'gemini_instruments.gmos.adclass.AstroDataGmos'>

From now on, ``ad`` knows it is GMOS data.  It knows how to access its headers
and when using the Recipe System (|recipe_system|), it will trigger the
selection of the GMOS primitives and recipes.

The original path and filename are stored in the object. If you were to write
the |AstroData| object to disk without specifying anything, those path and
filename would be used. ::

    >>> ad.path
    '../playdata/N20170609S0154.fits'
    >>> ad.filename
    'N20170609S0154.fits'


Accessing the content of a MEF file
-----------------------------------

Accessing pixel data, headers, tables will be covered in details in following
chapters.  Here we just introduce the basic content interface.

For details on the |AstroData| structure, please refer to the
:ref:`previous chapter <structure>`.

|AstroData| uses |NDData| as the core of its structure. Each FITS extension
becomes a |NDAstroData| object, subclassed from |NDData|, and is added to
a list.

Pixel data
^^^^^^^^^^

To access pixel data, the list index and the ``.data`` attribute are used. That
returns a :class:`numpy.ndarray`. The list of |NDAstroData| is zero-indexed.
*Extension number 1 in a MEF is index 0 in an |AstroData| object*. ::

    >>> ad = astrodata.open('../playdata/N20170609S0154_varAdded.fits')
    >>> data = ad[0].data
    >>> type(data)
    <type 'numpy.ndarray'>
    >>> data.shape
    (2112, 256)

Remember that in a :class:`~numpy.ndarray` the y-axis is the first number.

The variance and data quality planes, the VAR and DQ planes in Gemini MEF
files, are represented by the ``.variance`` and ``.mask`` attributes,
respectively. They are not their own "extension", they don't have their
own index in the list, unlike in a MEF. They are attached to the pixel data,
packaged together by the |NDAstroData| object. They are represented as
:class:`numpy.ndarray` just like the pixel data ::

    >>> var = ad[0].variance
    >>> dq = ad[0].mask

Tables
^^^^^^
Tables in the MEF file will also be loaded into the |AstroData| object. If a table
is associated with a specific science extension through the EXTVER header, that
table will be packaged within the same AstroData extension as the pixel data.
The |AstroData| "extension" is the |NDAstroData| object plus any table or other pixel
array. If the table is not associated with a specific extension and applies
globally, it will be added to the AstroData object as a global addition. No
indexing will be required to access it.  In the example below, one ``OBJCAT`` is
associated with each extension, while the ``REFCAT`` has a global scope ::

    >>> ad.info()
    Filename: ../playdata/N20170609S0154_varAdded.fits
    Tags: ACQUISITION GEMINI GMOS IMAGE NORTH OVERSCAN_SUBTRACTED OVERSCAN_TRIMMED
        PREPARED SIDEREAL

    Pixels Extensions
    Index  Content                  Type              Dimensions     Format
    [ 0]   science                  NDAstroData       (2112, 256)    float32
              .variance             ndarray           (2112, 256)    float32
              .mask                 ndarray           (2112, 256)    uint16
              .OBJCAT               Table             (6, 43)        n/a
              .OBJMASK              ndarray           (2112, 256)    uint8
    [ 1]   science                  NDAstroData       (2112, 256)    float32
              .variance             ndarray           (2112, 256)    float32
              .mask                 ndarray           (2112, 256)    uint16
              .OBJCAT               Table             (8, 43)        n/a
              .OBJMASK              ndarray           (2112, 256)    uint8
    [ 2]   science                  NDAstroData       (2112, 256)    float32
              .variance             ndarray           (2112, 256)    float32
              .mask                 ndarray           (2112, 256)    uint16
              .OBJCAT               Table             (7, 43)        n/a
              .OBJMASK              ndarray           (2112, 256)    uint8
    [ 3]   science                  NDAstroData       (2112, 256)    float32
              .variance             ndarray           (2112, 256)    float32
              .mask                 ndarray           (2112, 256)    uint16
              .OBJCAT               Table             (5, 43)        n/a
              .OBJMASK              ndarray           (2112, 256)    uint8

    Other Extensions
                   Type        Dimensions
    .REFCAT        Table       (245, 16)


The tables are stored internally as :class:`astropy.table.Table` objects. ::

    >>> ad[0].OBJCAT
    <Table length=6>
    NUMBER X_IMAGE Y_IMAGE ... REF_MAG_ERR PROFILE_FWHM PROFILE_EE50
    int32  float32 float32 ...   float32     float32      float32
    ------ ------- ------- ... ----------- ------------ ------------
         1 283.461 55.4393 ...     0.16895       -999.0       -999.0
    ...
    >>> type(ad[0].OBJCAT)
    <class 'astropy.table.table.Table'>

    >>> refcat = ad.REFCAT
    >>> type(refcat)
    <class 'astropy.table.table.Table'>


Headers
^^^^^^^
Headers are stored in the |NDAstroData| ``.meta`` attribute as :class:`astropy.io.fits.Header` objects,
which is a form of Python ordered dictionaries. Headers associated with extensions
are stored with the corresponding |NDAstroData| object. The MEF Primary Header
Unit (PHU) is stored "globally" in the |AstroData| object. Note that when slicing an |AstroData| object,
for example copying over just the first extension, the PHU will follow. The
slice of an |AstroData| object is an |AstroData| object.
Headers can be accessed directly, or for some predefined concepts, the use of
Descriptors is preferred. See the chapters on headers for details.

Using Descriptors::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    >>> ad.filter_name()
    'open1-6&g_G0301'
    >>> ad.filter_name(pretty=True)
    'g'

Using direct header access::

    >>> ad.phu['FILTER1']
    'open1-6'
    >>> ad.phu['FILTER2']
    'g_G0301'

Accessing the extension headers::

    >>> ad.hdr['CCDSEC']
    ['[1:512,1:4224]', '[513:1024,1:4224]', '[1025:1536,1:4224]', '[1537:2048,1:4224]']
    >>> ad[0].hdr['CCDSEC']
    '[1:512,1:4224]'

    With descriptors:
    >>> ad.array_section(pretty=True)
    ['[1:512,1:4224]', '[513:1024,1:4224]', '[1025:1536,1:4224]', '[1537:2048,1:4224]']


Modify Existing MEF Files
=========================
Before you start modify the structure of an |AstroData| object, you should be
familiar with it. Please make sure that you have read the previous chapter
on :ref:`the structure of the AstroData object <structure>`.

Appending an extension
----------------------
In this section, we take an extension from one |AstroData| object and append it
to another. Because we are mapping a FITS file, the ``EXTVER`` keyword gets
automatically updated to the next available value to ensure that when the
|AstroData| object is written back to disk as MEF, it will be coherent.

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
    [ 0]   science                  NDAstroData       (2112, 288)    uint16
    [ 1]   science                  NDAstroData       (2112, 288)    uint16
    [ 2]   science                  NDAstroData       (2112, 288)    uint16
    [ 3]   science                  NDAstroData       (2112, 288)    uint16

    >>> ad.append(advar[3])
    >>> ad.info()
    Filename: ../playdata/N20170609S0154.fits
    Tags: ACQUISITION GEMINI GMOS IMAGE NORTH RAW SIDEREAL UNPREPARED
    Pixels Extensions
    Index  Content                  Type              Dimensions     Format
    [ 0]   science                  NDAstroData       (2112, 288)    uint16
    [ 1]   science                  NDAstroData       (2112, 288)    uint16
    [ 2]   science                  NDAstroData       (2112, 288)    uint16
    [ 3]   science                  NDAstroData       (2112, 288)    uint16
    [ 4]   science                  NDAstroData       (2112, 256)    float32
              .variance             ndarray           (2112, 256)    float32
              .mask                 ndarray           (2112, 256)    int16
              .OBJCAT               Table             (5, 43)        n/a
              .OBJMASK              ndarray           (2112, 256)    uint8

    >>> ad[4].hdr['EXTVER']
    5
    >>> advar[3].hdr['EXTVER']
    4

As you can see above, the fourth extension of ``advar``, along with everything
it contains was appended at the end of the first |AstroData| object. Also, note
that the EXTVER of the extension in ``advar`` was 4, but once appended
to ``ad``, it had to be changed to the next available integer, 5, numbers 1 to
4 being already used by ``ad``'s own extensions.

In this next example, we are appending only the pixel data, leaving behind the other
associated data. The header associated with that data does follow however.

::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    >>> advar = astrodata.open('../playdata/N20170609S0154_varAdded.fits')

    >>> ad.append(advar[3].data)
    >>> ad.info()
    Filename: ../playdata/N20170609S0154.fits
    Tags: ACQUISITION GEMINI GMOS IMAGE NORTH RAW SIDEREAL UNPREPARED
    Pixels Extensions
    Index  Content                  Type              Dimensions     Format
    [ 0]   science                  NDAstroData       (2112, 288)    uint16
    [ 1]   science                  NDAstroData       (2112, 288)    uint16
    [ 2]   science                  NDAstroData       (2112, 288)    uint16
    [ 3]   science                  NDAstroData       (2112, 288)    uint16
    [ 4]   science                  NDAstroData       (2112, 256)    float32

Notice how a new extension was created but ``variance``, ``mask``, the OBJCAT
table and OBJMASK image were not copied over. Only the science pixel data was
copied over.

Please note, there is no implementation for the "insertion" of an extension.

Removing an extension or part of one
------------------------------------
Removing an extension or a part of an extension is straightforward. The
Python command :func:`del` is used on the item to remove. Below are a few
examples, but first let us load a file ::

    >>> ad = astrodata.open('../playdata/N20170609S0154_varAdded.fits')
    >>> ad.info()

As you go through these examples, check the new structure with :func:`ad.info()`
after every removal to see how the structure has changed.

Deleting a whole |AstroData| extension, the fourth one ::

    >>> del ad[3]

Deleting only the variance array from the second extension ::

    >>> ad[1].variance = None

Deleting a table associated with the first extension ::

    >>> del ad[0].OBJCAT

Deleting a global table, not attached to a specific extension ::

    >>> del ad.REFCAT



Writing back to disk
====================
The :class:`~astrodata.fits.AstroDataFits` layer takes care of converting
the |AstroData| object back to a MEF file on disk. When writing to disk,
one should be aware of the path and filename information associated
with the |AstroData| object.

::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    >>> ad.path
    '../playdata/N20170609S0154.fits'
    >>> ad.filename
    'N20170609S0154.fits'

Writing to a new file
---------------------
There are various ways to define the destination for the new FITS file.
The most common and natural way is ::

    >>> ad.write('new154.fits')

    >>> ad.write('new154.fits', overwrite=True)

This will write a FITS file named 'new154.fits' in the current directory.
With ``overwrite=True``, it will overwrite the file if it already exists.
A path can be prepended to the filename if the current directory is not
the destination.
Note that ``ad.filename`` and ``ad.path`` have not changed, we have just
written to the new file, the |AstroData| object is in no way associated
with that new file.  ::

    >>> ad.path
    '../playdata/N20170609S0154.fits'
    >>> ad.filename
    'N20170609S0154.fits'

If you want to create that association, the ``ad.filename`` and ``ad.path``
needs to be modified first.  For example::

    >>> ad.filename = 'new154.fits'
    >>> ad.write()

    >>> ad.path
    '../playdata/new154.fits'
    >>> ad.filename
    'new154.fits'

Changing ``ad.filename`` also changes the filename in the ``ad.path``. The
sequence above will write 'new154.fits' not in the current directory but
rather to the directory that is specified in ``ad.path``.

WARNING: :func:`ad.write` has an argument named ``filename``.  Setting ``filename``
in the call to :func:`ad.write`, as in ``ad.write(filename='new154.fits')`` will NOT
modify ``ad.filename`` or ``ad.path``.  The two "filenames", one a method argument
the other a class attribute have no association to each other.


Updating an existing file on disk
----------------------------------
Updating an existing file on disk requires explicitly allowing overwrite.

If you have not written 'new154.fits' to disk yet (from previous section) ::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    >>> ad.write('new154.fits')

Now let's open 'new154.fits', and write to it ::

    >>> adnew = astrodata.open('new154.fits')
    >>> adnew.write(overwrite=True)



Create New MEF Files
====================

A new MEF file can be created from an existing, maybe modified, file or it
can be created from scratch.  We discuss both cases here.

Create New Copy of MEF Files
----------------------------
To create a new copy of a MEF file, modified or not, the user has already
been given most of the tools in the sections above.  Yet, let's throw a
couple examples for completeness.

Basic example
^^^^^^^^^^^^^
As seen above, a MEF file can be opened with |astrodata|, the |AstroData|
object can be modified (or not), and then written back to disk under a
new name.  ::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    ... optional modifications here ...
    >>> ad.write('newcopy.fits')


Needing true copies in memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sometimes it is a true copy in memory that is needed.  This is not specific
to MEF.  In Python, doing something like ``adnew = ad`` does not create a
new copy of the AstrodData object; it just gives it a new name.  If you
modify ``adnew`` you will be modify ``ad`` too.  They point to the same block
of memory.

To create a true independent copy, the ``deepcopy`` utility needs to be used. ::

    >>> from copy import deepcopy
    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    >>> adcopy = deepcopy(ad)

Be careful using ``deepcopy``, you memory could balloon really fast.  Use it
only when truly needed.


Create New MEF Files from Scratch
---------------------------------
Before one creates a new MEF file on disk, one has to create the AstroData
object that will be eventually written to disk.  The |AstroData| object
created also needs to know that it will have to be written using the MEF
format. This is fortunately handled fairly transparently by |astrodata|.

The key to associating the FITS data provider to the |AstroData| object
is simply to create the |AstroData| object from :mod:`astropy.io.fits` header
objects. Those will be recognized by |astrodata| as FITS and the
constructor for FITS will be used. The user does not need to do anything
else special. Here is how it is done.

Create a MEF with basic header and data array set to zeros
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    >>> import numpy as np
    >>> from astropy.io import fits

    >>> phu = fits.PrimaryHDU()

    >>> pixel_data = np.zeros((100,100))

    >>> hdu = fits.ImageHDU()
    >>> hdu.data = pixel_data

    >>> ad = astrodata.create(phu)
    >>> ad.append(hdu, name='SCI')

    or another way to do the last two blocs:
    >>> hdu = fits.ImageHDU(data=pixel_data, name='SCI')
    >>> ad = astrodata.create(phu, [hdu])

Then it is just a matter of calling ``ad.write('somename.fits')`` on that
new ``Astrodata`` object.

Represent a table as a FITS binary table in an ``AstroData`` object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
One first needs to create a table, either an :class:`astropy.table.Table`
or a :class:`~astropy.io.fits.BinTableHDU`. See the |astropy| documentation
on tables and this manual's :ref:`section <tables>` dedicated to tables for
more information.

In the first example, we assume that ``my_astropy_table`` is
a :class:`~astropy.table.Table` ready to be attached to an |AstroData|
object.  (Warning: we have not created ``my_astropy_table`` therefore the
example below will not run, though this is how it would be done.)

::

    >>> phu = fits.PrimaryHDU()
    >>> ad = astrodata.create(phu)

    >>> astrodata.add_header_to_table(my_astropy_table)
    >>> ad.append(my_astropy_table, name='BOB')


In the second example, we start with a FITS :class:`~astropy.io.fits.BinTableHDU`
and attach it to a new |AstroData| object. (Again, we have not created
``my_fits_table`` so the example will not run.) ::

    >>> phu = fits.PrimaryHDU()
    >>> ad = astrodata.create(phu)
    >>> ad.append(my_fits_table, name='BILL')

As before, once the |AstroData| object is constructed, the ``ad.write()``
method can be used to write it to disk as a MEF file.



