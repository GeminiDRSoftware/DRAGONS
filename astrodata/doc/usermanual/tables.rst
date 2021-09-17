.. tables.rst

.. _tables:

**********
Table Data
**********
**Try it yourself**

Download the data package (:ref:`datapkg`) if you wish to follow along and run the
examples.  Then ::

    $ cd <path>/ad_usermanual/playground
    $ python

Then import core astrodata and the Gemini astrodata configurations. ::

    >>> import astrodata
    >>> import gemini_instruments

Tables and Astrodata
====================
Tables are stored as ``astropy.table`` ``Table`` class.   FITS tables too
are represented in Astrodata as ``Table`` and FITS headers are stored in
the NDAstroData `.meta` attribute.  Most table access should be done
through the ``Table`` interface.   The best reference on ``Table`` is the
Astropy documentation itself.  In this chapter we covers some common
examples to get the reader started.

The ``astropy.table`` documentation can be found at: `<http://docs.astropy.org/en/stable/table/index.html>`_


Operate on a Table
==================

Let us open a file with tables.  Some tables are associated with specific
extensions, and there is one table that is global to the `AstroData` object.

::

    >>> ad = astrodata.open('../playdata/N20170609S0154_varAdded.fits')
    >>> ad.info()

To access the global table named ``REFCAT``::

    >>> ad.REFCAT

To access the ``OBJCAT`` table in the first extension ::

    >>> ad[0].OBJCAT


Column and Row Operations
-------------------------
Columns are named.  Those names are used to access the data as columns.
Rows are not names and are simply represented as a sequential list.

Read columns and rows
+++++++++++++++++++++
To get the names of the columns present in the table::

    >>> ad.REFCAT.colnames
    ['Id', 'Cat_Id', 'RAJ2000', 'DEJ2000', 'umag', 'umag_err', 'gmag',
    'gmag_err', 'rmag', 'rmag_err', 'imag', 'imag_err', 'zmag', 'zmag_err',
    'filtermag', 'filtermag_err']

Then it is easy to request the values for specific columns::

    >>> ad.REFCAT['zmag']
    >>> ad.REFCAT['zmag', 'zmag_err']

To get the content of a specific row, row 10 in this case::

    >>> ad.REFCAT[9]

To get the content of a specific row(s) from a specific column(s)::

    >>> ad.REFCAT['zmag'][4]
    >>> ad.REFCAT['zmag'][4:10]
    >>> ad.REFCAT['zmag', 'zmag_err'][4:10]

Change values
+++++++++++++
Assigning new values works in a similar way.  When working on multiple elements
it is important to feed a list that matches in size with the number of elements
to replace.

::

    >>> ad.REFCAT['imag'][4] = 20.999
    >>> ad.REFCAT['imag'][4:10] = [5, 6, 7, 8, 9, 10]

    >>> overwrite_col = [0] * len(ad.REFCAT)  # a list of zeros, size = nb of rows
    >>> ad.REFCAT['imag_err'] = overwrite_col

Add a row
+++++++++
To append a row, there is the ``add_row()`` method.  The length of the row
should match the number of columns::

    >>> new_row = [0] * len(ad.REFCAT.colnames)
    >>> new_row[1] = ''   # Cat_Id column is of "str" type.
    >>> ad.REFCAT.add_row(new_row)

Add a column
++++++++++++
Adding a new column can be more involved.  If you need full control, please
see the AstroPy Table documentation.  For a quick addition, which might be
sufficient for your use case, we simply use the "dictionary" technique.  Please
note that when adding a column, it is important to ensure that all the
elements are of the same type.  Also, if you are planning to use that table
in IRAF/PyRAF, we recommend not using 64-bit types.

::

    >>> import numpy as np

    >>> new_column = [0] * len(ad.REFCAT)
    >>> # Ensure that the type is int32, otherwise it will default to int64
    >>> # which generally not necessary.  Also, IRAF 32-bit does not like it.
    >>> new_column = np.array(new_column).astype(np.int32)
    >>> ad.REFCAT['my_column'] = new_column

If you are going to write that table back to disk as a FITS Bintable, then
some additional headers need to be set.  Astrodata will take care of that
under the hood when the `write` method is invoked.

::

    >>> ad.write('myfile_with_modified_table.fits')


Selection and Rejection Operations
----------------------------------
Normally, one does not know exactly where the information needed is located
in a table.  Rather some sort of selection needs to be done.  This can also
be combined with various calculations.  We show two such examples here.

Select a table element from criterion
+++++++++++++++++++++++++++++++++++++

::

    >>> # Get the magnitude of a star selected by ID number
    >>> ad.REFCAT['zmag'][ad.REFCAT['Cat_Id'] == '1237662500002005475']

    >>> # Get the ID and magnitude of all the stars brighter than zmag 18.
    >>> ad.REFCAT['Cat_Id', 'zmag'][ad.REFCAT['zmag'] < 18.]


Rejection and selection before statistics
+++++++++++++++++++++++++++++++++++++++++

::

    >>> t = ad.REFCAT   # to save typing

    >>> # The table has "NaN" values.  ("Not a number")  We need to ignore them.
    >>> t['zmag'].mean()
    nan
    >>> # applying rejection of NaN values:
    >>> t['zmag'][np.where(~np.isnan(t['zmag']))].mean()
    20.377306



Accessing FITS table headers directly
-------------------------------------
If for some reason you need to access the FITS table headers directly, here
is how to do it.  It is very unlikely that you will need this.

To see the FITS headers::

    >>> ad.REFCAT.meta['header']
    >>> ad[0].OBJCAT.meta['header']

To retrieve a specific FITS table header::

    >>> ad.REFCAT.meta['header']['TTYPE3']
    'RAJ2000'
    >>> ad[0].OBJCAT.meta['header']['TTYPE3']
    'Y_IMAGE'

To retrieve all the keyword names matching a selection::

    >>> keynames = [key for key in ad.REFCAT.meta['header'] if key.startswith('TTYPE')]



Create a Table
==============

To create a table that can be added to an ``AstroData`` object and eventually
written to disk as a FITS file, the first step is to create an Astropy
``Table``.

Let us first add our data to NumPy arrays, one array per column::

    >>> import numpy as np

    >>> snr_id = np.array(['S001', 'S002', 'S003'])
    >>> feii = np.array([780., 78., 179.])
    >>> pabeta = np.array([740., 307., 220.])
    >>> ratio = pabeta / feii

Then build the table from that data::

    >>> from astropy.table import Table

    >>> my_astropy_table = Table([snr_id, feii, pabeta, ratio],
    ...                          names=('SNR_ID', 'FeII', 'PaBeta', 'ratio'))


Now we append this Astropy ``Table`` to a new ``AstroData`` object.

::

    >>> # Since we are going to write a FITS, we build the AstroData object
    >>> # from FITS objects.
    >>> from astropy.io import fits

    >>> phu = fits.PrimaryHDU()
    >>> ad = astrodata.create(phu)
    >>> ad.MYTABLE = my_astropy_table
    >>> ad.info()
    >>> ad.MYTABLE

    >>> ad.write('new_table.fits')
