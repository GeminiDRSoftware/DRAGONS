.. tables:

**********
Table Data
**********

AstroData does not provide any special wrappers for FITS Table Data.  But
since AstroData is built on top of PyFITS, the standard PyFITS
table functions can be used.  The reader should refer to the PyFITS 
table documentation for complete details (e.g. 
https://pythonhosted.org/pyfits/users_guide/users_table.html). In this chapter, 
a few useful examples of basic usage are shown.  As the reader will see, table 
manipulation can be tedious and tricky.



``astropy.table`` might simplify some of the examples presented
in this chapter.  This will be looked into at some future time.  (Suggestions
for improvements are welcome.)


**Try it yourself**


If you wish to follow along and try the commands yourself, download
the data package, go to the ``playground`` directory and copy over
the necessary files.

::

   cd <path>/gemini_python_datapkg-X1/playground
   cp ../data_for_ad_user_manual/estgsS20080220S0078.fits .

Then launch the Python shell::

   python


.. highlight:: python
   :linenothreshold: 5


Read from a FITS Table
======================

A FITS table is stored in a MEF file as a ``BinTableHDU``.  The table data is 
retrieved from the ``AstroData`` object with the same ``.data`` attribute as 
for pixel extensions, but for FITS tables ``.data`` returns a ``FITS_rec``, 
which is a PyFITS class, instead of a Numpy ``ndarray``.  Here is how to
get information out of a FITS table.

::

   from astrodata import AstroData
   
   adspec = AstroData('estgsS20080220S0078.fits')
   adspec.info()
      
   # For easier reference, assign FITS table to variable.
   tabledata = adspec['MDF'].data
   
   # Get the column names with 'names' or more details with 'columns'
   tabledata.names
   tabledata.columns
   
   # Get all the data for a column
   x_ccd_values = tabledata.field('x_ccd')
   third_col = tabledata.field(2)
   
   # Print the table content
   print tabledata
   
   # Print the first 2 rows
   print tabledata[:2]
   
   # Select rows based on some criterion
   select_tabledata = tabledata[tabledata.field('y_ccd') > 2000.]
   print select_tabledata

The first extension in that file is a FITS table with ``EXTNAME`` MDF, and 
``EXTVER`` 1 (Lines 4, 7) .  MDF stands for "Mask Definition File".  In 
Gemini data, those are used in the data reduction to identify, to first 
order, where spectra fall on the detector.

The output of ``names`` (Line 10) is a simple list of strings.  The output
of ``columns`` is a PyFITS ``ColDefs`` object.  When printed it looks like this::

   ColDefs(
       name = 'ID'; format = '1J'; null = -2147483647; disp = 'A3'
       name = 'x_ccd'; format = '1E'; disp = 'F8.2'
       name = 'y_ccd'; format = '1E'; disp = 'F8.2'
       name = 'slittype'; format = '20A'; disp = 'A10'
       name = 'slitid'; format = '1J'; null = -2147483647; disp = 'A3'
       name = 'slitpos_mx'; format = '1E'; disp = 'F8.2'
       name = 'slitpos_my'; format = '1E'; disp = 'F8.2'
       name = 'slitsize_mx'; format = '1E'; disp = 'F8.2'
       name = 'slitsize_my'; format = '1E'; disp = 'F8.2'
       name = 'slittilt_m'; format = '1E'; disp = 'F8.2'
       name = 'slitsize_mr'; format = '1E'; disp = 'F6.2'
       name = 'slitsize_mw'; format = '1E'; disp = 'F6.2'
   )

When a column is retrieved, like in Lines 14 and 15, the returned value is
a numpy ``ndarray``.

Note on Line 15 that the third column is in index position 2; all Python 
arrays are zero-indexed.

  
Create a FITS Table
===================

Creating a FITS table is mostly a matter of creating the columns, name and 
data.  The name is a string, the data is stored in a numpy ``ndarray``.

::

   from astrodata import AstroData
   import pyfits as pf
   import numpy as np
   
   # Create the input data
   snr_id = np.array(['S001','S002','S003'])
   feii = np.array([780.,78.,179.])
   pabeta = np.array([740.,307.,220.])
   ratio = pabeta/feii
   
   # Create the columns
   col1 = pf.Column(name='SNR_ID', format='4A', array=snr_id)
   col2 = pf.Column(name='ratio', format='E', array=ratio)
   col3 = pf.Column(name='feii', format='E', array=feii)
   col4 = pf.Column(name='pabeta', format='E', array=pabeta)
   
   # Assemble the columns
   cols = pf.ColDefs([col1, col2, col3, col4])
   
   # Create the table HDU
   tablehdu = pf.new_table(cols)
   
   # Create an AstroData object to contain the table
   # and write to disk.
   new_ad = AstroData(tablehdu)
   new_ad.rename_ext('MYTABLE', 1)
   new_ad.info()
   
   new_ad.write('mytable.fits')

A new FITS table can also be appended to an already existing AstroData object with
the ``.append()`` function.
  

Operate on a FITS Table
=======================
The PyFITS manual is the recommended source for more complete documentation
on working on FITS table with Python.  Here are a few examples of how one can
modify a FITS table.

Preparation for the examples
----------------------------

In order to run the examples in the next few sections, the reader will need
to create these three tables.

::

   from astrodata import AstroData
   import pyfits as pf
   import numpy as np
   
   # Let us first create tables to play with
   snr_id = np.array(['S001','S002','S003'])
   feii = np.array([780.,78.,179.])
   pabeta = np.array([740.,307.,220.])
   ratio = pabeta/feii
   col1 = pf.Column(name='SNR_ID', format='4A', array=snr_id)
   col2 = pf.Column(name='ratio', format='E', array=ratio)
   col3 = pf.Column(name='feii', format='E', array=feii)
   col4 = pf.Column(name='pabeta', format='E', array=pabeta)
   cols_t1 = pf.ColDefs([col1,col3])
   cols_t2 = pf.ColDefs([col1,col4])
   cols_t3 = pf.ColDefs([col2])
   
   table1 = pf.new_table(cols_t1)
   table2 = pf.new_table(cols_t2)
   table3 = pf.new_table(cols_t3)

Merging tables
--------------

WARNING:  The input tables must **not** share any common field (ie. column)
names.  For example, *table1* and *table2* created above cannot be merged this
way since they share ``col1``.

The merging of tables is effectively the equivalent of appending columns.

::

   merged_cols = table1.columns + table3.columns
   merged_table = pf.new_table(merged_cols)
   
   merged_table.columns.names  # or merged_table.data.names
   print merged_table.data

The columns are now::

   ['SNR_ID', 'feii', 'ratio']

It is interesting to note that table operations are actually *column* 
operations followed by the creation of a new table (Lines 1 and 2).  
The next example will illustrate this a bit better.

Appending and deleting columns
------------------------------

::
   
   # Append the 'pabeta' column from table2 to table1
   index_of_pabeta_col = table2.columns.names.index('pabeta')
   table1.columns.add_col(table2.columns[index_of_pabeta_col])
   table1 = pf.new_table(table1.columns)
   
   table1.columns.names
   print table1.data

The append example (Lines 2-4) shows that the real work is done on the 
columns, not on the table as such.  To add a column to ``table1``, once the 
columns have been reorganized, a *new* table is created and, in this case, 
replaces the original ``table1``.

The index of the ``pabeta`` column in ``table2`` is found with the ``index``
method as shown on Line 2.  Then it is just a matter of adding that column
from ``table2`` to the columns of ``table1`` (Line 3).

The columns in the new ``table1`` are::

   ['SNR_ID', 'feii', 'pabeta']


::
   
   #   To "delete" the 'pabeta' column from this new table1
   table1.columns.del_col('pabeta')
   table1 = pf.new_table(table1.columns)
   
   table1.columns.names
   print table1.data

To delete a column, the process is similar:  the work is done on the columns,
then a *new* table is created to replace the original (Lines 2, 3).

The columns in the final ``table1`` are::

   ['SNR_ID', 'feii']
  

Inserting columns
-----------------

Column insertion is really about gathering all the columns and reorganizing
them manually.  There are no "insertion" tool, per se, in pyfits.  
(``astropy.table`` does have one though.)

Below, we insert the column from ``table3`` in-between the first and second
column of ``table1``.

::

   t1_col1 = table1.columns[0]
   t1_col2 = table1.columns[1]
   t3_col1 = table3.columns[0] 
   table1 = pf.new_table([t1_col1,t3_col1,t1_col2])
   
   table1.columns.names
   print table1.data

The columns in the resulting ``table1`` are::

   ['SNR_ID', 'ratio', 'feii']

Changing the name of a column
-----------------------------

WARNING: There is a pyfits ``columns`` method called ``change_name`` but it 
does not seem to be working properly.

::

   table1.columns[table1.columns.names.index('feii')].name='ironII'
   table1 = pf.new_table(table1.columns)
   
   table1.columns.names

To change the name of a column, one needs to change the ``name`` attribute
of the column.  On the first line, the position index of the 
column named ``feii`` is used to select the column to change, and then the
name of that column is changed to ``ironII``.

Again, a *new* table needs to be created once the modifications to the columns
are completed.

The ``table1`` columns are now::

   ['SNR_ID', 'ratio', 'ironII']


Appending and deleting rows
---------------------------

Appending and deleting rows is uncannily complicated with PyFITS.
This is an area where the use ``astropy.table`` can certainly help.  We hope
to be able to add astropy-based examples to this manual in the near future.
But for now, let us study the PyFITS way.

*Disclaimer*:  This is the way the author figured out how to do the row
manipulations.  If the reader knows of a better way to do it with PyFITS,
please let us know.

Below, we append two new entries to ``table2``.  Only the ``SNR_ID`` and 
``pabeta`` fields will be added to the table since those are the only
two columns in ``table2``.  When an entry has fields not represented
in the table, those fields are simply ignored.

::

   # New entries for object S004 and S005.
   new_entries = {'SNR_ID': ['S004','S005'],
                'ratio' : [1.12, 0.72],
                'feii'  : [77., 87.],
                'pabeta': [69., 122.]
                }
   nb_new_entries = len(new_entries['SNR_ID'])
   
   # Create new, larger table.
   nrowst2 = table2.data.shape[0]
   large_table = pf.new_table(table2.columns, nrows=nrowst2+nb_new_entries)
   
   # Append the new entries and replace table2 with new table.
   for name in table2.columns.names:
      large_table.data.field(name)[nrowst2:] = new_entries[name]

   table2 = large_table
   print table2.data

The values must be entered for each column separately.  On Lines 14-15,
we loop through the columns by name.  To simplify things, it is convenient
to have the new values stored in a dictionary keyed on the column names
(Lines 2-6).

Adding, and deleting rows (next example), requires the creation of a new table of the correct,
new size (Lines 10-11).

::
   
   # Delete the last 2 entries from table2
   
   # Create new, smaller table.
   nb_bad_entries = 2
   nrowst2 = table2.data.shape[0]
   small_table = pf.new_table(table2.columns, nrows=nrowst2-nb_bad_entries)
   
   # Copy the large table minus the last two lines to the small table.
   for name in table2.columns.names:
      small_table.data.field(name)[:] = table2.data.field(name)[:-nb_bad_entries]
      
   table2 = small_table


Changing a value
----------------

Changing a value is simply a matter of identifying the column and the row that 
needs the new value.

Below we show how one might search one column to identify the row and then 
change that row in another column.

::

   # Change the 'pabeta' value for source S002 in table2
   rowindex = np.where(table2.data.field('SNR_ID') == 'S002')[0][0]
   table2.data.field('pabeta')[rowindex] = 888.
     