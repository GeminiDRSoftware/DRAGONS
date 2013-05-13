.. tables:

**********
Table Data
**********
``Astrodata`` does not provide any special wrappers for FITS Table Data.  But
since ``astrodata`` is built on top of ``pyfits``, the standard ``pyfits``
table functions can be used.  The reader should refer to the ``pyfits`` documentation
for complete details.  Here we show a few useful examples of basic usage.

Read from a FITS Table
======================
A FITS table is stored in a MEF file as a ``BinTableHDU``.  The table data is retrieved from
the ``AstroData`` object with the same ``.data`` attribute as for pixel extension, but for
FITS tables ``.data`` returns a ``FITS_rec``, which is a ``pyfits`` class.  Here is how to
get information out of a FITS table.::

  from astrodata import AstroData
  
  adspec = AstroData('estgsS20080220S0078.fits')
  adspec.info()
  # The first extension in that file is a FITS table with ``EXTNAME`` MDF, and ``EXTVER`` 1.
  # MDF stands for "Mask Definition File".  In Gemini data, those are used in the data reduction
  # to identify, to first order, where spectra fall on the detector.
  
  # Let's get the table data out of the AstroData object
  table = adspec['MDF'].data
  
  # Get the column names with 'names' or more details with 'columns'
  table.names
  table.columns
  
  # Get all the data for a column
  x_ccd_values = table.field('x_ccd')
  third_col = table.field(2)
  
  # Print the table content
  print table
  
  # Print the first 2 rows
  print table[:2]
  
  # Select rows based on some criterion
  select_table = table[table.field('y_ccd') > 2000.]
  print select_table

  
Create a FITS Table
===================
Creating a FITS table is mostly a matter of creating the columns, name and data.
The name is a string, the data is stored in a ``numpy`` array.::

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
  cols = pf.ColDefs([col1,col2,col3,col4])
  
  # Create the table HDU
  tablehdu = pf.new_table(cols)
  
  # Create an AstroData object to contain the table
  # and write to disk.
  new_ad = AstroData(tablehdu)
  new_ad.rename_ext('MYTABLE',1)
  new_ad.write('mytable.fits')

A new FITS table can also be appended to an already existing AstroData object with
the ``.append()`` function.
  

Operate on a FITS Table
=======================
The ``pyfits`` manual is the recommended source for a more complete documentation
on working on FITS table with Python.  Here are a few examples of what one can
modify a FITS table.::

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
  
  # Merge tables
  #   WARNING: The input tables must NOT share any common field names.
  #      For example, table1 and table2 cannot be merged this way since they share col1.
  merged_cols = table1.columns + table3.columns
  merged_table = pf.new_table(merged_cols)
  merged_table.columns.names  # or merged_table.data.names
  print merged_table.data
  
  # Add/Delete column
  #   To "add" the 'pabeta' column from table2 to table1
  table1.columns.add_col(table2.columns[table2.columns.names.index('pabeta')])
  table1 = pf.new_table(table1.columns)
  table1.columns.names
  print table1.data
  
  #   To "delete" the 'pabeta' column from this new table1
  table1.columns.del_col('pabeta')
  table1 = pf.new_table(table1.columns)
  table1.columns.names
  print table1.data
    
  # Insert column
  #   To insert a column, one has to extract the columns
  #   and reorganize them into a new table.
  #   Insert the first, and only column, in table3, between the first and second
  #   column in table1
  t1_col1 = table1.columns[0]
  t1_col2 = table1.columns[1]
  t3_col1 = table3.columns[0] 
  table1 = pf.new_table([t1_col1,t3_col1,t1_col2])
  table1.columns.names
  print table1.data
  
  # Change the name of a column
  #   WARNING: There is method .change_name but it does not seem to be
  #            working properly.
  table1.columns[table1.columns.names.index('feii')].name='ironII'
  table1 = pf.new_table(table1.columns)

  # Add/Delete row
  #  Adding and deleting rows requires the creation of a new table
  #  of the correct, new size.
  #
  #  Add 2 new entries to table2.  Only 'SNR_ID' and 'pabeta' will be 
  #  added as those are the columns already present in table2.
  nb_new_entries = 2
  new_entries = {'SNR_ID': ['S004','S005'],
                'ratio' : [1.12, 0.72],
                'feii'  : [77., 87.],
                'pabeta': [69., 122.]
                }
  nrowst2 = table2.data.shape[0]
  large_table = pf.new_table(table2.columns, nrows=nrowst2+nb_new_entries)
  for name in table2.columns.names:
      large_table.data.field(name)[nrowst2:]=new_entries[name]
  table2 = large_table
  
  # Delete the last 2 entries from table2
  nb_bad_entries = 2
  nrowst2 = table2.data.shape[0]
  small_table = pf.new_table(table2.columns, nrows=nrowst2-nb_bad_entries)
  for name in table2.columns.names:
      small_table.data.field(name)[:]=table2.data.field(name)[:-nb_bad_entries]
  table2 = small_table

  # Change the 'pabeta' value for source S002 in table2
  rowindex = np.where(table2.data.field('SNR_ID') == 'S002')[0][0]
  table2.data.field('pabeta')[rowindex] = 888.
