.. highlightlang:: rest

.. _mos_examples:

Examples
========

This section provides working examples to exercise Mosaic
and MosaicAD class and their functions.

.. _mosad_array:

Example 1: Create a mosaic using MosaicAD class.
--------------------------------------------------

- Start your favorite Python shell

- importing modules
  ::

    from astrodata import AstroData
    # The directory mosaicAD.py and gemMosaicFunction.py modules
    # will probably change when the code goes into production.
    from gempy.adlibrary.mosaicAD import MosaicAD
    #     This is a user function available for your use,
    #     it supports GMOS and GSAOI data
    from gempy.gemini.gemMosaicFunction import gemini_mosaic_function

.. _asastro_ex:

- Use *AstroData* to open a FITS file
  ::

   ad = AstroData('S20100113S0110vardq.fits')

- With :ref:`MosaicAD <mosad_input>` we instantiate an object using a user written function *gemini_mosaic_function*. The default image extension name is 'SCI'. Click :ref:`here <user_function_ad>` to see an example of a *user_function*.
  ::

   mo = MosaicAD(ad,mosaic_ad_function=gemini_mosaic_function)
   
- Use :ref:`mosaic_image_data <mosad_imdata>` method.
  The output *mosaic_array* is a numpy array of the same datatype as the input image array in the *ad* object. The blocks array are corrected (transformed) for shift, rotation and magnification with respect to the reference block.
  ::

    mosaic_array = mo.mosaic_image_data()

.. _mos_associated:

Example 2: Create an output AstroData object containing images and tables.
----------------------------------------------------------------------------

This example uses the *as_astrodata* method to create mosaics with merged associated tables, provided that your input file has BINTABLE extensions.  The default action is to act on all the extensions in the input AstroData object but you can specify a given extname as in the example. Please :ref:`see <mosad_asad>` the documentation for the available parameters in this method.

::

 from astrodata import AstroData
 # The directory for mosaicAD.py and gemMosaicFunction.py
 # will probably change when the code goes into production.
 from gempy.adlibrary.mosaicAD import MosaicAD
 #     This is a user function available for your use,
 #     it supports GMOS and GSAOI data
 from gempy.gemini.gemMosaicFunction import gemini_mosaic_function

 ad = AstroData(file)
 #     Create a mosaicAD object using the input ad
 #     and the default mosaic function.
 mo = MosaicAD(ad, gemini_mosaic_function)
 #     Now run the method to mosaic and associate the
 #     input ad extensions.
 outad = mo.as_astrodata()
 #     Print the content of the resulting ad object.
 print outad.info()
 #    DO THE SAME BUT USE ONLY THE 'VAR' extension.
 #    If the input tables are associated with 'VAR', they
 #    will be merged.
 # NOTICE that running 'as_astrodata' with the 'DQ' extension
 # is VERY slow since it will transform 8 DQ planes separately.
 outad_var = mo.as_astrodata(extname='VAR')
 #     Print the content of the resulting ad object.

 print outad.info()

.. _exam3:

Example 3: Create a tile array
------------------------------

A tile array is a mosaic array without the correction for shift,
rotation and magnification. Here we used the *mosaic_image_data* method
with the default extension name 'SCI'. By using the *extname* parameter
you can change the extname to get tile from. Notice the parameter *tile* must be
True.

::

 from astrodata import AstroData
 # The directory for mosaicAD.py and gemMosaicFunction.py
 # will probably change when the code goes into production.
 from gempy.adlibrary.mosaicAD import MosaicAD
 #     This is a user function available for your use,
 #     it supports GMOS and GSAOI data
 from gempy.gemini.gemMosaicFunction import gemini_mosaic_function
 
 ad = AstroData(file)
 #     Create a mosaicAD object using the input ad
 #     and the gemini mosaic function (assuming that file
 #     points to a GMOS or GSAOI FITS file).
 mo = MosaicAD(ad, gemini_mosaic_function)
 #     The default value for extname is 'SCI' 
 sci_tile = mo.mosaic_image_data(tile=True)
 #     Print the shape of the resulting tile.
 print 'Tile shape:',sci_tile.shape

 #     Use 'as_astrodata' to get an AstroData object as output.
 #     Notice the use of extname since the default action is
 #     to work on all extensions.
 outad = mo.as_astrodata(tile=True, extname='SCI')
 #     Print the content of the resulting ad object.
 print outad.info()

.. _exam4:

Example 4: Create a block from a given extension name
------------------------------------------------------------


A mosaic consists of one or more blocks, e.g. for GMOS 3-amp mode a mosaic has 3 
blocks; for a 6-amps mode still the mosaic has 3 blocks but each block has 2-amps.
The blocks' layout is represented with a tuple of the form (column, row) (zero-based).
Use the *extname* parameter from :ref:`mosaic_image_data <mosad_imdata>` 
to select which extname to get the block from.
::

 from astrodata import AstroData
 # The directory for mosaicAD.py and gemMosaicFunction.py
 # will probably change when the code goes into production.
 from gempy.adlibrary.mosaicAD import MosaicAD
 #     This is a user function available for your use,
 #     it supports GMOS and GSAOI data
 from gempy.gemini.gemMosaicFunction import gemini_mosaic_function
 
 ad = AstroData(file)
 mo = MosaicAD(ad, gemini_mosaic_function)
 #     Now use the mosaic_image_data method to generate
 #     an output ndarray by using the parameter block and
 #     its value set to a tuple (col,row) (0-based) of the block
 #     you want returned. For GMOS the block values are
 #     (0,0), (1,0), (2,0). The extension name is the
 #     default 'SCI'.
 block_array = mo.mosaic_image_data(block=(1,0),extname='VAR') 
 #     Get the shape: (height, width) in pixels.
 print block_array.shape

.. _user_function_ad:

Example 5: Write a *user_function* using Astrodata.
------------------------------------------------------------

A user function is necessary to instantiate a MosaicAD object. If you have an arbitrary FITS file then this one would probably work depending whether the input FITS file have the keywords DETSEC, CCDSEC and DATASEC.
::

 from astrodata import AstroData
 from gempy.gemini_metadata_utils import sectionStrToIntList

 # The directory for mosaic.py will probably 
 # change when the code goes into production.
 from gempy.library.mosaic import MosaicData, MosaicGeometry

 def my_input_function(file,extname=None):
    """
      ad: Input AstroData object.
      
      SUMMARY:
      1) Read image extensions 'SCI' from the ad.
         Append each extension to a *data_list* list.
      2) Read header keywords DETSEC and CCDSEC from the same
         extension as in 1) and form two lists with the keyword
         values. Turn these values to zero-based tuples of the
         form (x1,x2,y1,y2). The DETSEC list is named 'amp_mosaic_coord'
         and the CCDSEC list is named 'amp_block_coord'.
         If you don't have these keywords use other means to
         determine 'amp_mosaic_coord' and 'amp_block_coord'.
         Make a 'coords' dictionary with 'amp_mosaic_coord' and 'amp_block_coord' keys.
         So we would have:
         
         coords = {'amp_mosaic_coord': detsec_list, 'amp_block_coord': ccdsec_list}

      3) Instantiate a MosaicData object with the above lists.
      4) Set 'blocksize' to (nx,ny). nx is width and ny is the
         height -in pixels of the block containing the data_list
         elements.
      5) Set 'mosaic_grid'. (nblocks_x,nblocks_y), where nblocks_x 
         is the number of blocks in the x_direction and nblockcs_y
         is the number of rows. This is the mosaic layout. 

      RETURN: (mosaic_data, mosaic_geometry)
    """

    # TO THE READER: The basic of the AstroData assumptions if that EXTNAME
    #                is the same for all IMAGE extension of a certain type
    #                and the EXTVER value is different for all these extname's.
    #                Example: GMOS have EXTNAME 'SCI' and EXTVER have values
    #                1 for FITS extension 1, 2 for FITS extension 2, etc.
    
    # get data_list from the ad object
    
    ad = AstroData(file)
    data_list = [hdu.data for hdu in ad['SCI']]
    # Get DETSEC
    amps_mosaic_coord = ad.detector_section(extname='SCI').as_list()
    # Get CCDSEC. These can be equal for all extensions. The as_list() method
    # will remove duplicates, so let's try something else.
    amps_block_coord = (
        [sectionStrToIntList(hdu.header['CCDSEC']) for hdu in ad['SCI']] )
 
    # Form the coords dictionary 
    coords = {'amps_mosaic_coord': amps_mosaic_coord,
              'amps_block_coord': amps_block_coord}

    # Mosaic Data object
    md = MosaicData(data_list,coords)

    # Important: blocksize tuple is (blocksize_x, blocksize_y). Just to
    # keep the external representation in (x,y) order rather than
    # python's (y,x).
   
    # For simplicity make the blocksize the same as the input
    # data shape
 
    (sz_y, sz_x) = data_list[0].shape
    blocksize = (sz_y, sz_x)
    mosaic_grid = (2,2)

    # MosaicGeometry. We have a 'transformation' dictionary which
    # allows us to correct for rotation in this case.
    geo_dict = {
        'mosaic_grid':    mosaic_grid,
        'blocksize':      blocksize,
        'ref_block':      (0,0),  # 0-based
        'transformation': {  # shift and magnification will 
			     # have default values
              'rotation':  (0, 5.0, 4.5, 5.3),  # Rotation in degrees
                                                # for each block.
                          },
                }
         
    mg = MosaicGeometry(geo_dict)

    # Return require objects
    # 
    return md,mg

.. _user_function_pf:

Example 6: Write a *user_function* using Pyfits
------------------------------------------------------------

A user function is necessary to instantiate a MosaicAD object. If you have an arbitrary FITS file then this one would probably work depending whether the input FITS file have the keywords DETSEC, CCDSEC and DATASEC.
::

 import pyfits as pf
 from gempy.gemini_metadata_utils import sectionStrToIntList

 # The directory for mosaic.py will probably 
 # change when the code goes into production.
 from gempy.library.mosaic import MosaicData, MosaicGeometry

 def my_input_function(file,extname=None):
    """
      
      SUMMARY:
      1) Read image extensions 'SCI' from the hdulist.
         Append each extension to a *data_list* list.
         If the FITS file already have extension names other than
         'SCI' will try something else.
      2) Read header keywords DETSEC and CCDSEC from the same
         extension as in 1) and form two lists with the keyword
         values. Turn these values to zero-based tuples of the
         form (x1,x2,y1,y2). The DETSEC list is named 'amp_mosaic_coord'
         and the CCDSEC list is named 'amp_block_coord'.
         If you don't have these keywords use other means to
         determine 'amp_mosaic_coord' and 'amp_block_coord'.
         Make a 'coords' dictionary with 'amp_mosaic_coord' and 'amp_block_coord' keys.
         So we would have:
         
         coords = {'amp_mosaic_coord': detsec_list, 'amp_block_coord': ccdsec_list}

      3) Instantiate a MosaicData object with the above lists.
      4) Set 'blocksize' to (nx,ny). nx is width and ny is the
         height -in pixels of the block containing the data_list
         elements.
      5) Set 'mosaic_grid'. (nblocks_x,nblocks_y), where nblocks_x 
         is the number of blocks in the x_direction and nblockcs_y
         is the number of rows. This is the mosaic layout. 

      RETURN: (mosaic_data, mosaic_geometry)
    """

    fits = pf.open(file)
    data_list = [hdu.data for hdu in fits[1:]]

    amps_mosaic_coord = (
         [sectionStrToIntList(hdu.header['DETSEC']) for hdu in fits[1:]] )
    amps_block_coord = (
         [sectionStrToIntList(hdu.header['CCDSEC']) for hdu in fits[1:]] )
 
    # Form the coords dictionary 
    coords = {'amps_mosaic_coord': amps_mosaic_coord,
              'amps_block_coord': amps_block_coord}

    # Mosaic Data object
    md = MosaicData(data_list,coords)

    # Important: blocksize tuple is (blocksize_x, blocksize_y). Just to
    # keep the external representation in (x,y) order rather than
    # python's (y,x).
   
    # For simplicity make the blocksize the same as the input
    # data shape
 
    (sz_y, sz_x) = data_list[0].shape
    blocksize = (sz_y, sz_x)
    mosaic_grid = (2,2)

    # MosaicGeometry. We have a 'transformation' dictionary which
    # allows us to correct for rotation in this case.
    geo_dict = {
        'mosaic_grid':mosaic_grid,
        'blocksize':blocksize,
        'ref_block': (0,0),  # 0-based
        'transformation': {  # shift and magnification will 
			     # have default values
              'rotation':  (0, 5.0, 4.5, 5.3),  # Rotation in degrees
                                                # for each block.
                          },
                }
         
    mg = MosaicGeometry(geo_dict)

    # Return require objects
    # 
    return md,mg

.. _help_mdata:

Example 7: Ingest a list of numpy arrays using MosaicData
------------------------------------------------------------

In order to create a mosaic we need at least a MosaicData object to be used as input to the Mosaic class initialize function. Let's make a list of numpy arrays and a dictionary of the arrays locations in the mosaic.

The location of the data arrays is set with a dictionary of the corner coordinates containing 'amp_mosaic_coord' and 'amp_block_coord' keys, where 'amp_mosaic_coord' is a list tuples (x1,x2,y1,y2). (x1,y1) is the lower left, (x2,y2) is the right top corner with respect to the origin (0,0) at the lower left corner of the mosaic to be created. The 'amp_block_coord' is a list of tuples (x1,x2,y1,y2) describing the corners of each data array element but with origin as the lower left corner of each *block*. A *block* is defined as a subsection of the mosaic containing one or more data arrays; e.g. a detector array data having two readouts (amplifiers).


::

 import numpy
 # The directory for mosaic.py will probably 
 # change when the code goes into production.
 from gempy.library.mosaic import MosaicData

 # Make 4 data arrays of size nx:1024, ny:2048
 data = numpy.linspace(0.,1000.,1024*2048).reshape(2048,1024)
 data_list = [data*(-1)**k for k in numpy.arange(4)]

 # Image section are: (x1,x2,y1,y2)
 coords = {'amp_mosaic_coord': [(0,    1024, 0, 2048), (1024, 2048, 0, 2048),
                      (2048, 3072, 0, 2048), (3072, 4096, 0, 2048)],
           'amp_block_coord': [(0, 1024, 0, 2048), (0, 1024, 0, 2048),
                      (0, 1024, 0, 2048), (0, 1024, 0, 2048)]
          }

 # Now instantiate the MosaicData object
 data_object = MosaicData(data_list, coords)

.. _help_mgeo_example:

Example 8: Create a MosaicGeometry class object.
------------------------------------------------------------

Each data block might need to be corrected for shift, rotation and magnification. In this example we have four data blocks and the 'geo_dict' contains values for these parameters . There are 4 tuples for shift, one for each data block. The first tuple correspond to the values for the reference block *ref_block* with values shift:(0,0) to not shift, rotation: 0.0 to not rotate and magnification:1.0 to not magnify; all the rest of the list are values with respect to the reference block.


::

 from gempy.library.mosaic import MosaicGeometry

 geo_dict = {
    'transformation': {
           'shift':[(0,0),         (43.60, -1.24), 
                    (0.02, 41.10), (43.42, 41.72)], # List of (x,y) shift in pixel
           'rotation': (0.0,     -1.033606,
                        0.582767, 0.769542),        # List of degrees, counterwise
                                                    # w/r to the x_axis
           'magnification': (1.,     1.0013, 
                             1.0052, 1.0159),       # List of magnification
                        },
          # Gap dictionary:
          #         block: (x_gap,y_gap) in pixel
    'gaps':        {(0,0):(0,0), (1,0):(36,0), (2,0):(36,0), (3,0):(36,0)},
    'blocksize':   (1024,2048),        # (npix_x, npix_y)
    'mosaic_grid': (4,1),              # Number of blocks in x and number of rows. 
    'ref_block':   (0,0),        # Reference block (column,row) 0-based.
    'interpolator': 'linear',    # Interpolator
           }

 # Now instantiate the MosaicGeometry object
 geometry_object = MosaicGeometry(geo_dict)

.. _mos_base_example:

Example 9: Creates a mosaic using base class Mosaic.
------------------------------------------------------------


Use the *data_object* from Example 6 and *geometry_object* from Example 7 to instantiate a Mosaic object.  We print the shape of the output mosaic and display it -using ds9.  Make sure you have ds9 up and running.
::
 
 from numdisplay import display
 # The directory for mosaic.py will probably 
 # change when the code goes into production.
 from gempy.library.mosaic import Mosaic

 # Go to Example 6 and create the data_object, go to
 # example 7 and create the geometry_object.
 mo = Mosaic(data_object, geometry_object)
 #  Now produce a mosaic with the layout given by 'amp_mosaic_coord' and 'amp_block_coord'
 #  from 'data_object' attribute.
 mosaic_array = mo.mosaic_image_data()
 print mosaic_array.shape
 # display
 display(mosaic_array,frame=1)

.. _exam9:

Example 10: Display the mask 
------------------------------------------------------------

The Mosaic class method *mosaic_image_data* generates mask of the same shape as 
the output mosaic and with pixel value 0 for image data and 1 for no-data values
in the output mosaic. No-data values are gaps areas and those produced by transformation when the image is shifted and/or rotated.
::

 # 
 # display the mask for the mosaic in the previous example.
 display(mo.mask,frame=2,z1=0,z2=1.5)

.. _exam10:

Example 11: Transforms a block
------------------------------------------------------------

Using the data_object and geometry_object from Examples  6 and 7 create a Mosaic object, then transform the block (0,1) (the top left block).

The purpose of this example is to show the usage of the Mosaic method 'transform'.

::

 import numpy as np
 from gempy.library.mosaic import Mosaic, MosaicGeometry, MosaicData
 from numdisplay import display

 geo_dict = {
   'transformation': { # these tuples correspond to blocks:
                       #  (0,0)    (1,0)     (0,1)     (1,1)
       'shift':         [(0.,0.), (-10,20), (-10,20), (0,0)],
       'rotation':       (0.0,     0.0,       45.0,    45.0),
       'magnification':  (1.0,     1.0,        1.0,     0.5),
                     },
   'interpolator': 'linear',
        # gaps     block:  (x_gap,y_gap) (pixels)
   'gaps':        {(0,0):(0,0), (1,0):(20,0), (0,1):(0,30), (1,1):(20,30)}
   'blocksize':    (200,300), # number of pixels in x and in y.
   'ref_block':    (0,0),     # 0-base reference block
   'mosaic_grid':  (2,2)      # number of blocks in x and in y
                    }

 mosaic_geometry = MosaicGeometry(geo_dict)

 #   Make a rectangle (200,300) (wide,high).
 data = np.ones((300,200),dtype=np.float32)
 data = data*20   # make the background 20

 #   Make a four elements data_list (mosaic_grid).
 #   The blocks layout in the mosaic is:
 #   (0,0), (1,0), (0,1), (1,1)
 data_list = [data,data,data,data] 

 #   Inside each block, make a small box 50x50
 #   starting at (50,50) with value 100
 for k in range(4): 
      data_list[k][50:101,50:101] = 100. 
      # Mark the block borders with value 400
      data_list[k][:,0]  =400 
      data_list[k][:,199]=400 
      data_list[k][0,:]  =400 
      data_list[k][299,:]=400 

 #     Now create the MosaicData object
 mosaic_data = MosaicData(data_list)

 #      With these two objects we instantiate a Mosaic object
 mo = Mosaic(mosaic_data, mosaic_geometry)

 # Let take the block corresponding to the location (0,1) within the 
 # mosaic and transform. The values used are: shift: (-10,-20) in (x,y),
 # rotation: 45 degrees about the center and magnification: 1.0
 # (no magnification)
 trans_data = mo.transform(mo.data_list[2],(0,1))

 #  ---- Now display both blocks to visually see the difference 
 #       between original and transformed blocks.
 # display input data
 display(mo.data_list[2],frame=1)
 # display transform data
 display(trans_data,frame=2)

.. _exam11:

Example 12: Use set_transformation function
---------------------------------------------

When transforming a block, a default interpolation function is used (linear). The available functions are: 'nearest', 'linear', and 'spline' with order (0-5). 

The purpose of this example is to illustrate the effects on a transformed block when
resetting the interpolation function.

The method to reset the interpolation function is:
::

 mo.set_transformation_function(function_name, order)

Create 2 ndarrays list and mark a strip of the 2nd ndarray with a higher value.  Set the Geometry dictionary with 'rotate' this ndarray by 5 degrees. Now we create the mosaic with  default interpolation function and again with the 'spline' function of order 5. We plot a column from each image.
::
	
 import numpy as np
 from gempy.library.mosaic import Mosaic,MosaicGeometry, MosaicData
 from matplotlib import pyplot as pl

 geo_dict = {
     'transformation': {  # shift and magnification will 
			  # have default values
          'rotation':  (0.0, 5.),
                       },
     'blocksize':   (100,100),
     'mosaic_grid':  (2,1)
            }

 # With this dictionary create a MoaicGeometry object
 geometry_object = MosaicGeometry(geo_dict)

 #   Make an ndarray
 data = np.zeros((100,100),dtype=np.float32)
 #  put a stripe of 5 rows with value 5
 data[45:50,:] = 5

 #   Make an 2x1 array with this rectangle.
 data_list = [data,data] 

 # Create a MosaicData object
 data_object = MosaicData(data_list)

 #   With these two objects we instantiate a Mosaic object
 mo = Mosaic(data_object, geometry_object)

 #   Finally make the mosaic
 mosaic_linear = mo.mosaic_image_data()

 #   Now reset the interpolator function the spline or order 5.
 mo.set_interpolator('spline',spline_order=5)

 # Create the mosaic 
 mosaic_spline = mo.mosaic_image_data()

 # Now plot across the stripes
 pl.plot(mosaic_linear[:,140])
 pl.plot(mosaic_spline[:,140])

 # The difference between the 2 plots is the edge effect at the
 # low and high stripe corners plot due to interpolation.


