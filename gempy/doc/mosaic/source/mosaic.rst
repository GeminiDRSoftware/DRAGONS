.. _mos_class:

Mosaic
------

**Mosaic** is a base class that provides functionality to layout a list of data :ref:`ndarrays <why_ndarray>` of the same size into an output mosaic. The main characteristics are:

- Input data and their location in the output mosaic is done via :ref:`MosaicData <help_mdata>` objects

- Information about gaps between the :ref:`blocks <block_def>` and :ref:`transformation <mos_transf>` is given by the :ref:`MosaicGeometry <mos_geom>` object

- Mosaic object can generate :ref:`masks <mos_mask>` associated with the output mosaic

- The interpolated function used in the transformation can be reset via a Mosaic class function

- Preserve flux when transforming a block


.. _mos_works:

How to use the Mosaic class
----------------------------

The basic steps to generate a mosaic using the Mosaic class are: 

 1) :ref:`Handling input data <handling_input>`
 2) :ref:`Describe the coordinates of each of the input data elements <desc_coords>`
 3) :ref:`Characterize the blocks' geometry <block_geometry>`

Out of these, the input data list is the only requirement which will result in a horizontal tiling of each of the input data elements.

.. _handling_input:

**1) Handling input data**
  
The possible ways to obtain a python list of ndarrays (data_list) suitable for Mosaic are:

- Creating a data_list from a FITS file. For example: read a FITS file with three image extensions using pyfits to create the list of numpy arrays (aka ndarrays):

 ::

  import pyfits

  fits = pyfits('kp445403.fits')

  # Read image extension 1,2 and 3.
  data_list = [fits[k].data for k in range(1,4)]

- Or by creating your own data list:

 ::

   # Make 4 data arrays of size nx:1024, ny:2048
   data = numpy.linspace(0.,1000.,1024*2048).reshape(2048,1024)
   data_list = [data*(-1)**k for k in numpy.arange(4)]

- Or by making use of the gemMosaicFunction function to generate a MosaicData and a MosaicGeometry objects from GMOS/GSAOI data. See :ref:`Example <mosad_array>`.

.. _desc_coords:

**2) Describe the coordinates of each data list element (amplifier)**

Each data element coordinate description contains two sets of coordinates given by (x1,x2,y1,y2) where x1 and x2 are the start and end column pixel location; y1 and y2 are the start and end row location of the data piece with respect to a given origin. One tuple origin is with respect to the lower left corner of the block containing the data, the other tuple origin is with respect to the lower left corner of the mosaic. The coordinates values are zero-based and the end values x2,y2 are none inclusive.

These two tuple lists are given as a dictionary callied coords, with keys: *amp_mosaic_coord* with origin the lower left corner of the mosaic and *amp_block_coord* with origin the lower left corner of the block. Here is an example of the dictionary. The order on these lists is the same as the input list of ndarrays (data_list) order.

 ::

  # Coordinate description of a data list with four amplifier 
  # ndarrays of size 1024 columns by 2048 rows.
 
  # Image sections are: (x1,x2,y1,y2)
  coords = {'amp_mosaic_coord':
                  [(0,    1024, 0, 2048), (1024, 2048, 0, 2048),
                   (2048, 3072, 0, 2048), (3072, 4096, 0, 2048)],

            'amp_block_coord':
                  [(0, 1024, 0, 2048), (0, 1024, 0, 2048),
                   (0, 1024, 0, 2048), (0, 1024, 0, 2048)]
             }


.. _block_geometry:

**3) Geometry description of input data and output mosaic**

Use a geometry dictionary to list block properties such as block separation (gaps) in the mosaic and transformation values for each block with respect to a reference block, etc. :ref:`Here <mos_geom>` is the list of all the geometry keys. This is an example
of a typical geometry dictionary:

 ::

  geo_dict = {
    'transformation': {
                  # The order here is the same as the order given in the
                  # tile and transform gaps ('gap_dict').
           'shift':[(0,0),         (43.60, -1.24), 
                    (0.02, 41.10), (43.42, 41.72)], # List of (x,y) shift in pixel

           'rotation': (0.0,     -1.033606,
                        0.582767, 0.769542),        # List of degrees, counterwise
                                                    # w/r to the x_axis
           'magnification': (1.,     1.0013, 
                             1.0052, 1.0159),       # List of magnification
                        },
    'gap_dict': {                                   # (x_gap,y_gap) in pixels
           # Key values are block location (0-based) (column,row) with 
           # respect to the lower left block in the mosaic.
       'tile_gaps': {(0,0):(15,25), (1,0):(15,25),
                     (0,1):(15,25), (1,1):(15,25)}, 

       'transform_gaps': {(0,0):(14,23.4), (1,0):(14.0,23.4),
                          (0,1):(14,20.4), (1,1):(12.6,23.4)},
            }, 
    'blocksize':   (1024,2048),        # (npix_x, npix_y)
    'mosaic_grid': (4,1),              # Number of blocks in x and number of rows. 
    'ref_block':   (0,0),        # Reference block (column,row) 0-based.
    'interpolator': 'linear',    # Interpolator
           }

    # NOTE: if the gaps values are the same for tile_gaps and transform_gaps
    #       then instead of the 'gap_dict' use the 'gaps' key. e.g.
    'gaps': {(0,0):(15,25), (1,0):(15,25),
             (0,1):(15,25), (1,1):(15,25)},

For simplicity if you want to create a tile mosaic, the only requirement then
if the *blocksize* and the *mosaic_grid*.

In practical terms if you have GMOS or GSAOI data all this work is done for you
by using the gemini_mosaic_function in the module gemMosaicFunction.py

.. _mos_data:

mosaic.MosaicData Class
-----------------------

MosaicData is a class that provides functionality to verify and store a list of ndarrays. An object of this class is used as input to the initialize function of the Mosaic class.

To create a MosaicData object:
 ::

  mosaic_data = MosaicData(data_list=None, coords=None)

**Input parameters**

- data_list
    List of ndarrays with pixel data. The ordering system is given by *coords* as a list of coordinates describing the layout of the ndarrays into blocks and the layout of the blocks into the mosaic. If data_list is None and coords is None, the user gets an object with attributes names that can be set.

- coords
    A dictionary with keys ‘amp_mosaic_coord’ and ‘amp_block_coord’. The ‘amp_mosaic_coord’ values contain a list of tuples describing the corners of the ndarrays, i.e., (x1,x2,y1,y2) with respect to the mosaic lower left corner (0,0). The ‘amp_block_coord’ values contain a list of tuples. describing the corners of the ndarrays, i.e., (x1,x2,y1,y2) with respect to the block lower left corner (0,0). Notice that we can have more than one ndarray per block. If coords is None and the object contains only the data_list attribute, when used in Mosaic, it will result in an output tile array arrange in a horizontal manner.

**Attributes**

- data_list. Same as input
- coords. Same as input
 

.. _mos_geom:

mosaic.MosaicGeometry Class
---------------------------

The MosaicGeometry class provides functionality to verify the input geometry elements and set all the require attributes. A MosaicGeometry object is not necessary to produce a mosaic, reulting in an horizontal stack of the blocks. If an object is created, the only required attributes are: *blocksize* and *mosaic_grid*.

To create a MosaicData object:
 ::

  mosaic_geometry = MosaicGeometry(dictionary)

**Input Parameter**

- dictionary: A dictionary with the following keys:

  (NOTE: blocksize and mosaic_grid are required to produce a mosaic)

  blocksize
      Tuple of (npixels_x,npixels_y). Size of the block.
  mosaic_grid
      Tuple (ncols,nrows). Number of blocks per row and
      number of rows in the output mosaic array.

  transformation: dictionary with 
      'shift'
          List of tuples (x_shift,y_shift). Amount in pixels (floating
          numbers) to shift to align with the ref_block. There
          are as many tuples as number of blocks.
      'rotation'
          (Degrees). List of real numbers. Amount to rotate each
          block to align with the ref_block. There are as
          many numbers as number of blocks. The angle is counter
          clockwise from the x-axis.
      'magnification'
          List of real numbers. Amount to magnify each
          block to align with the ref_block. There are as
          many numbers as number of blocks. The magnification is abouti
          the block center

  ref_block
      Reference block tuple. The block location (x,y) coordinate
      in the mosaic_grid. This is a 0-based tuple. 'x' increases to
      the right, 'y' increases in the upwards direction.
  interpolator
      (String). Default is 'linear'. Name of the transformation
      function used for translation,rotation, magnification of the
      blocks to be aligned with the reference block. The possible
      values are: 'linear', 'nearest', 'spline'.
  spline_order
      (int). Default 3. Is the 'spline' interpolator order. Allow 
      values are in the range [0-5].
  gap_dict 
       A dictionary of dictionaries of the form:

       ::

        gap_dict = { 'tile_gaps': {(col,row): (x_gap,y_gap),...},
                     'transform_gaps': 
                                  {(col,row): (x_gap, y_gap),...}
                   }

        The '(col,row)' tuple is the block location with (0,0) being
        the lower left block in the mosaic.

        The '(x_gap, y_gap)' tuple is the gap in pixels at the left of
        the block (x_gap) and at the bottom of the block (y_gap); hence
        the (0,0) block will have values (0,0) for gaps.

        For some instruments the gaps are different depending whether we 
        produce a mosaic in 'tile' or 'transform' mode.

  gaps
       If the 'gap_dict' has the same values for 'tile_gaps' and
       'transform_gaps', then use this simpler entry instead:
       ::

        gaps = {(col,row): (x_gap,y_gap),...},

 
**Class Attributes**

- blocksize:    Same as input
- mosaic_grid:  Same as input
- interpolator: Same as input
- ref_block:    Same as input
- transformation:  Same as input

.. _inst_class:

mosaic.Mosaic
-------------

To instantiate a Mosaic object you need to have at least a list of ndarrays of the same same size contained in a MosaicData object.

 ::

  from gempy.library.mosaic import Mosaic

  mosaic = Mosaic(mosaic_data, mosaic_geometry=None, dq_data=False)

**Input parameters**

- mosaic_data
    MosaicData class object containing the data_list and list of coordinates. The members of this class are: data_list, coords. (see :ref:`example <help_mdata>` for details).
- mosaic_geometry
    MosaicGeometry class object (optional). See :ref:`example <help_mgeo_example>` on how to set it up.
- dq_data
    If the MosaicData contains  DQ data type, then this parameter should be set to True to properly transform the individual bit-planes; otherwise whole pixel transformation is done.

**Mosaic Class Attributes**

- data_list: Same in MosaicData input parameter
- coords: Same in MosaicData input parameter
- geometry: MosaicGeometry object
- data_index_per_block 
    Dictionary to contain the list indices of each data_list element that falls in one block. The dictionary key is the block tuple.
- return_ROI
    Boolean to set a minimum area enclosing all the data_list elements in the mosaic.
- mask
    Mask array for the resulting mosaic.  0: good data, 1: no-data
- jfactor: Conservation flux factor.

.. _mos_imdata:

mosaic.mosaic_image_data function
----------------------------------

Method to layout the blocks of data in the output mosaic grid.  Correction for rotation, shifting and magnification is performed with respect to the reference block.  A Mask is also created containing value zero for positions were there are pixel data and one for everywhere else -like gaps and areas of no-data due to shifting when transforming the data.

 Usage:
 ::

  mosaic = mosaic_image_data(tile=False,block=None,return_ROI=True)

**Input parameters**

- tile. (boolean)
    If True, layout the block in the mosaic grid with no correction for rotation nor shift.  Gaps are included.

- block. (tuple)
    Allows a specific block to be returned as the output mosaic. The tuple notation is (col,row) (zero-based) where (0,0) is the lower left block.  The blocks layout is given by the attribute mosaic_grid.

- return_ROI. (boolean)
    Flag to use the minimum frame enclosing all the block_data elements.

**Output:**
     An ndarray with the mosaic. The Mask created is available as an attribute with name 'mask'.

.. _mos_mask:

Mosaic masks 
----------------

Masks are ndarrays products from the Mosaic class. When a mosaic is produced, a mask is also created with the same shape but with value zero for image pixel in the mosaic and value one for non-data pixel such as gaps areas in between blocks and no-data areas generated when transforming the blocks. If 'mos_obj' is the Mosaic object created when instantiating the class, then 'mos_obj.mask' is the ndarray with the mask, available after the 'mosaic_image_data' method is invoked. 

.. _mos_transform:

mosaic.set_transformations function
-----------------------------------

Instantiates the Transformation class objects for each block that needs correction for rotation, shift and/or magnification. Set a dictionary with (column,row) as a key and value the Transformation object.


.. _mos_set_trans_function:

mosaic.set_interpolator function
-----------------------------------

Changing the interpolation method to use when correcting the blocks for rotation, translation and magnification. 

USAGE
::

 mo.set_interpolator(tfunction='linear',spline_order=2)

**Input parameters**

- tfunction
      Interpolator name. The supported values are: 'linear', 'nearest', 'spline'.

- spline_order
      Used when tfunction is 'spline' and is the order of the spline interpolator.  (default is 2). Allowed values are in the range [0-5], where order zero is equivalent to a 'linear' interpolator, one is equivalent to a 'nearest' interpolator.

Here is an :ref:`Example <exam11>`  on how to use *set_interpolator*.
