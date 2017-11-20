.. include examples

.. _mosad_class:

MosaicAD
========

.. _mos_intro:

**MosaicAD** as a subclass of Mosaic and extends its functionality by providing 
support for:

- AstroData objects with more than one extension name; i.e. 'SCI', 'VAR', 'DQ'

- Creating output mosaics in AstroData objects

- Updating the WCS information in the output AstroData object mosaic header

- A user_function as a parameter to input data and geometric values of the 
  individual data elements.

- A user_function (already written) to support GMOS and GSAOI data

.. _mosad_input:

How to use  MosaicAD class
--------------------------

1) Create an AstroData object with the input FITS file containing GMOS or 
   GSAOI data.

2) Instantiate a MosaicAD class object with the user supplied 
   :ref:`mosaic function <user_function_ad>` name as input parameter.

3) Now generate :ref:`mosaics ndarrays <mosad_array>` or 
   :ref:`AstroData objects <mos_associated>`.

To instantiate a MosaicAd object:
 ::

  # Assuming you have module 'mosaicAD.py' and 'gemMosaicFunction.py' in your
  # directory tree and is part of your PYTHONPATH environment variable.

  from gempy.mosaic.mosaicAD import MosaicAD
  from gempy.mosaic.gemMosaicFunction import gemini_mosaic_function

  mosad = MosaicAD(ad, mosaic_ad_function, column_names=def_columns)

**MosaicAD Input parameters**

- ad
    Input AstroData object

- mosaic_ad_function
    Is a user supplied function that will act as an interface to the particular 
    astrodata object, e.g., knows which keywords represent the coordinate systems 
    to use and whether they are binned or not, or which values in the geometry 
    look up table require to be binned. For Gemini GMOS and GSAOI data there is 
    a user function available 'gemini_mosaic_function' in a module 
    *gemMosaicFunction.py*. If you have other data, see the Example section for 
    :ref:`'Write a user_function <user_function_ad>`

- column_names
    Dictionary with bintable extension names that are associates with the input 
    images extension names. The extension name is the key and the value a tuple: 
    (X_pixel_columnName, Y_pixel_columnName, RA_degrees_columnName, 
    DEC_degrees_columnName). Example::

      column_names = { 
        'OBJCAT': ('X_IMAGE', 'Y_IMAGE', 'X_WORLD', 'Y_WORLD'),
        'REFCAT': (None, None, 'RAJ2000', 'DEJ2000')
      }

    The dictionary has two table entries: the 'OBJCAT' extension name with 
    four values which are the column names in the table and the other table if 
    'REFCAT' containing two column names for RA and DEC of objects in the field.

                        
:ref:`MosaicAD example <mosad_array>`

**MosaicAD class Attributes**

These attributes are in addition to the Mosaic class. 

- .log gemini_python logger object.
- .ad  AstroData object.
- .column_names column names for catalog merge.
- .jfactor <list>, Jacobians applied to interpolated pixels.
- .mosaic_shape <tuple>, output shape of the mosaic array.

.. _mosad_asad:


MosaicAD Methods
----------------

as_astrodata()
**************

This function has the same functionality as the *mosaic_image_data* function 
but, results in a fullly formed astrodata object returned to the caller. 
WCS information in the headers of the IMAGE extensions and any pixel coordinates 
in the output BINTABLEs will be updated appropriately. As_astrodata returns an 
AstroData object. Notice that as_astrodata can return more than one mosaic is 
the input AstroData object contains different image extension names, 
e.g. a MEF file with 'SCI', 'VAR' and 'DQ' image extensions.

Usage:
 ::
  
  from astrodata import AstroData
  # The directory mosaicAD.py and gemMosaicFunction.py modules
  # will probably change when the code goes into production.
  from gempy.adlibrary.mosaicAD import MosaicAD
  #     This is a user function available for your use,
  #     it supports GMOS and GSAOI data
  from gempy.mosaic.gemMosaicFunction import gemini_mosaic_function


  # Instantiate an AstroData object using a GMOS FITS file.
  ad = astrodata.open('S20030201S0173.fits')

  # Instantiate the MosaicAD object. The 'gemini_mosaic_function' will
  # be executed inside using the 'ad' object to return the MosaicData and
  # MosaicGeometry objects.

  mosad = MosaicAD(ad, gemini_mosaic_function)

  # Using the 'mosad' object executes the method as_astrodata returning an
  # AstroData object.

  adout = mosad.as_astrodata(block=None, tile=False, doimg=False, return_ROI=True,
                             update_with='wcs')


**as_astrodata parameters**

::

 as_astrodata(block=None, doimg=False, tile=False, return_ROI=True, update_with='wcs')

- block: <2-tuple>. Default is None.
    Allows a specific block to be returned as the output mosaic. The tuple 
    notation is (col,row) (zero-based) where (0,0) is the lower left block.  
    The blocks layout is given by the attribute mosaic_grid.

- doimg: <bool>. Default is False.
    Specifies that *only* the SCI image data are tiled or transformed (see 
    parameter, ``tile``). False indicates all image extensions are processed, 
    i.e. all SCI, VAR, DQ extensions.

- tile: <bool>. Default is False
    If True, the mosaics returned are not corrected for shifting, rotation or 
    magnification.

- return_ROI: <bool>. Default is True
    Returns the minimum frame size calculated from the location of the 
    amplifiers in a given block. If False uses the blocksize value.

- update_catalog_method: ('wcs').
    Specifies if the X and Y pixel coordinates of any source positions in the 
    BINTABLEs are to be recalculated using the output WCS and the sources R.A.  
    and Dec. values within the table. If set to 'transform' the updated X and Y 
    pixel coordinates will be determined using the transformations used to mosaic 
    the pixel data. In the case of tiling, a shift is technically being applied 
    and therefore update_catalog_method='wcs' should be set internally (Not yet 
    implemented).

:ref:`as_astrodata example <asastro_ex>`

.. _mosad_imdata:

mosaic_image_data()
*******************

Method to layout the blocks of data in the output mosaic grid.  Correction for 
rotation, shifting and magnification is performed with respect to the reference 
block.  A Mask is also created containing value zero for positions where there 
are pixel data and one for everywhere else, like gaps and areas of no-data due 
to shifting when transforming the data.
 
Usage:
::

 mosad = MosaicAD(ad, gemini_mosaic_function)
 mosaic = mosad.mosaic_image_data(tile=False, block=None, return_ROI=True)

**Input parameters**

- tile: <bool> (default is False)
    If True, the mosaics returned are not corrected for shifting and rotation.

- block: <2-tuple> (default is None)
    Allows a specific block to be returned as the output mosaic.  The tuple 
    notation is (col,row) (zero-based) where (0,0) is the lower left block of the 
    output mosaic.

- return_ROI: <bool> (default is True).
    Returns the minimum frame size calculated from the location of the 
    amplifiers in a given block. If False uses the blocksize value.

**Output**

- mosaic: ndarray with mosaic data.

:ref:`mosaic_image_data example <asastro_ex>`

..
   .. _mosad_merge:

   MosaicAD.merge_table_data function
   -------------------------------------

   Merges input BINTABLE extensions that matches the extension name given in the 
   parameter *tab_extname*. Merging is based on RA and DEC columns and the repeated 
   RA, DEC values in the output table are removed. The column names for pixel and 
   equatorial coordinates are given in a dictionary with class attribute name: 
   *column_names*

    Usage
    ::

     mosad = MosaicAD(ad, gemini_mosaic_function, column_names='default')

	   # column_names is a dictionary with default values:
	   # column_names = {'OBJCAT': ('Xpix', 'Ypix', 'RA', 'DEC'),
	   #                 'REFCAT': (None, None, 'RaRef', 'DecRef')} 
     adout = mosad.merge_table_data(ref_wcs, tile, tab_extname, block=None,
			update_catalog_method='wcs')



   - ref_wcs: Pywcs object containing the WCS from the output header

   - tile: Boolean. 
       If True, the function will use the gaps list of values for tiling, if False 
   it uses the Transform list of gap values.

   - tab_extname: Binary table extname

   - block: default is (None).
       Allows a specific block to be returned as the output mosaic. The tuple 
   notation is (col,row) (zero-based) where (0,0) is the lower left block in 
   the output mosaic.

   - update_catalog_method
       If 'wcs' use the reference extension header WCS to recalculate the x,y 
   values. If 'transform', apply the linear equations using to correct the x,y 
   values in each block.

   **Output**

   - adout: AstroData object with the merged output BINTABLE

.. _mosad_jfactor:

calculate_jfactor()
*******************

Calculate the ratio of reference input pixel size to output pixel size for each 
reference extension in the AstroData object.  In practice this ratio is formulated 
as the determinant of the WCS transformation matrix.  This is the ratio that we will 
applied to each pixel to conserve flux in an image after magnification in the 
transformation.  
 
 Usage:
 ::

  MosaicAD.calculate_jfactor()


**Justification**

In general CD matrix element is the ratio between partial derivative of the 
world coordinate (ra,dec) with respect to the pixel coordinate (x,y). We have 4 
elements in the FITS header CD1_1, CD1_2, CD2_1 and CD2_2 that defines a CD matrix.

For an adjacent image in the sky (GMOS detectors 1,2,3 for example), the 
cd matrix elements will have slightly different values.

Given the CD matrices from adjacent fields, the jfactor is calculated as the 
dot product of the inverse of one of the matrices times the other matrix.

**Output**

- MosaicAD.jfactor
    The mosad attribute list is filled with one floating value per block.

.. _mosad_getdl:

get_data_list(attr)
*******************

Returns a list of image data for all the ad. It assumes that the input 
AstroData Descriptor *data_section* has been defined for this astrodata type, 
i.e. GMOS or GSAOI.

 Usage
 ::

  sci_data_list = MosaicAD.get_data_list('data')
  var_data_list = MosaicAD.get_data_list('variance')
  dq_data_list  = MosaicAD.get_data_list('mask')

**Output**

- data_list. List of image data ndarrays.

.. _mosad_info:

info()
******

 Creates a dictionary with coordinates, amplifier and block information:
 ::
 
  The keys for the info dictionary are:

  filename
       The ad.filename string
  amps_per_block
       Number of amplifiers per block
  amp_mosaic_coord: (type: List of tuples (x1,x2,y1,y2))
       The list of amplifier location within the mosaic.  
       These values do not include the gaps between the blocks
  amp_block_coord (type: list of tuples (x1,x2,y1,y2))
       The list of amplifier indices within a block.
  interpolator
       Interpolator name
  ref_extname
       Reference extension name
  ref_extver
       Reference extension version
  reference_block
       Reference block tuple (col,row)

 Usage
 ::

  dictionary = MosaicAD.info()

**Output**

- MosaicAD.info. Dictionary with the above information

 
