.. _mosad_class:

MosaicAD
========

.. _mos_intro:

**MosaicAD** as a subclass of Mosaic extends its functionality by providing support for:

- AstroData objects with more than one extension name; i.e. 'SCI', 'VAR', 'DQ'

- Associating object catalogs in BINARY FITS extensions with the image extensions

- Creating output mosaics and merge tables in AstroData objects

- Updating the WCS information in the output AstroData object mosaic header

- A user_function as a parameter to input data and geometric values of the individual data elements

- A user_function (already written) to support GMOS and GSAOI data

.. _mosad_input:

How to use  MosaicAD class
--------------------------

1) Create an AstroData object with the input FITS file containing a GMOS or GSAOI data.

2) Instantiate a MosaicAD class object with the user supplied :ref:`mosaic function <user_function>` name as input parameter.

3) Now generate :ref:`mosaics ndarrays <mosad_array>` or :ref:`AstroData objects <mos_associated>`.

To instantiate a MosaicAd object:
 ::
  
  # Assuming you have module 'mosaicAD.py' and 'gemMosaicFunction.py' in your
  # directory tree and is part of your PYTHONPATH environment variable.

  from gempy.adlibrary.mosaicAD import MosaicAD
  from gempy.gemini.gemMosaicFunction import gemini_mosaic_function

  mosad = MosaicAD(ad, gemini_mosaic_function, ref_extname='SCI', column_names='default')
        
**MosaicAD Input parameters**

- ad: Input AstroData object

- gemini_mosaic_function
    Is a user supplied function that will act as an interface to the particular ad, e.g., knows which keywords represent the coordinate systems to use and whether they are binned or not, or which values in the geometry look up table require to be binned. For Gemini GMOS and GSAOI data there is a user function available 'gemini_mosaic_function' in a module *gemMosaicFunction.py*. If you have other data, see the Example section for 'Write a user_function'.
    
- ref_extname 
    Is the IMAGE EXTNAME that should be used as the primary reference when reading the ad data arrays. Default value is 'SCI'.

- column_names 
    Dictionary with bintable extension names that are associates with the input images extension names. The extension name is the key and the value a tuple: (X_pixel_columnName, Y_pixel_columnName, RA_degrees_columnName, DEC_degrees_columnName)
     ::

      Example:
      column_names = {'OBJCAT': ('Xpix', 'Ypix', 'RA', 'DEC'),
                     'REFCAT': (None, None, 'RaRef', 'DecRef')}

    The dictionary has two table entries: the 'OBJCAT' extension name with four values which are the column names in the table and the other table if 'REFCAT' containing two column names for the RA and DEC of objects in the field.

                        
:ref:`MosaicAD example <mosad_array>`

**MosaicAD class Attributes**

These attributes are in addition to the Mosaic class. 

- log. Logutils object 
- ad. AstroData object
- ref_extname
    Is the IMAGE EXTNAME that should be used as the primary reference when reading the ad data arrays. Default value is 'SCI'.
- extnames. Contains all extension names in ad
- im_extnames. All IMAGE extensions names in ad
- tab_extnames. All BINTABLE extension names in ad
- process_dq. Boolean. True when transforming a DQ image data
- associated_tab_extns
    List of binary extension names that have the same number and values of extvers as the reference extension name.
- associated_im_extns
    List of image extension names that have the same number and values of extvers as the reference extension name.
- non_associated_extns
    List of remaining extension names that are not in the above 2 lists.

.. _mosad_asad:

MosaicAD.as_astrodata function
-------------------------------

This function has the same functionality as the *mosaic_image_data* function but, in addition it the merges associated BINTABLEs and all other non-associated extensions of any other type. WCS information in the headers of the IMAGE extensions and any pixel coordinates in the output BINTABLEs will be updated appropriately. As_astrodata returns an AstroData object. Notice that as_astrodata can return more than one mosaic is the input AstroData object contains different image extension names; e.g. a MEF file with 'SCI', 'VAR' and 'DQ' image extensions.

 Usage:
 ::
  
  from astrodata import AstroData
  # The directory mosaicAD.py and gemMosaicFunction.py modules
  # will probably change when the code goes into production.
  from gempy.adlibrary.mosaicAD import MosaicAD
  #     This is a user function available for your use,
  #     it supports GMOS and GSAOI data
  from gempy.gemini.gemMosaicFunction import gemini_mosaic_function


  # Instantiate an AstroData object using a GMOS FITS file.
  ad = AstroData('gS20030201S0173.fits')

  # Instantiate the MosaicAD object. The 'gemini_mosaic_function' will
  # be executed inside using the 'ad' object to return the MosaicData and
  # MosaicGeometry objects.

  mosad = MosaicAD(ad, gemini_mosaic_function)

  # Using the 'mosad' object executes the method as_astrodata returning an
  # AstroData object.

  adout = mosad.as_astrodata(extname=None, tile=False, block=None, return_ROI=True,
                   return_associated_bintables=True, return_non_associations=True,
                   update_catalog_method='wcs')


**as_astrodata parameters**

::

 as_astrodata(extname=None, tile=False, block=None, return_ROI=True,
              return_associated_bintables=True, return_non_associations=True,
              update_catalog_method='wcs')

- extname: (string). Default is None
    If None, mosaic all IMAGE extensions. Otherwise, only the given extname. This becomes the ref_extname.

- tile: (boolean). Default is False
    If True, the mosaics returned are not corrected for shifting, rotation or magnification.

- block: (tuple). Default is None
    Allows a specific block to be returned as the output mosaic. The tuple notation is (col,row) (zero-based) where (0,0) is the lower left block.  The blocks layout is given by the attribute mosaic_grid.

- return_ROI: (boolean). Default is True
    Returns the minimum frame size calculated from the location of the amplifiers in a given block. If False uses the blocksize value.

- return_associated_bintables: (boolean). Default is True
    If a bintable is associated to the ref_extname then is returned as a merged table in the output AD.  If False, they are not returned in the output AD.

- return_non_associations: (boolean). Default is True
    Specifies whether to return extensions that are not deemed to be associated with the ref_extname.

- update_catalog_method: ('wcs').
    Specifies if the X and Y pixel coordinates of any source positions in the BINTABLEs are to be recalculated using the output WCS and the sources R.A.  and Dec. values within the table. If set to 'transform' the updated X and Y pixel coordinates will be determined using the transformations used to mosaic the pixel data. In the case of tiling, a shift is technically being applied and therefore update_catalog_method='wcs' should be set internally (Not yet implemented).

:ref:`as_astrodata example <asastro_ex>`

.. _mosad_imdata:

MosaicAD.mosaic_image_data function
--------------------------------------

Method to layout the blocks of data in the output mosaic grid.  Correction for rotation, shifting and magnification is performed with respect to the reference block.  A Mask is also created containing value zero for positions where there are pixel data and one for everywhere else, like gaps and areas of no-data due to shifting when transforming the data.
 
Usage:
::

 mosad = MosaicAD(ad, gemini_mosaic_function)
 mosaic = mosad.mosaic_image_data(extname='SCI',tile=False, 
                                   block=None,return_ROI=True)

**Input parameters**

- extname: (default 'SCI'). Extname from AD to mosaic.

- tile: (boolean)
    If True, the mosaics returned are not corrected for shifting and rotation.

- block: default is (None)
    Allows a specific block to be returned as the output mosaic.  The tuple notation is (col,row) (zero-based) where (0,0) is the lower left block of the output mosaic.

- return_ROI: (True).
    Returns the minimum frame size calculated from the location of the amplifiers in a given block. If False uses the blocksize value.

**Output**

- mosaic: ndarray with mosaic data.

:ref:`mosaic_image_data example <asastro_ex>`

.. _mosad_merge:

MosaicAD.merge_table_data function
-------------------------------------

Merges input BINTABLE extensions that matches the extension name given in the parameter *tab_extname*. Merging is based on RA and DEC columns and the repeated RA, DEC values in the output table are removed. The column names for pixel and equatorial coordinates are given in a dictionary with class attribute name: *column_names*

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
    If True, the function will use the gaps list of values for tiling, if False it uses the Transform list of gap values.

- tab_extname: Binary table extname

- block: default is (None).
    Allows a specific block to be returned as the output mosaic. The tuple notation is (col,row) (zero-based) where (0,0) is the lower left block in the output mosaic.

- update_catalog_method
    If 'wcs' use the reference extension header WCS to recalculate the x,y values. If 'transform', apply the linear equations using to correct the x,y values in each block.

**Output**
  
- adout: AstroData object with the merged output BINTABLE

.. _mosad_jfactor:

MosaicAD.calculate_jfactor function
------------------------------------

 Calculate the ratio of reference input pixel size to output pixel size for each reference extension in the AstroData object.  In practice this ratio is formulated as the determinant of the WCS transformation matrix.  This is the ratio that we will applied to each pixel to conserve flux in an image after magnification in the transformation.  
 
 Usage:
 ::

  MosaicAD.calculate_jfactor()


**Justification**

 In general CD matrix element is the ratio between partial derivative of the world coordinate (ra,dec) with respect to the pixel coordinate (x,y). We have 4 elements in the FITS header CD1_1, CD1_2, CD2_1 and CD2_2 that defines a CD matrix.

 For an adjacent image in the sky (GMOS detectors 1,2,3 for example), the cd matrix elements will have slightly different values.

 Given the CD matrices from adjacent fields, the jfactor is calculated as the dot product of the inverse of one of the matrices times the other matrix.

**Output**

- MosaicAD.jfactor
    The mosad attribute list is filled with one floating value per block.

.. _mosad_upd:

MosaicAD.update_data function
-------------------------------

 Replaces the data_list attribute in the mosaic_data object with a new list containing ndarrays from the AD extensions 'extname'.The attribute data_list is updated with the new list.

 Usage
 ::

  MosaicAD.update_data(extname)   


**Input parameter**

- extname
    Reads all the image extensions from AD that matches the extname.

**Output**

- MosaicAD.data_list
   The MosaicAD object attribute is update with the new data list.


.. _mosad_mkasso:

MosaicAD.make_associations function
-------------------------------------

 This determines three lists: one list of IMAGE extension EXTNAMEs and one of BINTABLE extension EXTNAMEs that are deemed to be associated with the reference extension. The third list contains the EXTNAMEs of extensions that are not deemed to be associated with the reference extension. The definition of association is as follows: given the ref_extname has n extension versions (EXTVER), then if any other EXTNAME has the same number and exact values of EXTVER as the ref_extname these EXTNAMEs are deemed to be associated to the ref_extname.

 Usage
 ::

  MosaicAD.make_associations()  

**Output**

- MosaicAD.associated_tab_extns
    List of associated BINTABLE extension names
- MosaicAD.associated_im_extns
    List of associated IMAGES extension names
- MosaicAD.non_associated_extns
    All the rest of extensions in mosad.extnames that are not associated_tab_extns nor associated_im_extns

.. _mosad_gextn:

MosaicAD.get_extnames function
---------------------------------

 Form two dictionaries (images and bintables) with key the EXTNAME value and values the corresponding EXTVER values in a list.  E.g.: {'VAR': [1, 2, 3, 4, 5, 6], 'OBJMASK': [1, 2, 3, 4, 5, 6]}

 Usage
 ::

  MosaicAD.get_extnames() 

**Output**

- MosaicAD.extnames
    Dictionary with extname as key and the corresponding extver's as a list.

.. _mosad_getdl:

MosaicAD.get_data_list function
---------------------------------

 Returns a list of image data for all the extname extensions in ad.  It assumes that the input AstroData object Descriptor *data_section* has been defined for this astrodata type; i.e. GMOS or GSAOI.

 Usage
 ::

  data_list = MosaicAD.get_data_list()  

**Output**

- data_list. List of image data ndarrays.

.. _mosad_info:

MosaicAD.info function
-------------------------

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

 
