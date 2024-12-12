.. include examples

.. _mosad_class:

MosaicAD
========

.. _mos_intro:

**MosaicAD** as a subclass of the Mosaic class and extends its functionality by
providing support for:

- AstroData objects with a full suite of extensions, including SCI, VAR, DQ,
  object masks, and tables.

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

  mos_ad = MosaicAD(ad, gemini_mosaic_function)

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

:ref:`MosaicAD example <mosad_array>`

**MosaicAD class Attributes**

These attributes are in addition to the Mosaic class.

- .log DRAGONS logger object.
- .ad  AstroData object.
- .jfactor <list>, Jacobians applied to interpolated pixels.
- .data_list <list>, a list of *<ndarray>* sections.
- .geometry <MosaicGeometry>, geometric parameters for transformation.
- .mosaic_shape <tuple>, output shape of the mosaic array.

.. _mosad_asad:


MosaicAD Methods
----------------

as_astrodata()
**************

This function acts as a wrapper around calls on the *mosaic_image_data* function.
An astrodata object is returned to the caller. WCS information in the headers of
the IMAGE extensions are updated appropriately. *as_astrodata()* will work on all
extensions and extension types. That is, an MEF file with 'SCI', 'VAR' and 'DQ'
image extensions. This will also process any provided mask of name, OBJMASK.

Usage:
 ::

  import astroData
  import gemini_instruments

  # gemini_mosaic_function is a user function ready made for use with GMOS
  # and GSAOI datasets. You may use this function for all Gemini/GMOS or
  # Gemini/GSAOI data.

  from gempy.mosaic.mosaicAD import MosaicAD
  from gempy.mosaic.gemMosaicFunction import gemini_mosaic_function

  # Make an AstroData object for a GMOS dataset.
  ad = astrodata.open('S20030201S0173.fits')

  # Instantiate the MosaicAD object. The 'gemini_mosaic_function' will
  # be executed inside using the 'ad' object to return the MosaicData and
  # MosaicGeometry objects.

  mos_ad = MosaicAD(ad, gemini_mosaic_function)

  # Using the 'mos_ad' object, execute the method as_astrodata(), returning an
  # AstroData object.

  adout = mos_ad.as_astrodata()

**Input parameters**

::

 as_astrodata(block=None, doimg=False, tile=False, return_ROI=True)

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

**Output**

- adout: An Astrodata object with transformed pixel data. Once in the form of
  an Astrodata object, the object can be written to a file,
  E.g.::

    adout.write('newfile.fits')

See Examples for an example of `as_astrodata()`
:ref:`as_astrodata example <asastro_ex>`


.. _tile_asad:

tile_as_astrodata()
*******************

This function evaluates the *tile_all* parameter and then passes parameters
to calls on either *as_astrodata()* or the quasi-private *_tile_blocks()*
function. An astrodata object is returned to the caller.

Usage:
 ::

  import astroData
  import gemini_instruments

  # gemini_mosaic_function is a user function ready made for use with GMOS
  # and GSAOI datasets. You may use this function for all Gemini/GMOS or
  # Gemini/GSAOI data.

  from gempy.mosaic.mosaicAD import MosaicAD
  from gempy.mosaic.gemMosaicFunction import gemini_mosaic_function

  # Make an AstroData object for a GMOS dataset.
  ad = astrodata.open('S20030201S0173.fits')

  # Instantiate the MosaicAD object. The 'gemini_mosaic_function' will
  # be executed inside using the 'ad' object to return the MosaicData and
  # MosaicGeometry objects.

  mos_ad = MosaicAD(ad, gemini_mosaic_function)

  # Using the 'mos_ad' object, execute the method as_astrodata(), returning an
  # AstroData object.

  adout = mos_ad.tile_as_astrodata()

**Input parameters**

::

 tile_as_astrodata(tile_all=False, doimg=False, return_ROI=True)

- tile_all: <bool>. Default is False.
    Data are tiled onto a single output grid wrapping all detectors
    chips (blocks). This results in a dataset with a single extension -- the tiled
    data arrays. When False, data are tiled onto separate blocks, one for each
    detector chip. Default is False.

- doimg: <bool>. Default is False.
    Specifies that *only* the SCI image data are tiled or transformed (see
    parameter, ``tile``). False indicates all image extensions are processed,
    i.e. all SCI, VAR, DQ extensions.

- return_ROI: <bool>. Default is True
    Returns the minimum frame size calculated from the location of the
    amplifiers in a given block. If False uses the blocksize value.

**Output**

- adout: <Astrodata> object with tiled pixel data. Once in the form of
  an Astrodata object, the object can be written to a file,
  E.g.::

    adout.write('newfile.fits')

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

 mos_ad = MosaicAD(ad, gemini_mosaic_function)
 mosaic = mos_ad.mosaic_image_data(block=None, dq_data=False, tile=False,
                                   return_ROI=True)

**Input parameters**

- block: <2-tuple>. Default is None.
    Allows a specific block to be returned as the output mosaic. The tuple
    notation is (col,row) (zero-based) where (0,0) is the lower left block.
    The blocks layout is given by the attribute mosaic_grid.

- dq_data: <bool>
    Handle data in self.data_list as bit-planes.

- tile: <bool>
    If True, the mosaics returned are not corrected for shifting, rotation or
    magnification. Default is False.

- return_ROI: <bool> (default is True).
    Returns the minimum frame size calculated from the location of the
    amplifiers in a given block. If False uses the blocksize value.

**Output**

- mosaic: <ndarray> with mosaic data.

See Examples for an example of `mosaic_image_data()`
:ref:`mosaic_image_data example <asastro_ex>`

.. _mosad_jfactor:

calculate_jfactor()
*******************

Calculate the ratio of reference input pixel size to output pixel size for each
reference extension in the AstroData object.  In practice this ratio is formulated
as the determinant of the WCS transformation matrix.  This ratio is applied to each
pixel to conserve flux in an image after magnification in the transformation.

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

- MosaicAD.jfactor, <list>
    The jfactor attribute is a list providing one Jacobian factor (float) per amp.

.. _mosad_getdl:

get_data_list(attr)
*******************

Returns a list of image data for all the ad. It assumes that the input
AstroData Descriptor *data_section* has been defined for this astrodata type,
i.e. GMOS or GSAOI. The data list returned by this function should be used to
set the instance attribute, *self.data_list*, which is what is worked on by
subsequent calls to *.mosaic_image_data()*.

 Usage
 ::

  sci_data_list = MosaicAD.get_data_list('data')
  var_data_list = MosaicAD.get_data_list('variance')
  dq_data_list  = MosaicAD.get_data_list('mask')

**Input parameters**
::

   get_data_list(attr):

- attr: <str>.
    The <str> indicates the data extensions to be listed. The *as_astrodata()*
    method calls this function for each of 'SCI', 'VAR', 'DQ' and 'OBJMASK'.
    In the context of Astrodata objects, these data will be describe by attribute
    names, 'data', 'variance', 'mask', and 'OBJMASK' (see above, *Usage*.)

**Output**

- data_list. List of *<ndarray>* pixel data.

.. _mosad_info:

info()
******
 The *info()* method returns a dictionary containing critical geometric
 metadata used to create mosaics and tiled arrays. The dictionary provides
 coordinates, amplifier, and block information. The keys for the dictionary
 are::


  'amp_block_coord': <list> of tuples (x1,x2,y1,y2))
       The list of amplifier indices within a block.
  'amp_mosaic_coord': <list> of tuples (x1, x2, y1, y2)).
       The list of amplifier location within the mosaic.
       These values do not include the gaps between the blocks.
  'amps_per_block': <int>
       Number of amplifiers per block.
  'amps_shape_no_trimming': <list>
       Full amp sections without overscan trimming.
  'data_index_per_block': <dict>  indicating which amplifier
       extensions belong to which block (chip).
  'filename': <str>
       The filename of the input dataset.
  'interpolator': <str>
       Interpolator name.
  'reference_block'
       Reference block tuple (col,row).

 Usage
 ::

  dictionary = MosaicAD.info()

**Output**

- <dict>
  Dictionary with the above information.

  We can use pprint to render the info dictionary in a more readable format::

    import pprint
    dictionary = MosaicAD.info()
    pprint.pprint(dictionary)
    {'amp_block_coord': [(0, 512, 0, 4224),
                         (512, 1024, 0, 4224),
			 (1024, 1536, 0, 4224),
			 (1536, 2048, 0, 4224),
			 (0, 512, 0, 4224),
			 (512, 1024, 0, 4224),
			 (1024, 1536, 0, 4224),
			 (1536, 2048, 0, 4224),
			 (0, 512, 0, 4224),
			 (512, 1024, 0, 4224),
			 (1024, 1536, 0, 4224),
			 (1536, 2048, 0, 4224)],
    'amp_mosaic_coord': [(0, 512, 0, 4224),
                         (512, 1024, 0, 4224),
			 (1024, 1536, 0, 4224),
			 (1536, 2048, 0, 4224),
			 (2048, 2560, 0, 4224),
			 (2560, 3072, 0, 4224),
			 (3072, 3584, 0, 4224),
			 (3584, 4096, 0, 4224),
			 (4096, 4608, 0, 4224),
			 (4608, 5120, 0, 4224),
			 (5120, 5632, 0, 4224),
			 (5632, 6144, 0, 4224)],
    'amps_per_block': 4,
    'amps_shape_no_trimming': [(544, 4224),
                               (544, 4224),
                               (544, 4224),
                               (544, 4224),
                               (544, 4224),
                               (544, 4224),
                               (544, 4224),
                               (544, 4224),
                               (544, 4224),
                               (544, 4224),
                               (544, 4224),
                               (544, 4224)],
    'data_index_per_block': {(0, 0): array([0, 1, 2, 3]),
                             (1, 0): array([4, 5, 6, 7]),
                             (2, 0): array([ 8,  9, 10, 11])},
    'filename': 'S20161025S0111.fits',
    'interpolator': 'linear',
    'reference_block': (1, 0)}
