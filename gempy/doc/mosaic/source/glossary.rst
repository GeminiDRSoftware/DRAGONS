.. _mos_glossary:

Glossary
--------

**Astrodata**
  Is DRAGONS data abstraction for astronomical datasets. Currently, only FITS files
  can opened and constructed into an astrodata object, but is extensible to other
  data formats, such as HDF5.

**amplifier**
  In the context of the Mosaic class, amplifier is the ndarray containing the
  data from any element in the input data list. From the MosaicAD class is the
  amount of data from one FITS IMAGE extension limited by the image section
  from the header keyword DATASEC.

**array**
  An array describes the individual component that detect photons within an
  instrument; eg, a CCD or an infrared array.

.. _block_def:

**block**
  Is an ndarray containing one or more amplifier data arrays and corresponds
  to one (1) detector chip. I.e., a block is the abstract representation of
  a detector chip. For GMOS, three CCD chips are represented by three corresponding
  blocks. For GSAOI, four blocks represent the four chips.

**mask**
  Ndarray of the same shape (ny, nx); i.e. number of pixels in y and x, as the
  output mosaic with zero as the pixel value for image data and 16 as
  non-image data ("no data") in the output mosaic. Example of non-image data
  are the gaps between the blocks and the areas of no data resulting from
  transformation.

**MosaicData**
  A class providing functions to verify input data lists. The object created
  with this class is required as input to create a Mosaic object. For more
  details see :ref:`MosaicData example <help_mdata>`.

**MosaicGeometry**
  A class providing functions to verify the input data ndarrays geometry
  properties values and the geometry of the output mosaic. Some of these
  values are rotation, shifting and magnification, and are used to transform
  the blocks to match the reference block geometry. For more details see
  :ref:`MosaicGeometry example <help_mgeo_example>`.

**Mosaic**
  The base class with low level functionality to generate a mosaic from
  MosaicData and MosaicGeometry object inputs. Depending on the amount of
  input geometry values supplied when creating the MosaicGeometry, the user
  can generate a mosaic with or without transforming blocks. This class object
  also contains a mask as an attribute.

**MosaicAD**
  Python derived class of Mosaic. Together with the Astrodata input object,
  this class offers functionality to output an Astrodata object containing
  one or more mosaics and/or merged catalogs in binary tables which are
  :ref:`associated <mos_associated>` with the mosaics.

.. _why_ndarray:

**ndarray**
  An "N-dimensional array" is a `numpy` array of values. The term is used here
  to differentiate it from a CCD or other detector array.

**reference block**
  Is a tuple (column_number, row_number) with respect to the lower
  left origin (0, 0). It notes the reference block to which the transformation
  values are given. These values are given in the geometry dictionary with key
  *transformation*.

.. _mos_transf:

**transformation**
  The act of applying interpolation to a block to correct for rotation, shifting
  and magnification with respect to the reference block.
