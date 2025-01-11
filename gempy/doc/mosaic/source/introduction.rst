.. include supptools

.. _Introduction:

Introduction
============

.. _what_is:

What is Mosaic
--------------

**Mosaic** is a pure Python implementation to "mosaic" multi-chip instrument
detectors. The mosaic package is used to create both *mosaic* images, which are
produced by affine transformations based on the geometry of detector chips, and
*tiled* images, where data arrays have been simply concatenated and layed onto
the output grid.

The implementation provides programmatic access to **mosaic** interfaces for
other packages to import and use the mosaic package. The package resides in the
DRAGONS gempy package.

Throughout this document the term *mosaic* will have the following meanings:

- *mosaic* is the Python package described in this document.

- A *mosaic* is the output ndarray resulting from running the *Mosaic* software.

- *Mosaic* is a Python class name defined in this software. The *Mosaic* class
  is the base class for subclasses implementing Mosaic under different contexts.
  In the case of the `mosaic` package, **MosaicAD** is subclassed on **Mosaic**
  and supports working with AstroData objects.

**What is the Mosaic class**

- The Mosaic class provides functionality to create a mosaic by pasting a set of
  individual ndarrays of the same size and data type.

- Layout description of the ndarrays on the output mosaic is done via the
  MosaicData class.

- Information about geometric transformation of the ndarrays is carried using
  the MosaicGeometry class.

What is the MosaicAD class
--------------------------

- MosaicAD extends the generic Mosaic class to supoort AstroData objects. Both
  MosaicAD and Mosaic provide support for tiling and transformation of multiple
  image arrays onto a single image plane.

- MosaicAD provides support of Gemini astronomical data by working with
  AstroData objects, allowing instrument-agnostic access to Multi-Extension
  FITS (MEF) files.

.. _user_help:

Getting Help
------------

If you experience problems running Mosaic please contact the
Gemini `Helpdesk <http://www.gemini.edu/sciops/helpdesk/?q=sciops/helpdesk>`_
(under gemini IRAF/Python).

Quick Example
-------------

Create a mosaic with MosaicAD class.

- DRAGONS is installed on your system.

- Import required modules ::

   import astrodata
   import gemini_instruments
   from gempy.mosaic.mosaicAD import MosaicAD

   # This is a user function available and supports GMOS and GSAOI data
   from gempy.mosaic.gemMosaicFunction import gemini_mosaic_function

- Use *astrodata.open()* to open a FITS file ::

    ad = astrodata.open('S20170427S0064.fits')

- Create a *MosaicAD* Class object.
  The user function *gemini_mosaic_function* currently supports only GMOS and
  GSAOI data at this time. ::

    mos = MosaicAD(ad, mosaic_ad_function=gemini_mosaic_function)

- Use *mosaic_image_data* method to generate a mosaic with all the 'SCI'
  extensions in the input Astrodata data list.  The output *mosaic_array* is a
  numpy *<ndarray>* of the same datatype as the input image array in the *ad*
  object.
  The input data pieces (blocks) are corrected (transformed) for shift, rotation
  and magnification with respect to the reference block. This information is
  available in the 'geometry' configuration file for each supported instrument. ::

    mosaic_array = mos.mosaic_image_data()

- Display the resulting mosaic using DS9. Make sure you have DS9 up and
  running::

   from gempy.numdisplay import display

   display(mosaic_array)

.. _primitives:

Mosaic in Primitives
--------------------

The primitive **mosaicDetectors** in the module *primitives_visualize.py* handles
GMOS and GSAOI images and can be called directly from the reduce command line.

Example ::

  # Using reduce to mosaic a GMOS raw in tile mode.

  reduce -r mosaicDetectors S20170427S0064.fits

  # where mosaciDetectors is the primitive name, called here directly.

Tiling can be done from the *reduce* command line with the *tileArrays*
primitive::

    # Using reduce to tile a GMOS raw in tile mode.

    reduce -r tileArrays S20170427S0064.fits

    # where tileArrays is the primitive name, called here directly.
