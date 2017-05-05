.. include supptools

.. _Introduction:

Introduction
============

.. _what_is:

What is Mosaic
--------------

**Mosaic** is a pure Python implementation of the gemini_iraf task, `gmosaic`.
Of course, this implementation provides programmatic access to **mosaic**  
interfaces for other packages to import and use the mosaic package. The 
package resides (currently) in the `gemini_python` gempy package.

Throughout this document the term *mosaic* will have the following meanings:

- *mosaic* is the Python package described in this document.

- A *mosaic* is the output ndarray resulting from running the *Mosaic* software.

- *Mosaic* is a Python class name defined in this software. This class is
  provided to serve as the base class for subclasses implementing Mosaic under
  different contexts. In the case of the `mosaic` package, **MosaicAD** is 
  subclassed on **Mosaic** and supports working with AstroData objects.

**What is the Mosaic class**

- The Mosaic class provides functionality to create a mosaic by pasting a set of 
  individual ndarrays of the same size and data type.

- Layout description of the ndarrays on the output mosaic is done via the 
  MosaicData class.

- Information about geometric transformation of the ndarrays is carried using 
  the MosaicGeometry class.

.. _mos_installation:

Mosaic scripts availability
---------------------------

Mosaic scripts are in the `gemini_python` distribution, hence you need to have 
the distribution available on your machine to use Mosaic.

For users inside the Gemini firewall the software installed in the gemini_python
*gempy* directory which need to be imported before running mosaic.

What is the MosaicAD class
--------------------------

- MosaicAD is a subclass of Mosaic to provide easy support of Gemini astronomical
  data by using the AstroData layer class, allowing instrument-agnostic access to 
  Multi Extension FITS files.

- MosaicAD extends the generic Mosaic class to supoort AstroData objects. Both
  MosaicAD and Mosaic provide support for tiling and transformation of multiple 
  image arrays onto a single image plane.

.. _user_help:

Getting Help
------------

If you experience problems running Mosaic please contact the
Gemini `Helpdesk <http://www.gemini.edu/sciops/helpdesk/?q=sciops/helpdesk>`_ 
(under gemini IRAF/Python).

Quick Example
-------------

Create a mosaic with MosaicAD class.

- The `gemini-python` package is installed on your system.

- Start your favorite Python shell

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

    mo = MosaicAD(ad,mosaic_ad_function=gemini_mosaic_function)
   
- Use *mosaic_image_data* method to generate a mosaic with all the 'SCI' 
  extensions in the input Astrodata data list.  The output *mosaic_array* is a 
  numpy array of the same datatype as the input image array in the *ad* object. 
  The input data pieces (blocks) are corrected (transformed) for shift, rotation 
  and magnification with respect to the reference block. This information is 
  available in the 'geometry' configuration file for each supported instrument. ::

    mosaic_array = mo.mosaic_image_data()

- Display the resulting mosaic using DS9. Make sure you have DS9 up and running
  and the *numdisplay* python module is available in your Python installation. ::

   # numdisplay package is from STScI
   from numdisplay import display

   display(mosaic_array)


The `mosaic` package also provides a command line tool called ``automosaic.`` 
This script is a convenience tool that allows users to pass FITS data to 
Mosaic/MosaicAD directly from the command line. It provides access to a subset 
of `mosaic` options. 

E.g.,

Running automosaic in your favorite Unix shell. ::

   # Use it with one or more files:
   automosaic S20170427*.fits

`automosaic` is detailed  in :ref:`Supplemental Tools <auto_mos>`.

.. _primitives:

Mosaic in Primitives
--------------------

.. todo:: (Update required for current or future primitive names.)

The primitive **mosaicADdetectors** in the module *primitives_GEMINI.py* handles 
GMOS and GSAOI images. The parameter 'tile' default value is False, but it can be 
change via the 'reduce par' option. 

Example ::
 
  # Using reduce to mosaic a GMOS raw in tile mode.

  reduce -r mosaicad -p tile=True S20170427S0064.fits

  # where 'mosaicad' refers to a recipe name in the RECIPES_Gemini directory
