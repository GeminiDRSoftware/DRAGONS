.. _Introduction:

Introduction
------------

.. _what_is:

**What is Mosaic**

Through out this document the word *mosaic* has the following meanings:

- *Mosaic* is the Python software we are describing in this document.

- A *mosaic* is the output ndarray resulting from running the *Mosaic* software.

- *Mosaic* is a Python class name defined in this software.

**What is the Mosaic class**

- The Mosaic class provides functionality to create a mosaic by pasting a set of individual ndarrays of the same size and data type.

- Layout description of the ndarrays on the output mosaic is done via the MosaicData class.
- Information about geometric transformation of the ndarrays is carried using the MosaicGeometry class.

.. _mos_installation:

**Mosaic scripts availability**

Mosaic scripts are in the Gemini_python distribution, hence you need to have the distribution available in your machine to use Mosaic.

For user inside Gemini the software installed in the gemini_python *gempy* directory which need to be imported before running mosaic.

**What is the MosaicAD class**

- MosaicAD is a subclass of Mosaic to provide easy support of Gemini astronomical data by using the AstroData layer class, allowing instrument-agnostic access to Multi Extension FITS files. 

- MosaicAD provides a method to associate IMAGE and BINTABLE FITS extensions. For example a reduced GMOS exposure can contain three IMAGE and three BINTABLE extensions with objects information in the image. The method will merge these 2 sets into one IMAGE extension with the mosaic and one BINTABLE extension.

.. _user_help:

**Getting Help**

If you experience problems running Mosaic please contact the
Gemini `Helpdesk <http://www.gemini.edu/sciops/helpdesk/?q=sciops/helpdesk>`_ (under gemini iRAF/Python)

.. _quick_example:

**Quick Example: Create a mosaic with MosaicAD class.**

- This example assumes you have the gemini-python layer installed in your system.

- Start your favorite Python shell

- Importing required modules
  ::

    from astrodata import AstroData
    # The directory mosaicAD.py and gemMosaicFunction.py modules
    # will probably change when the code goes into production.
    #
    from gempy.adlibrary.mosaicAD import MosaicAD

    #     This is a user function available for your use,
    #     it supports GMOS and GSAOI data
    #
    from gempy.gemini.gemMosaicFunction import gemini_mosaic_function

- Use *AstroData* to open a FITS file
  ::

   ad = AstroData('S20100113S0110vardq.fits')

- Create a *MosaicAD* Class object.
  Notice that the user function *gemini_mosaic_function* supports only GMOS 
  and GSAOI data at this time. The image extension name use to create the mosaic is 'SCI'.
  ::

   mo = MosaicAD(ad,mosaic_ad_function=gemini_mosaic_function,
                 ref_extname='SCI')
   
- Use *mosaic_image_data* method to generate a mosaic with all the 'SCI' extensions in the input Astrodata data list.  The output *mosaic_array* is a numpy array of the same datatype as the input image array in the *ad* object. The input data pieces (blocks) are corrected (transformed) for shift, rotation and magnification with respect to the reference block. This information is available in the 'geometry' configuration file for each supported instrument.
  ::

    mosaic_array = mo.mosaic_image_data()

- Display the resulting mosaic using DS9. Make sure you have DS9 up and running
  and the *numdisplay* python module is available in your Python installation.
  ::

    # numdisplay package is from STScI
    from numdisplay import display

    display(mosaic_array)

- Running mosaicFactory.py in your favorite Unix shell.
  ::

   # Define a unix alias
   alias <path_to_trunk>/trunk/gempy/scripts/mosaicFactory.py mosaicFactory

   # Use it with on or more files:
   mosaicFactory S20120413*.fits

.. _mos_glossary:

.. _primitives:

Mosaic in Primitives
--------------------

The primitive **mosaicADdetectors** in the module *primitives_GEMINI.py* handles GMOS and GSAOI images. The parameter 'tile' default value is False, but it can be change via the 'reduce par' option. 

Example
::
 
 # Using reduce to mosaic a GMOS raw in tile mode.
 # 
 reduce -r mosaicad -p tile=True gS20100113S0110.fits

 # where 'mosaicad' refers to a recipe name in the RECIPES_Gemini directory

Glossary
------------

**Astrodata**
 Python class that serves as an active abstraction for a dataset or a group of datasets

**amplifier**
     In the context of the Mosaic class, amplifier is the ndarray containing the data from any element in the input data list. From the MosaicAD class is the amount of data from one FITS IMAGE extension limited by the image section from the header keyword DATASEC.

**array**
  An array describes the individual component that detect photons within an instrument; eg, a CCD or an infrared array.

.. _block_def:

**block**
    Is an ndarray containing one or more amplifier data.

**mask**
   Ndarray of the same shape (ny,nx); i.e. number of pixels in y and x, as the output mosaic but with zero as the pixel value for image data and 1 as non-image data in the output mosaic. Example of non-image data are the gaps between the blocks and the areas of no data resulting from transformation.

**MosaicData**
    Python class with functions to verify input data lists. The object created with this class is required as input to create a Mosaic object. For more details see :ref:`MosaicData example <help_mdata>`

**MosaicGeometry**
    Python class with functions to verify the input data ndarrays geometry properties values and the geometry of the output mosaic. Some of these values are rotation, shifting and magnification, and are used to transform the blocks to match the reference block geometry. For more details see :ref:`MosaicGeometry example <help_mgeo_example>`.  

**Mosaic**
    Python base class with low level functionality to generate a mosaic from MosaicData and MosaicGeometry object inputs. Depending on the amount of input geometry values supplied when creating the MosaicGeometry, the user can generate a mosaic with or without transforming blocks. This class object also contains a mask as an attribute.

**MosaicAD**
   Python derived class of Mosaic. Together with the Astrodata input object, this class offers functionality to output an Astrodata object containing one or more mosaics and/or merged catalogs in binary tables which are :ref:`associated <mos_associated>` with the mosaics.

.. _why_ndarray:

**ndarray**
    Is a Numpy (python package for numerical computation) array of values. The term is used in here to make a difference with the CCD array.

**reference block**
  Is a 1-based tuple (column_number, row_number) with respect to the lower left origin (1,1), it notes the reference block to which the transformation values are given. These values are given in the geometry dictionary with key *transformation*.

.. _mos_transf:

**transformation**
    The act of applying interpolation to a block to correct for rotation, shifting and magnification with respect to the reference block.

