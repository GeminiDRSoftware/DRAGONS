.. coding:

.. _Writing_Adding:

**********************************
Writing and Adding New Descriptors
**********************************

Overview of the Descriptor Code
-------------------------------

The infrastructure for the descriptor code is located within the ``astrodata``
subpackage in the ``gemini_python`` package. In addition, a default set of
descriptors that access FITS standard keywords is provided by ``astrodata``.
This code is generic and is not Gemini specific.

The Gemini descriptor code is contained within the ``astrodata_Gemini`` addon
in the ``gemini_python`` package. The Gemini descriptors inherit the FITS
descriptors, since the Gemini telescopes produce files in FITS format. Other
telescope / instrument addons can exist, enabling the addition of descriptors
from non-Gemini instruments. 

The following sections will refer specifically to adding descriptors to the
already existing Gemini descriptor code contained within the
``astrodata_Gemini`` addon.

Overview of the FITS Descriptor Code
------------------------------------

The FITS descriptors provide access to those keywords that are part of the FITS
standard. There are currently 53 keywords defined in the FITS standard
(http://heasarc.gsfc.nasa.gov/docs/fcg/standard_dict.html). There are four FITS
descriptors available that return the value of the corresponding FITS standard
keywords in the PHU of the AstroData object: 

  - ``instrument [INSTRUME]``
  - ``object [OBJECT]``
  - ``telescope [TELESCOP]``
  - ``ut_date [DATE-OBS]``

The FITS descriptor code is currently located in the ``gemini_python`` package
in the ``astrodata_FITS/ADCONFIG_FITS/descriptors`` directory [NOTE: the
``astrodata_FITS`` directory will be moved to the ``astrodata`` directory in
the near future]:

  - ``calculatorIndex.FITS.py``
  - ``CalculatorInterface_FITS.py``
  - ``DescriptorsList_FITS.py``
  - ``docstrings.py``
  - ``FITS_Descriptors.py``
  - ``FITS_Keywords.py``

The function of each of these files is the same as the corresponding files in
the Gemini descriptor code and will be described in the following sections.

Overview of the Gemini Descriptor Code
--------------------------------------

The Gemini descriptors provide access to keywords available in the headers of
Gemini data.

The Gemini descriptor code is located in the ``gemini_python`` package in the
``astrodata_Gemini/ADCONFIG_Gemini/descriptors`` directory. When writing and
adding new Gemini descriptors, a developer will require knowledge of the
following files:

  - ``calculatorIndex.GEMINI.py``
  - ``CalculatorInterface_GEMINI.py``
  - ``GEMINI_Descriptors.py``
  - ``GEMINI_Keywords.py``
  - ``<INSTRUMENT>/<INSTRUMENT>_Descriptors.py``
  - ``<INSTRUMENT>/<INSTRUMENT>_Keywords.py``

The following files are required to create the
``CalculatorInterface_GEMINI.py`` file:

  - ``astrodata/scripts/mkCalculatorInterface``
  - ``DescriptorsList_GEMINI.py``
  - ``docstrings.py``

Introduction to the Gemini Descriptor Code
------------------------------------------

When a descriptor is called (as described in the :ref:`Basic Descriptor Usage
<Basic_Descriptor_Usage>` section and the :ref:`Advanced Descriptor Usage
<Advanced_Descriptor_Usage>` section), the CalculatorInterface class (``CI``)
is accessed, which is either contained in the module
``CalculatorInterface_GEMINI.py`` or, if this module doesn't exist, is
automatically generated and stored in memory. The ``CI`` contains a function
for each available descriptor (see :ref:`Appendix B <Appendix_CI>` for an
example function). These functions attempt to determine a value for the 
descriptor by performing the following steps:

  - First, depending on the AstroData Type of the AstroData object, a single
    set of keywords and a single set of descriptor functions are determined via
    inheritance from the keyword files (``<INSTRUMENT>_Keywords.py``,
    ``GEMINI_Keywords.py`` and ``FITS_Keywords.py``) and the descriptor files 
    (``<INSTRUMENT>_Descriptor.py``, ``GEMINI_Descriptor.py`` and
    ``FITS_Descriptor.py``), respectively, with the files at the start of
    each list taking precedence over files later in the list.

  - If a descriptor function exists in the single set of descriptor functions
    determined in the first step for the descriptor being called, then that
    descriptor function is used to determine the value of the descriptor.

  - If no descriptor function is found, the value of a keyword in the PHU of
    the AstroData object is returned as the value of the descriptor, where the
    keyword is the one directly associated to the descriptor and is contained
    in the single set of keywords determined in the first step.

  - If a value for the descriptor is returned, either from a descriptor
    function or directly from the PHU of the AstroData object, a
    DescriptorValue (``DV``) object is instantiated (which contains the value
    of the descriptor) and returns this to the user.

  - If a value for the descriptor is not found, an exception is raised (see the
    :ref:`Descriptor Exceptions <Descriptor_Exceptions>` section).

A descriptor function is necessary in the cases where the descriptor does more
than simply access a single keyword from the PHU of the AstroData object, e.g.,
if a descriptor requires access to multiple keywords, requires access to
keywords in the pixel data extensions (a dictionary must be returned by the
descriptor function, where the key is the ("``EXTNAME``", ``EXTVER``) tuple)
and / or requires some validation.

Description of the files used in the Gemini Descriptor Code
-----------------------------------------------------------

.. _calculatorIndex.GEMINI.py:

``calculatorIndex.GEMINI.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``calculatorIndex.GEMINI.py`` file contains a Python dictionary named
``calculatorIndex`` and is used to define which Python object (i.e., the
descriptor class that defines the descriptor functions, in the form
``<module_name>.<calculator_class_name>``) to use as the calculator for a given
``<INSTRUMENT>``::

  calculatorIndex = {
    "<INSTRUMENT>":"<INSTRUMENT>_Descriptors.<INSTRUMENT>_DescriptorCalc()",
    }

When adding descriptors for a new, undefined instrument, an appropriate entry
must be added to the ``calculatorIndex`` dictionary in the
``calculatorIndex.GEMINI.py`` file as shown above.

``CalculatorInterface_GEMINI.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``CalculatorInterface_GEMINI.py`` module contains the CalculatorInterface
class (``CI``), which contains a function for each available descriptor (see
:ref:`Appendix B <Appendix_CI>` for an example function). The
``CalculatorInterface_GEMINI.py`` module is automatically generated by the
``mkCalculatorInterface`` script, which is located in ``astrodata/scripts``
directory. Therefore, the ``CalculatorInterface_GEMINI.py`` module should never
be edited directly. If the ``CalculatorInterface_GEMINI.py`` module does not
exist, the ``CI`` is automatically generated and stored in memory.

``GEMINI_Keywords.py``
~~~~~~~~~~~~~~~~~~~~~~

The Gemini specific keyword file ``GEMINI_Keywords.py`` contains a Python
dictionary named ``GEMINI_KeyDict``, where the key is a variable in the form
``key_<descriptor>`` and the value is the keyword directly associated to the
descriptor ``<descriptor>``::

  GEMINI_KeyDict = {
    "key_airmass":"AIRMASS",
    ...
    "key_camera":"CAMERA",
    ...
    }

As shown above, the ``AIRMASS`` keyword is associated to the ``airmass``
descriptor via the variable ``key_airmass``. When the ``airmass`` descriptor is
called, the value of the single keyword ``AIRMASS`` in the PHU of the AstroData
object is returned.

If a descriptor function in the Gemini specific descriptor file
``GEMINI_Descriptors.py`` requires access to additional keywords, appropriate
variables must be defined in the ``GEMINI_KeyDict`` dictionary, so that keyword
names are not hard-coded in the descriptor files, allowing a single variable
to be used consistently by multiple descriptor functions::

  GEMINI_KeyDict = {
    ...
    "key_pwfs1":"PWFS1_ST",
    ...
    }

``GEMINI_Descriptors.py``
~~~~~~~~~~~~~~~~~~~~~~~~~

The Gemini specific descriptor file ``GEMINI_Descriptors.py`` contains
descriptor functions that apply to all Gemini data.

``<INSTRUMENT>_Keywords.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The instrument specific keyword files ``<INSTRUMENT>_Keywords.py``, which are
located in the corresponding ``<INSTRUMENT>`` directory in the
``astrodata_Gemini/ADCONFIG_Gemini/descriptors`` directory, contain a Python
dictionary named ``<INSTRUMENT>_KeyDict``, where the key is a variable in the
form ``key_<descriptor>`` and the value is the keyword directly associated to
the descriptor <descriptor>, and is used to overwrite (via inheritance) any of
the Gemini specific keywords directly associated with the descriptor as defined
in ``GEMINI_Keywords.py``::

  GMOS_KeyDict = {
    ...
    "key_camera":"INSTRUME",
    ...
    }

For GMOS data, the value of the single keyword ``INSTRUME`` in the PHU of the
AstroData object is returned when the ``camera`` descriptor is called, instead
of the value of the single keyword ``CAMERA``, as defined in the Gemini
specific keyword file ``GEMINI_Keywords.py``.

If a descriptor function in the instrument specific descriptor files
``<INSTRUMENT>_Descriptors.py`` require access to additional keywords,
appropriate variables must be defined in the ``<INSTRUMENT>_KeyDict``
dictionary, so that keyword names are not hard-coded in the descriptor files,
allowing a single variable to be used consistently by multiple descriptor
functions::

  GMOS_KeyDict = {
    ...
    "key_ccdsum":"CCDSUM",
    ...
    }

``<INSTRUMENT>_Descriptors.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The instrument specific descriptor files ``<INSTRUMENT>_Descriptors.py``, which
are located in the corresponding ``<INSTRUMENT>`` directory in the
``astrodata_Gemini/ADCONFIG_Gemini/descriptors`` directory, contain descriptor
functions that are specific to ``<INSTRUMENT>`` and are used to overwrite (via
inheritance) any of the Gemini specific descriptor functions as defined in
``GEMINI_Descriptors.py``. An example descriptor function (``detector_x_bin``)
from ``GMOS_Descriptors.py`` can be found in :ref:`Appendix C
<Appendix_descriptor>`.

``mkCalculatorInterface``
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``mkCalculatorInterface`` script is located in the ``astrodata/scripts``
directory and contains the code required to automatically generate the
``CalculatorInterface_GEMINI.py`` module. To create
``CalculatorInterface_GEMINI.py``, run the following command in the
``astrodata_Gemini/ADCONFIG_Gemini/descriptors`` directory::

  shell> mkCalculatorInterface > CalculatorInterface_GEMINI.py

The ``mkCalculatorInterface`` script uses the information in the
``DescriptorsList_GEMINI.py`` file and the ``docstrings.py`` file to create the
``CalculatorInterface_GEMINI.py`` module. This script should be run after
making changes to the ``DescriptorsList_GEMINI.py`` file or the
``docstrings.py`` file.

``DescriptorsList_GEMINI.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``DescriptorsList_GEMINI.py`` file contains a list of Gemini descriptors
and their default Python type::

  [
    DD("airmass", pytype=float),
    ...
  ]

The ``DescriptorsList_GEMINI.py`` file is used by the ``mkCalculatorInterface``
script to generate a function for every descriptor in the
``CalculatorInterface_Gemini.py`` file.

``docstrings.py``
~~~~~~~~~~~~~~~~~

The ``docstrings.py`` file contains a function for every Gemini descriptor (as
listed in the ``DescriptorsList_GEMINI.py`` file) where the docstring for each
descriptor can be defined. The ``docstrings.py`` file is used by the
``mkCalculatorInterface`` script to include the docstrings for the descriptor
functions in the ``CalculatorInterface_Gemini.py`` file.

How to add a new Gemini descriptor
----------------------------------

The following instructions describe how to add a new descriptor to the system.

  1. First, check to see whether the new descriptor has the same concept as a
     descriptor that already exists (:ref:`Appendix A <Appendix_typewalk>`). If
     a new descriptor is required, edit the ``DescriptorsList_GEMINI.py`` file
     and add the new descriptor to the list in alphabetical order. Ensure that
     the default Python type for the descriptor is defined::
       
       [
         ...
         DD("<my_descriptor_name>", pytype=str),
         ...
       ]
  
  2. Add a function with the same name as the new descriptor to the
     ``docstrings.py`` file and write a docstring for the new descriptor so
     that it can be included in the ``CalculatorInterface_GEMINI.py`` file.

  3. Regenerate the ``CalculatorInterface_GEMINI.py`` file::
       
       shell> mkCalculatorInterface > CalculatorInterface_GEMINI.py

  4. If the new descriptor is for a new, undefined ``<INSTRUMENT>``, create an
     ``<INSTRUMENT>`` directory containing an ``<INSTRUMENT>_Descriptors.py``
     file::

       from GEMINI_Descriptors import GEMINI_DescriptorCalc
       from <INSTRUMENT>_Keywords import <INSTRUMENT>_KeyDict

       class <INSTRUMENT>_DescriptorCalc(GEMINI_DescriptorCalc):
           # Updating the global key dictionary with the local key dictionary
           # associated with this descriptor class
           _update_stdkey_dict = <INSTRUMENT>_KeyDict

     and an ``<INSTRUMENT>_Keywords.py`` file::

       <INSTRUMENT>_KeyDict = {}

     In addition, add an appropriate entry to the ``calculatorIndex.Gemini.py``
     file. 

  5. If the new descriptor simply returns the value of a single keyword in the
     PHU of the AstroData object, check whether the correct keyword is already
     defined in the ``GEMINI_KeyDict`` dictionary in the keyword file
     ``GEMINI_Keywords.py``. If not, add an entry to the
     ``<INSTRUMENT>_KeyDict`` dictionary in the instrument specific keyword
     file ``<INSTRUMENT>_Keywords.py``, specifying the new descriptor and the
     associated keyword::

       <INSTRUMENT>_KeyDict = {
         "key_<my_descriptor_name>":"MYKEYWRD",
       }

     The descriptor can now be tested; go to step 7.

  6. If the new descriptor requires access to multiple keywords, requires
     access to keywords in the pixel data extensions and / or requires some
     validation, a descriptor function must be created. Depending on the type
     of information the new descriptor will provide, edit one of the following
     files to include the new descriptor function: 

       - ``GEMINI_Descriptor.py``
       - ``<INSTRUMENT>_Descriptor.py``

     If access to a particular keyword is required, first check the keyword
     files (``FITS_Keywords.py``, ``GEMINI_Keywords.py`` and
     ``<INSTRUMENT>_Keywords.py``) to see if it has already been defined. If
     required, the ``<INSTRUMENT>_Keywords.py`` file should be edited to
     contain any new keywords required for this new descriptor function. 

  7. Test the descriptor::
       
       >>> from astrodata import AstroData
       >>> ad = AstroData("N20091027S0137.fits")
       >>> print ad.<my_descriptor_name>()

Descriptor Coding Guidelines
----------------------------

When creating descriptor functions, the guidelines below should be followed:

  1. Return value

     - The descriptors will return the correct value, regardless of the data
       processing status of the AstroData object.
     - The descriptors will not write keywords to the headers of the AstroData
       object or cache any information, since it is no effort to use the
       descriptors to obtain the correct value as and when it is required.
     - The value of a descriptor can be written to the history, for information
       only.

  2. Return value Python type

     - The descriptors will always return a ``DV`` object to the user.
     - The ``DV`` object is instantiated by the ``CI`` for descriptors that
       obtain their values directly from the headers of the AstroData object.
       For descriptors that obtain their values from the descriptor functions,
       the descriptor functions should be coded to return a ``DV`` object. The
       ``DV`` object contains information related to the descriptor, including
       the value of the descriptor, the default Python type for that descriptor
       and the units of the descriptor. 

  3. Keyword access

     - The ``phu_get_key_value`` and ``get_key_value`` AstroData member 
       functions should be used in the descriptor functions to access keywords
       in the PHU and the headers of the pixel data extensions, respectively,
       of an AstroData object. 

  4. Logging

     - Descriptors will not log any messages.

  5. Raising exceptions

     - If the value of a descriptor can not be determined for whatever reason,
       the descriptor function should raise an exception.
     - The descriptor functions should never be coded to return None. Instead,
       a descriptor function should throw an exception with a message
       explaining why a value could not be returned (e.g., if the concept does
       not directly apply to the data). An exception thrown from a descriptor
       function will be caught by the ``CI``.

  6. Exception rule

     - Descriptors should throw exceptions on fatal errors.
     - Exceptions thrown on fatal errors (e.g., if a descriptor function is not
       found in a loaded calculator) should never be caught by the ``CI``. The
       high level code, such as a script or a primitive, should catch any
       relevant exceptions. 

  7. Descriptor names

     - Descriptor names will be:

       - all lower case
       - terms separated with "_"
       - not instrument specific
       - not mode specific, mostly

     - A descriptor should describe a particular concept and apply for all
       instrument modes.

  8. Standard parameters

     - Descriptors accept parameters, some with general purposes are 
       standardized.
     - It is especially important for descriptor parameters to follow the
       Standard Parameter Names
       (http://gdpsg.wikis-internal.gemini.edu/index.php/GDPSG-NamingConventions#Standard_Parameter_Names) as they are front-facing to the user and should 
       therefore be consistent.

.. _Descriptor_Exceptions:

Descriptor Exceptions
---------------------

When writing descriptor functions, if a descriptor is unable to return a value,
an exception should be raised in the code with an appropriate, explicit error
message, so that it is clear to the user exactly why a value could not be
returned. The exception information is stored in ``exception_info`` by the
``CI`` so that a user can access that information, if they wish to do so. A
list of descriptor exceptions can be found in the ``astrodata/Errors.py``
module. These descriptor exceptions inherit from the DescriptorError base class
and are caught by the ``CI``. If the default setting of ``throwExceptions =
False`` (line 62 in ``astrodata/Calculator.py``), ``None`` is returned. During
development, the developer should set ``throwExceptions = True`` so that
exceptions are thrown.
