*********************
What are Descriptors?
*********************

Descriptors are designed such that essential keyword values that describe a
particular concept can be accessed from the headers of a given dataset in a
consistent manner, regardless of which instrument was used to obtain the
data. This is particularly useful for Gemini data, since the majority of
keywords used to describe a particular concept at Gemini are not uniform
between the instruments.

.. _Basic_Descriptor_Usage:

**********************
Basic Descriptor Usage
**********************

The command ``typewalk -l`` lists all available descriptors. As of August 3, 
2011, there are 61 descriptors available (:ref:`Appendix A
<Appendix_typewalk>`). 

The following commands show an example of how to use descriptors and can be 
entered at an interactive Python prompt (e.g., ``ipython``, ``pyraf``)::

  >>> from astrodata import AstroData
  # Load the fits file into AstroData
  >>> ad = AstroData("N20091027S0137.fits")
  # Count the number of science extensions in the AstroData object
  >>> ad.count_exts(extname="SCI")
  3
  # Get the airmass value using the airmass descriptor
  >>> airmass = ad.airmass()
  >>> print airmass
  1.327
  # Get the instrument name using the instrument descriptor
  >>> print "My instrument is %s" % ad.instrument()
  My instrument is GMOS-N
  # Get the gain value for each science extension
  >>> for ext in ad["SCI"]:
  ...     print ext.gain()
  ... 
  2.1
  2.337
  2.3
  >>> print ad.gain()
  {('SCI', 2): 2.3370000000000002, ('SCI', 1): 2.1000000000000001, 
  ('SCI', 3): 2.2999999999999998}

In the examples above, the airmass and instrument apply to the dataset as a
whole (the keywords themselves only exist in the PHU) and so only one value is 
returned. However, the gain applies to the pixel data extensions and so for
this AstroData object, three values are returned, since there are three pixel
data extensions. In this case, a Python dictionary is used to store the values,
where the key of the dictionary is the ("``EXTNAME``", ``EXTVER``) tuple.

.. _Advanced_Descriptor_Usage:

*************************
Advanced Descriptor Usage
*************************

DescriptorValue
---------------

When a descriptor is called, what is actually returned is a ``DescriptorValue``
(``DV``) object::

  >>> print type(airmass)
  <type 'instance'>

Each descriptor has a default Python type defined::

  >>> print ad.airmass().pytype
  <type 'float'>

If any of the following operations are applied to the ``DV`` object, the ``DV``
object is automatically cast to the default Python type for that descriptor::

  +   -   *   /   //   %   **   <<   >>   ^   <   <=   >   >=   ==

For example::

  >>> print type(airmass*1.0)
  <type 'float'>

When using operations with a ``DV`` object and a numpy object, care must be
taken. Consider the following cases::

  >>> ad[0].data / ad.gain()
  >>> ad[0].data / ad.gain().as_pytype()
  >>> ad[0].data / 12.345

All the above commands return the same result (assuming that ad.gain() =
12.345). However, the first command is extremely slow but the second and third
commands are fast. In the first case, since both operands have overloaded
operators, the operator from the operand on the left will be used. For some
reason, the ``__div__`` operator from the numpy object loops over each pixel in
the numpy object and uses the ``DV`` object as an argument, which is very time
consuming. Therefore, the ``DV`` object should be cast to an appropriate
Python type before using it in an operation with a numpy object. 

The descriptor value can be retrieved with the default Python type by using
``as_pytype()``::

  >>> elevation = ad.elevation().as_pytype()
  >>> print elevation
  48.6889222222
  >>> print type(elevation)
  <type 'float'>

The ``as_pytype()`` member function of the ``DV`` object should only be 
required when the ``DV`` object can not be automatically cast to it's default 
Python type. For example, when it is necessary to use an actual descriptor
value (e.g., string, float, etc.) rather than a ``DV`` object as a key to a
Python dictionary, the ``DV`` object can be cast to it's default Python type
using ``as_pytype()``::

  >>> my_key = ad.gain_setting.as_pytype()
  >>> my_value = my_dict[my_key]

If an alternative Python type is required for whatever reason, the ``DV`` 
object can be cast to the appropriate Python type as follows::

  >>> elevation_as_int = int(ad.elevation())
  >>> print elevation_as_int
  48
  >>> print type(elevation_as_int)
  <type 'int'>

In the case where a descriptor returns multiple values (one for each pixel data
extension), a Python dictionary is used to store the values, where the key of
the dictionary is the ("``EXTNAME``", ``EXTVER``) tuple::

  >>> print ad.gain()
  {('SCI', 2): 2.3370000000000002, ('SCI', 1): 2.1000000000000001, 
  ('SCI', 3): 2.2999999999999998}

A descriptor that is related to the pixel data extensions but has the same
value for each pixel data extension will "act" as a single value that has a
Python type defined by the default Python type of that descriptor::

  >>> xbin = ad.detector_x_bin()
  >>> print xbin
  2
  >>> print type(xbin)
  <type 'instance'>
  >>> print xbin.pytype
  <type 'int'>

If the original value of the descriptor is required, it can be retrieved by
using ``get_value()``::

  >>> xbin = ad.detector_x_bin().get_value()
  >>> print xbin
  {('SCI', 2): 2, ('SCI', 1): 2, ('SCI', 3): 2}
  >>> print type(xbin)
  <type 'dict'>
  >>> print xbin[("SCI", 1)]
  2

DescriptorUnits
---------------

The DescriptorUnits (``DU``) object provides a way to access and update the
units for a given descriptor value. Basic implementation of this feature has
been completed and development is ongoing.

**********************************
Writing and Adding New Descriptors
**********************************

Introduction to the Gemini Descriptor Code
------------------------------------------

The Gemini descriptor code is located in the gemini_python package in the
``astrodata_Gemini/ADCONFIG_Gemini/descriptors`` directory. When writing and
adding new Gemini descriptors, a user / developer will require knowledge of the
following files: 

  - ``CalculatorInterface.py``
  - ``mkCalculatorInterface.py``
  - ``StandardDescriptorKeyDict.py``
  - ``StandardGenericKeyDict.py``
  - ``StandardGEMINIKeyDict.py``
  - ``Standard<INSTRUMENT>KeyDict.py``
  - ``descriptorDescriptionDict.py``
  - ``calculatorIndex.Gemini.py``
  - ``Generic_Descriptor.py``
  - ``GEMINI_Descriptor.py``
  - ``<INSTRUMENT>_Descriptor.py``

Overview of the Gemini Descriptor Code
--------------------------------------

.. figure:: descriptor_files.jpg

   An overview showing the relationship between the descriptor files. The files
   located in the dashed box on the left are the descriptor files that
   contain descriptor functions. The files located in the dotted box on the
   right are the keyword files that contain dictionaries describing the
   one-to-one relationship between a descriptor / variable that is used in the 
   associated descriptor files on the left (as shown by the double headed
   arrows) and the keyword associated with that descriptor / variable. In
   addition, the files within both the dashed box and the dotted box are
   subject to inheritance, i.e., any information contained within the files
   located lower in the chart will overwrite equivalent information that is
   contained within files located higher in the chart. The result of this
   inheritance is a single set of descriptor functions and keywords that will
   be used for a given AstroData object.

When a descriptor is called (as described in the :ref:`Basic Descriptor Usage
<Basic_Descriptor_Usage>` section and the :ref:`Advanced Descriptor Usage
<Advanced_Descriptor_Usage>`) section, the ``CalculatorInterface.py`` file is
accessed. This file contains a function for each available descriptor
(see :ref:`Appendix C <Appendix_CI>` for an example function), which first look
directly in the PHU of the AstroData object for the associated descriptor 
keyword as defined in one of the AstroData standard key dictionary files
(``Standard<INSTRUMENT>KeyDict.py``, ``StandardGEMINIKeyDict.py``,
``StandardGenericKeyDict.py`` or ``StandardDescriptorKeyDict.py``). The
keywords in these dictionaries are combined via inheritance to create a final
dictionary containing a single set of keywords associated with a particular
AstroData object. The value of the keyword in the header of the data is
returned as the value of the descriptor. If the associated descriptor keyword
is not found in the header of the data, the descriptor files are then searched
in the order below to attempt to find an appropriate descriptor function, which
will return the value of the descriptor.

.. _Descriptor_Files:

  - ``<INSTRUMENT>_Descriptor.py``
  - ``GEMINI_Descriptor.py``
  - ``Generic_Descriptor.py``

A descriptor function is used if a descriptor requires access to multiple
keywords, requires access to keywords in the pixel data extensions (a
dictionary must be created) and / or requires some validation. If no
appropriate descriptor function is found, an exception is raised
(see the :ref:`Descriptor Exceptions <Descriptor_Exceptions>` section). If a
descriptor value is returned, either directly from the header of the data or
from a descriptor function, the ``CalculatorInterface.py`` file instantiates
the ``DV`` object (which contains the descriptor value) and returns this to the
user.

``CalculatorInterface.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``CalculatorInterface.py`` file contains the ``CalculatorInterface`` 
(``CI``) class, which contains a member function for each available descriptor 
(see :ref:`Appendix C <Appendix_CI>` for an example function). This file is
auto-generated by the ``mkCalculatorInterface.py`` file (:ref:`Appendix B
<Appendix_mkCI>`) and should never be edited directly.

``mkCalculatorInterface.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``mkCalculatorInterface.py`` file contains the code required to generate 
the file ``CalculatorInterface.py``. To create ``CalculatorInterface.py``, 
run the following command::

  shell> python mkCalculatorInterface.py > CalculatorInterface.py

``StandardDescriptorKeyDict.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``StandardDescriptorKeyDict.py`` file contains a Python dictionary named
``globalStdkeyDict``, which describes the one-to-one relationship between a 
descriptor and the AstroData standard keyword associated with that descriptor
(:ref:`Appendix D <Appendix_SDKD>`).

``StandardGenericKeyDict.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``StandardGenericKeyDict.py`` file contains a Python dictionary named
``stdkeyDictGeneric``, which describes the one-to-one relationship between a 
descriptor / variable that is used in the generic descriptor file 
``Generic_Descriptor.py`` and the keyword associated with that descriptor /
variable. The values in this dictionary overwrite (via inheritance) the
AstroData standard values defined in ``StandardDescriptorKeyDict.py``.

``StandardGEMINIKeyDict.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``StandardGEMINIKeyDict.py`` file contains a Python dictionary named
``stdkeyDictGEMINI`` and is used to overwrite (via inheritance) any of the
AstroData standard keywords defined in ``StandardDescriptorKeyDict.py`` and any
of the generic keywords defined in ``StandardGenericKeyDict.py``. It is also
used to define variables for any keywords that need to be accessed in the
Gemini specific descriptor file ``GEMINI_Descriptor.py``.

``Standard<INSTRUMENT>KeyDict.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Standard<INSTRUMENT>KeyDict.py`` files contain a Python dictionary named
``stdkeyDict<INSTRUMENT>`` and is used to overwrite (via inheritance) any of 
the AstroData standard keywords defined in ``StandardDescriptorKeyDict.py``,
any of the generic keywords defined in ``StandardGenericKeyDict.py`` and any of
the Gemini specific keyword defined in ``StandardGEMINIKeyDict.py``. It is also
used to define variables for any keywords that need to be accessed in the
instrument specific descriptor file ``<INSTRUMENT>_Descriptor.py``. These
instrument specific files are located in the corresponding ``<INSTRUMENT>``
directory in the ``astrodata_Gemini/ADCONFIG_Gemini/descriptors`` directory.

``descriptorDescriptionDict.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``descriptorDescriptionDict.py`` file contains four Python dictionaries
named ``descriptorDescDict``, ``detailedNameDict``, ``asDictArgDict`` and
``stripIDArgDict``, which are used by the ``mkCalculatorInterface.py`` file to
automatically generate the docstrings for the descriptor functions located in
the ``CalculatorInterface.py`` file. It is likely that the information in the
``descriptorDescriptionDict.py`` file will be stored in the
``mkCalculatorInterface.py`` file in the future.

.. _calculatorIndex.Gemini.py:

``calculatorIndex.Gemini.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``calculatorIndex.Gemini.py`` file contains a Python dictionary named
``calculatorIndex`` and is used to define which Python object (i.e., the
descriptor class that defines the descriptor functions, in the form
``<module_name>.<calculator_class_name>``) to use as the calculator for a given
``<INSTRUMENT>``. An example of this dictionary is shown below.

::

  calculatorIndex = {
    "<INSTRUMENT>":"<INSTRUMENT>_Descriptor.<INSTRUMENT>_DescriptorCalc()",
    }

``Generic_Descriptor.py``
~~~~~~~~~~~~~~~~~~~~~~~~~

The generic descriptor file ``Generic_Descriptor.py`` contains descriptor 
functions describing those keywords that are part of the FITS standard. There
are currently 53 keywords defined in the FITS standard 
(http://heasarc.gsfc.nasa.gov/docs/fcg/standard_dict.html). There are 4
descriptors currently available that access these FITS standard keywords:

  - ``instrument [INSTRUME]``
  - ``object [OBJECT]``
  - ``telescope [TELESCOP]``
  - ``ut_date [DATE-OBS]``

``GEMINI_Descriptor.py``
~~~~~~~~~~~~~~~~~~~~~~~~

The Gemini specific descriptor file ``GEMINI_Descriptor.py`` contains 
descriptor functions describing those keywords that are standard within
Gemini. There are currently 142 Gemini standard keywords, which are relevant to
the data file as a whole and exist in the PHU of the data. There are 21
descriptors currently available that access these Gemini standard keywords:

  - ``airmass [AIRMASS]``
  - ``azimuth [AZIMUTH]``
  - ``cass_rotator_pa [CRPA]``
  - ``data_label [DATALAB]``
  - ``dec [DEC]``
  - ``elevation [ELEVATIO]``
  - ``local_time [LT]``
  - ``observation_class [OBSCLASS]``
  - ``observation_id [OBSID]``
  - ``observation_type [OBSTYPE]``
  - ``observation_epoch [OBSEPOCH]``
  - ``program_id [GEMPRGID]``
  - ``qa_state [RAWPIREQ, RAWGEMQA]``
  - ``ra [RA]``
  - ``raw_bg [RAWBG]``
  - ``raw_cc [RAWCC]``
  - ``raw_iq [RAWIQ]``
  - ``raw_wv [RAWWV]``
  - ``ut_time [UT]``
  - ``wavefront_sensor [OIWFS_ST, PWFS2_ST, AOWFS_ST]``
  - ``x_offset [XOFFSET]``
  - ``y_offset [YOFFSET]``

``<INSTRUMENT>_Descriptor.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The instrument specific descriptor files ``<INSTRUMENT>_Descriptor.py`` contain
descriptor functions specific to that ``<INSTRUMENT>``. These instrument
specific files are located in the corresponding ``<INSTRUMENT>`` directory in
the ``astrodata_Gemini/ADCONFIG_Gemini/descriptors`` directory.

How to add a new descriptor
---------------------------

The following instructions describe how to add a new descriptor to the system. 

  1. First, check to see whether a descriptor already exists that has the same 
     concept as the new descriptor to be added (:ref:`Appendix A 
     <Appendix_typewalk>`). If a new descriptor is required, edit the
     ``mkCalculatorInterface.py`` file and add the new descriptor to the ``DD``
     constructor in the descriptors list in alphabetical order. Ensure that the
     default Python type for the descriptor is defined:: 
       
       descriptors =   [   
                           ...
                           DD("<my_descriptor_name>", pytype=str),
                           ...
                       ]
  
  2. Regenerate the ``CalculatorInterface.py`` file::
       
       shell> python mkCalculatorInterface.py > CalculatorInterface.py
       
  3. Edit the ``globalStdkeyDict`` dictionary in the 
     ``StandardDescriptorKeyDict.py`` file to include the AstroData standard
     keyword associated with the new descriptor (:ref:`Appendix D
     <Appendix_SDKD>`).

  4. If the new descriptor simply requires access to the AstroData standard
     keyword in the header of the data and returns the value, the descriptor
     can now be tested; go to step 7. 

     However, if the new descriptor requires access to a keyword that is
     different from the AstroData standard keyword (perhaps specific to a
     particular ``<INSTRUMENT>``), go to step 5. If the new descriptor requires
     access to multiple keywords and / or requires some validation, a
     descriptor function must be created; go to step 6.

  5. If the new descriptor requires access to a keyword that is different from
     the AstroData standard keyword, edit either the
     ``StandardGenericKeyDict.py`` file, the ``StandardGEMINIKeyDict.py`` file
     or the ``Standard<INSTRUMENT>KeyDict.py`` file (as appropriate) to include
     the keyword associated with the new descriptor. The descriptor can now be
     tested; go to step 7.

  6. If the new descriptor requires access to multiple keywords and / or 
     requires some validation, a descriptor function should be
     created. Depending on the type of information the new descriptor will
     provide, edit one of the following files to include the new descriptor
     function:

       - ``Generic_Descriptor.py``
       - ``GEMINI_Descriptor.py``
       - ``<INSTRUMENT>_Descriptor.py``

     The ``<INSTRUMENT>_Descriptor.py`` descriptor file is located in the
     ``<INSTRUMENT>`` directory. If the new descriptor is for a new
     ``<INSTRUMENT>``, create an ``<INSTRUMENT>`` directory and edit the
     ``calculatorIndex.Gemini.py`` appropriately. An example descriptor
     function (``detector_x_bin``) from ``GMOS_Descriptor.py`` can be found in
     :ref:`Appendix E <Appendix_descriptor>`. If the descriptor should return
     more than one value, i.e., one value for each pixel data extension, a
     dictionary should be returned by the descriptor function, where the key is
     the ("``EXTNAME``", ``EXTVER``) tuple. If access to a particular keyword
     is required, first check the appropriate keyword files
     (``StandardDescriptorKeyDict.py``, ``StandardGenericKeyDict.py``,
     ``StandardGEMINIKeyDict.py`` and ``Standard<INSTRUMENT>KeyDict.py``) to
     see if it has already been defined. If required, the
     ``Standard<INSTRUMENT>KeyDict.py`` file should be edited to contain any
     new keywords required for this new descriptor function.
 
  7. Update the Python dictionaries in ``descriptorDescriptionDict.py`` so that
     a docstring can be automatically generated for the new descriptor.

  8. Test the descriptor::
       
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
     - The descriptor values can be written to the history, for information
       only. 

  2. Return value Python type

     - The descriptors will always return a ``DV`` object to the user.
     - The ``DV`` object is instantiated by the ``CI`` for descriptors that
       obtain their values directly from the headers of the AstroData object.
       For descriptors that obtain their values from the descriptor functions
       (i.e., those functions located in the :ref:`descriptor files
       <Descriptor_Files>`), the descriptor functions should be coded to return
       a ``DV`` object. The ``DV`` object contains information related to the
       descriptor, including the descriptor value, the default Python type for
       that descriptor and the units of the descriptor.

  3. Keyword access

     - The ``phu_get_key_value`` and ``get_key_value`` AstroData member 
       functions should be used in the descriptor functions to access keywords
       in the header of an AstroData object.

  4. Logging

     - Descriptors will not log any messages.

  5. Raising exceptions

     - If a descriptor value can not be determined for whatever reason, the
       descriptor function should raise an exception.
     - The descriptor functions should never be coded to return None. Instead,
       a descriptor function should throw an exception with a message
       explaining why a value could not be returned (e.g., if the concept does
       not directly apply to the data). An exception thrown from a descriptor
       function will be caught by the ``CI``.

  6. Exception rule

     - Descriptors should throw exceptions on fatal errors.
     - Exceptions thrown on fatal errors (e.g., if a descriptor function is not
       found in a loaded calculator) should never be caught by the ``CI``. The
       high level code, such as script or primitive, should catch any relevant
       exceptions. 

  7. Descriptor names

     - Descriptor names will be:

       - all lower case
       - terms separated with "_"
       - not instrument specific
       - not mode specific, mostly

     - A descriptor should describe a particular concept and apply for all
       instrument modes.

  8. Standard arguments

     - Descriptors accept arguments, some with general purposes are 
       standardized.
     - It is especially important for descriptor arguments to follow the
       Standard Parameter Names as they are front-facing to the user and should
       therefore be consistent.

For example, for raw GMOS data, the ``gain`` descriptor uses the raw keywords
in the header and a look up table to determine the gain value. During the first
data processing step for Gemini data (which includes standardizing the headers
of the data), the value of the raw ``GAIN`` keyword is overwritten (since it
was incorrect in the raw data) and the value of the raw ``GAIN`` keyword is
written to the HISTORY keyword (for information only). If a descriptor is then
called after the first processing step, the ``gain`` descriptor reads the value
directly from the ``GAIN`` keyword. This way, keyword values are always
correct, regardless of the processing state of the data (and any external
system that wishes to work on that data will also access the correct values). 

.. _Descriptor_Exceptions:

Descriptor Exceptions
---------------------

Normally, if a descriptor is unable to return a value, ``None`` is returned
instead. However, exceptions that describe exactly why a value could not be
returned (where applicable) are stored so that a user can access that
information, if they wish to do so. 

When writing descriptor functions, exceptions should be raised in the code with
an appropriate, explicit error message, so that it is clear to the user exactly
what went wrong. The exception is caught by the ``CI`` and if ``throwExceptions
= False`` (line 62 in ``astrodata/Calculator.py``), the exception information
is stored in ``exception_info`` and ``None`` is returned. Otherwise the
exception is thrown. During development, ``throwExceptions = True`` so that
exceptions are thrown. When the code is released, ``throwExceptions = False``
and the exception information will be available in
``exception_info``. Available astrodata exceptions can be found in
``astrodata/Errors.py``. Additional required exceptions can be added to this
file, if necessary. 
