Creating An AstroDataType
!!!!!!!!!!!!!!!!!!!!!!!!!

Overview
@@@@@@@@

AstroData types are defined in Python classes located in either of two path 
locations in the configuration package::

* ``astrodata_Sample/classification/types`` - for typological types
* ``astrodata_Sample/classification/status`` - for types related to processing status.

The type definition syntax is equivalent in both cases,
the distinction is only for organization between two
sorts of dataset classfication:


#. Classifications that characterize instrument-modes or generic *types* 
   of dataset.
#. Classifications that characterize the processing state of data.

For example, from the ``astrodata_Gemini`` configuration, the ``RAW`` and
``PREPARED`` are "processing types" in ``astrodata_Gemini/status/...``, whereas
``NICI``, ``GMOS`` and ``GMOS_IMAGE`` are "typological types" located in the
``astrodata_Gemini/status/...`` subdirectory directory.

Since we don't know anything about the instrument or mode that this  custom package is
being developed for, the sample package will add some somewhat artificial example types
as processing types,  provided as examples in the sample package that will demonstrate
the point in general. For more complicated examples of type requirements, we'll
use examples from ``astrodata_Gemini``..


The Class Definition Line by Line
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

To inspect the types in the custom package change 
directory to ``astrodata_Sample/classifications/status`` and get a directory
listing::

    cd <base_path>/astrodata_Sample/classifications/status
    cat adtype.UNMARKED.py

The contents of the file should be as below:   

.. code-block:: python
   :linenos:
    
    class UNMARKED(DataClassification):
        name="UNMARKED"
        usage = "Processing Type for data not yet 'marked'."
        parent = "OBSERVED"
        requirement = PHU({"{prohibit}THEMARK":'.*'})

    newtypes.append(UNMARKED())

Note that type source files are read into memory and executed in a prepared environment. Thus
there is no need to import
the ``DataClassification`` class from the particular astrodata module,
this standard base class is already in
scope. 

The two elements are the class itself and the ``newtypes.append(UNMARKED())`` line
which instantiates an object of the class and appends it to a list that the
ClassificationLibrary can use to inspect datasets. The ClassificationLibrary uses the
``newtypes`` list to recieve types defined in the module, allowing multiple types to be
added to this list in a single type module if desired. At Gemini we have decided to
have just one type definition per python type file.


1. ``class UNMARKED(DataClassification)``:
   By convention, we name the class identically to the chosen string name, in
   this case ``UNMARKED``, however this is not required by the system.
   
2. ``name="UNMARKED"``:
   The classification ``name`` property stores the string used by the system
   to identify the type. NOTE: when using type functionality, the user never
   sees the classification object, and deals with types as strings.
    
3. ``usage="Processing Type for data not yet 'marked'."``:
   This is used for automatically generated documentation.

4. ``parent="OBSERVED"``:
   This is the type name of a parent class.  Note, the type need not also be
   recognizes as the parent type.  The parent member is used to determine
   overriding assignments in the type tree such that, of course, leaf nodes
   override root nodes, e.g. for descriptor calculator and primitive set
   assignments.
   
5. ``requirement = PHU({"{prohibit}THEMARK":'.*'})``:
   The requirement member uses requirement classes (see below) to define the given type. 
   In this case, this is a PHU check to ensure that the header keyword "THEMARK" is not set at all
   in the PHU.
   
6. ``newtypes.append(UNMARKED())``:
   This line appends an object instance of the new class to a pre-defined 
   ``newtypes`` array variable. Note, this name is the **class name** from line
   1, not the type name, though by convention in Gemini AstroData Types we use
   the type name as the class name.
   
   
The Requirement Classes
@@@@@@@@@@@@@@@@@@@@@@@

The requirement member of a type classification is intended to be declared
with an expression built from requirement classes.  Again, the type definition
is evaluated in a controlled environment and these classes, as well as aliases
for convienience, are already in scope.

Concrete Requirements
#####################

Concrete Requirements are those that make actual physical checks of dataset characteristics.

================  =======  ======================================================
Requirement Type  Alias    Description
================  =======  ======================================================
ClassReq          ISCLASS  For ensuring this type is also some other 
                           classification
PhuReq            PHU      Checks a PHU key/value header against a regular 
                           expression.
================  =======  ======================================================

Object Oriented design enables us to extend requirement class ability and/or create new 
requirements.  Examples: the current PHU requirement checks values only against 
regular expressions, it could be expanded to make numerical comparisons (e.g. to
have a dataset type dependent on seeing thresholds). Another example that we 
anticipate needing is a requirement class that checkes header values in extensions.

Currently all type checking resolves to PHU checks, see below for 
a description of the PHU requirement object.

ISCLASS(other_class_name)
$$$$$$$$$$$$$$$$$$$$$$$$$

The ISCLASS requirement accepts a string name and will cause the classification to check
if the other type applies.  Circular definitions are possible and the configuration author
must ensure such do not exist.

ISCLASS example::

    class GMOS(DataClassification):
        name="GMOS"
        usage = '''
            Applies to all data from either GMOS-North or GMOS-South instruments in any mode.
            '''
        parent = "GEMINI"
        requirement = ISCLASS("GMOS_N") | ISCLASS("GMOS_S")
        
        # equivalent to...
        #   requirement = OR(   
        #                    ClassReq("GMOS_N"), 
        #                    ClassReq("GMOS_S")
        #                   )

    newtypes.append( GMOS())

Since there are in fact two GMOS instruments at Gemini, one in Hawaii, one in Chile, the GMOS
type really means checking that one of these two instruments was used.

.. note::
   This is also an example of use of the OR requirement, and specifically a convenience
   feature allowing the "|" symbol to be used for pair-wise or-ing. The included comment 
   shows another form using the OR object constructor
   which allows more than two operands to be listed.

PHU(keyname=re_val, [keyname2=re_val2 [...]])
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

The PHU requirement accepts any number of arguments.  Each argument name  is used as
the PHU key name, and the value is a regular expression against which the header
value will be compared.

An example::

    class GMOS_NODANDSHUFFLE(DataClassification):
        name="GMOS_NODANDSHUFFLE"
        usage = "Applies to data from a GMOS instrument in Nod-And-Shuffle mode"
        parent = "GMOS"
        requirement = PHU(NODPIX='.*')

    newtypes.append(GMOS_NODANDSHUFFLE())

It is also possible to prohibit a match, and to use regular expressions for key matching using a
special syntax for the key name. This is done by prepending an instruction to the key name,
but also requires passing arguments to the PHU object constructor in a different way. For
example the following requirement checks to ensure that the PHU key ``MASKNAME`` *does not*
match ``"IFU*"``::

    PHU({"{prohibit}MASKNAME": "IFU*"})

Note that in this case the arguments are passed to the PHU object constructor as a dictionary.
The keys in the dictionary are used to match PHU keys, and the values are regular expressions
which will be compared to PHU values.

Generally, Python helps instantiating the PHU object by turning the constructor parameter 
names and their settings into the keys and values of the dictionary it uses internally.
However, Python does not like special characters like "{" in argument names, so to use the
extended key syntax requires passing the dictionary.

To use regular expressions in key names (which is also considered dangerous and prone to
inefficiency), use the following syntax::

    class PREPARED(DataClassification):

        name="PREPARED"
        usage = 'Applies to all "prepared" data.'
        parent = "UNPREPARED"
        requirement = PHU( {'{re}.*?PREPARE': ".*?" })

    newtypes.append(PREPARED())

Due to our legacy reduction software conventions, Gemini datasets which have been run
through the system will have a keyword of the sort "<x>PREPARE" with a value set to a
time stamp.  The need for caution is due to, one, efficiency, since the classification
must cycle through all headers to see if the regular expression matches, and two, this
technique is prone to a name collision, i.e. in our example above... if a PHU
happens to have a key matching ``"*PREPARE"`` for some other reason than having been
processed by the Gemini Package.  

Please use this feature with caution.

Logical Requirement Classes
###########################

The logical requirement classes use OO design to behave like requirement operators,
returning true or false based on a combination of requirements given as arguments.

================  =======  ======================================================
Requirement Type  Alias    Description
================  =======  ======================================================
AndReq            AND      For comparing two other requirements with a logical
                           ``and``
NotReq            NOT      For negating the truth value of another requirement
OrReq             OR       For comparing two other requirements with a logical 
                           ``or``
================  =======  ======================================================

AND(<requirement>,<requirement> [, <requirement> [, <requirement> ] .. ])
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

The AND requirement accepts other requirements as arguments. At least two arguments are needed
for the AND to be sensible, but if more are present they are also checked for truth value.

It is possible also to use the "&" operator as a logical "and"::

    requirement = AND(PHU("key1", "val1"), PHU("key2", "val2"))
    
...is equivalent to::

    requirement = PHU("key1", "val1") & PHU("key2", "val2")

NOT(<requirement>)
$$$$$$$$$$$$$$$$$$

The NOT requirement accepts a single other requirement as arguments. 
"NOT" is used to negate some requirement. For example at Gemini we
do not view a GMOS_BIAS as a
GMOS_IMAGE, but it does satisfy the requirements of GMOS_IMAGE. The need
for a separate type is due to the fact that GMOS_IMAGE and GMOS_BIAS require
different automated reduction (e.g. in a pipeline deployment). To accomplish
this we add a ``NOT`` requirement to GMOS_IMAGE::

    class GMOS_IMAGE(DataClassification):
        name="GMOS_IMAGE"
        usage = """
            Applies to all imaging datasets from the GMOS instruments
            """
        parent = "GMOS"
        requirement = AND([  ISCLASS("GMOS"),
                             PHU(GRATING="MIRROR"),
                             NOT(ISCLASS("GMOS_BIAS"))  ])

    newtypes.append(GMOS_IMAGE())

OR(<requirement>,<requirement> [, <requirement> [, <requirement> ] .. ])
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

The OR requirement accepts other requirements as arguments. At least two arguments are needed
for the OR to be sensible, but if more are present they are also checked for truth value.

It is possible also to use the "|" operator as a logical "or"::

    requirement = OR(PHU("key1", "val1"), PHU("key2", "val2"))
    
...is equivalent to::

    requirement = PHU("key1", "val1") | PHU("key2", "val2")

