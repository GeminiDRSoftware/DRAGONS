


Concepts
--------


Background
~~~~~~~~~~


Dataset Abstraction
```````````````````

The AstroData class traces back to a request by Gemini Astronomers to
"handle MEFs better" in our reduction package. A *Multi-Extension FITS*,
or *MEF* is the standard storage format that Gemini adopted for its datasets. 
Investigation showed that the MEF libraries were
sufficient for handling MEFs as such and the real meaning of the
request was for a better dataset abstraction for Gemini's datasets.
Gemini MEFs, and MEFs in general, are usually meant to be coherent
collections of data; the separate pixel arrays, or extensions, are
collocated in a common MEF for that reason. The MEF abstraction itself
does not recognize these connections, however, and views the MEF as a
list of separate header-data units, their only relation being
collocation in the list. Even the Primary Header Unit, PHU, which has certain artifacts as
a special header, generally not having pixel data, and which is used
as a file-wide header, is merely presented as the header-data unit at
index 0. AstroData relies on one pair of relational meta-data
available in MEF which is indexing of the list of datasets with
(EXTNAME, EXTVER) tuple. EXTNAME operates as an extension-type
specifier, and EXTVER serves to associate the extention with other
extensions (e.g. by convention ("VAR",2) is the variance plane for
("SCI", 2).

FITS libraries (e.g. ``pyfits``) return opened MEFs as objects which act
as lists of Header-Data Units, or *extensions*. ``AstroData`` on the
other hand is designed to be configured to recognize many internal
connections that MEF does not directly encode. ``AstroData`` detects the
type of the data, and then can make assumptions about what the
data is and how to handle it. Any particular (Python-level) actions on
the data are then performed by implementations in the configuration
space.


Meta-Data
`````````

An additional role of the ``AstroData`` abstraction is to standardize
access to metadata. FITS allows copious metadata in each extension and
in the shared Primary Header Unit (PHU), but it standardizes
only a small subset of what sort of information is stored there. Many
properties which are for Gemini essentially universal properties for
all of our datasets, across instruments and modes, are not
standardized by FITS. For different instruments and modes these bits
of information are distributed across different header key-value pairs
and stored in different units. This leads to a situation where there
is information that is in principle available in all datasets, but
which requires instrument-mode-specific coding to be retrieved in a
particular unit and with a particular technical meaning. AstroData
hides the particulars by allowing functions that calculate the
metadata to be defined in the same configuration space in which the
dataset type itself is defined.

The AstroDataType system is able to look at any aspect of the dataset
to judge if it belongs in a given classification, but the intent is to
find characteristics in the MEF's PHU. Using this knowledge, AstroData
loads and applies particular instrument-mode-specific methods to
obtain general behavior through a uniform interface, as desired for
the developer. This uniform interface can be presented not only in the case
of meta-data but also in the case of transformations and any other
dataset-type-specific behavior.


AstroData Types
~~~~~~~~~~~~~~~

To first order, Astrodata Types map to instrument-modes, and these
provide a good concrete image of what Astrodata Type are. However more
abstract types of dataset identification are also possible and make
themselves useful, such as generic types such as "IFU" vs "IMAGE", or
processing status types such as "RAW" vs "PREPARED".


Recipes and Primitives
``````````````````````

The Astrodata package's "Recipe System" handles all abstractions
involved in transforming a dataset and is built on top of the
AstroData dataset abstraction. The system is called the *Recipe
System* because the top level instructions for transforming data are
"recipes", text files of sequential instructions to perform. For
example the recipe ``makeProcessedBias`` contains the following::

.. code-block:: python
    :linenos:

    
   prepare
   addDQ
   addVAR(read_noise=True)
   overscanCorrect
   addToList(purpose="forStack")
   getList(purpose="forStack")
   stackFrames
   storeProcessedBias


Each of these instructions is either a "primitive", which is a python
function implemented in the configuration space for a dataset of the
given classification, or another recipe. Note that the
"storeProcessedBias" primitive above takes an argument in this
example, "clob(ber)" equals "True", which tells the storage primitive
to overwrite (clobber) any pre-existing bias of the same name.


Zero Recipe System Overhead for AstroData-only Users
++++++++++++++++++++++++++++++++++++++++++++++++++++

Use of AstroData does NOT lead to importing any part of the "Recipe
System". Thus there there is no overhead borne by users of the
AstroData dataset abstraction if they do not specifically invoke the
Recipe System. Neither the configuration package nor even the related
astrodata package modules are imported until the Recipe System is
explicitly invoked by the calling program.


The Astrodata Lexicon and Configurations
````````````````````````````````````````

An Astrodata Configuration package, defining types, metadata, and
transformations, relies on a configuration which defines a lexicon of
elements which are implemented in the configuration package in a way
such that Astrodata can load and apply the functionality involved. Put
simply a combination of location and naming conventions allows the
configuration author to define elements in a way that astrodata will
discover. In the current system there are three types of elements to be
concerned with:


+ dataset classification names, **Astrodata Types**
+ high level metadata names, **Astrodata Descriptors**
+ scientifically meaningful discrete dataset transformation names,
  **Primitives**


Each of these have associated actions:


+ **Astrodata Type**: checks a dataset for adherence to AstroData
  type classification criteria, generally by checking key-value pairs in
  the PHU.
+ **Astrodata Descriptors**: calculate and return a named piece of
  high-level metadata for a particular Astrodata Type in particular
  units.
+ **Primitives**: performs a transformation on a dataset of a
  particular Astrodata Type.


The ``astrodata_Gemini`` package contains these definitions for Gemini
datasets separated into two parts, one for the basic AstroData related
configuration information, and another for Recipe System
configuration. The first section, in its own subdirectory in the
configuration package directory. In Gemini's case is found in the
``ADCONFIG_Gemini`` configuration subdirectory. Configurations in this
subdirectory define types, descriptor functions, and other AstroData-
related features. The second section, in a sibling subdirectory in the
configuration package, in Gemini's case, ``RECIPES_Gemini``. There are defined
configurations and implementations needed by the Recipe System, such
as recipes and primitives.


Astrodata Type
~~~~~~~~~~~~~~

An Astrodata Type is a named set of dataset characteristics.

The lack of a central system for type detection in our legacy package
meant that scripts and tasks made extended checks on
the header data in the datasets they manipulate. Often these checks
only verify that the right type of data is being worked on, a very
common task, yet these checks can still be somewhat complex and
brittle, for example relying on specific headers which may change when
an instrument is upgraded.

The Astrodata classification system on the other hand allows the defining
of dataset classifications via configuration packaging such that the type
definitions are shared throughout the system. The calling code can
refer to type information by a string name for the type, and any
subtleties in or changes to the means of detection are centralized,
providing some forward and backward compatibility. The system also
allows programmers to check dataset types with a single line of code:

.. code-block:: python
    :linenos:

    
    from astrodata.AstroData import AstroData
    
    ad = AstroData("N20091027S0134.fits")
    
    if ad.isType("GMOS_IMAGE"):
       gmos_specific_function(ad)
    
    if ad.isType("RAW") == False:
       print "Dataset is not RAW data, already processed."
    else:
       handle_raw_dataset(ad)


The ``isType(..)`` function on lines 5 and 8 above is an example of 
one-line type checking. The one-line check replaces a larger set of PHU
header checks which would otherwise have to be used. Users benefit in
a forward-compatible way from any future improvements to the named
type, such as better checks or incorporation of new instruments and
modes, and also gain additional sophistication such as type-hierarchy
relationships which are simply not present with the legacy approach.

The most general of benefits to a clean type system is the ability to
assign type-specific behaviors and still provide the using programmer
with a consistent interface to the type of functionality involved.


Astrodata Descriptors
~~~~~~~~~~~~~~~~~~~~~

A descriptor is named metadata.

Significant amounts of information about
the data is present along with the pixel data and much of it is important to data analysis processes.
The `MEF </index.php/MEF>`__ file structure supports such meta-data in
the Primary Header Unit and the header section of the extensions.

The problem retrieving metadata consistently is that
while the values of interest are stored in some form in the headers,
the header key names do not follow consistent conventions across instruments.
It's easy to assume that there is a one to one relationship between
particular metadata headers of different instrument-modes and that the
discrepancy is that the developers have merely chosen different header
key names. If that were the entire problem a table oriented approach
could be used and one could look up the proper header key name for a
particular named piece of metadata based on the type of dataset. This
particular key would be used to look up the information in the
headers.

However, this table-driven approach is not workable because the
situation turns out to be more complex. Firstly, the units of the
given header value may be different for different instruments and
modes. A table could be expanded to have columns for the value's
storage and return type, but expanding the table in this way would
also still not be sufficient for the general case.

The decisive complications that preclude a simple table look-up
approach are two, and lead us to a function-based approach. One, the
information needed to provide the named metadata is sometimes
distributed across multiple key/header values. These require
combination or computation, and for different instruments and modes
the distribution and combination required differ. Two, a correct
calculation of the metadata sometimes requires use of look-up tables
that must be loaded from a configuration space with instrument-specific 
information, based on the dataset's Astrodata Type.

For metadata which complies with the more simple expectations, widely 
shared descriptors for some metadata are standard
functions able to lookup the meta-data based on standard names or
using simple rules that generalize whatever variation there is in the
storage of that particular meta-data across different instruments.
While it is possible for a descriptor to store its calculated value in
the header of the dataset, and return that if called again,
essentially caching the value in the header, Gemini descriptors choose
as a matter of policy to always recalculate, and leave such caching
schemes to the calling program.

A complete descriptor definition includes the proper unit for the
descriptor and a conceptual description. E.g.
Any CCD based data will have an associated "gain", relating to the
electronics used to take the image. Given an AstroData instance, ad ,
to get the "gain" for any supported Astrodata Type, you would use the
following source code regardless of the instrument-mode of the
dataset:

.. code-block:: python
    :linenos:

     gain = ad.gain()


Because the proper descriptors are assigned to the correct Astrodata
Types for Gemini Instruments, the line above will take into account
any type-specific peculiarities that exist for the supported
dataset. The current ADCONFIG_Gemini configuration implementation has
descriptors present for all Gemini instruments.


Recipe System Primitives
~~~~~~~~~~~~~~~~~~~~~~~~

A primitive is a transformation.

A primitive is an specific dataset transformation for
which we will want to assign concrete implementations for the
Astrodata Type. For example, ``subtractSky`` is a transformation that has
meaning for a variety of wavelength regimes which involve subtracting
sky frames from the science pixels. Different
instruments in different modes will require different implementations
for this transformation, due both to differences in the data type and
data layout produced by a particular instrument-mode, and also due to
different common reduction practices in different wavelength regimes.

Recipe and primitive names both play a role bridging the gap between
what the computer does and what the science user expects to be done.
The primitives are meant to be human-recognizable steps such as come
up in a discussion among science users about data flow procedures. The
recipes are, loosely, the names of data processing work.
This puts a constraint on how functionally fine grained primitives
should becomes. For example at Gemini we have assumed the concept of
primitives as "scientifically meaningful" steps means the data should
never be in an incoherent or invalid state, scientifically, after a
given step. Each step is at least a mini-milestone in a reduction
process. So, for example, no primitive should require another
primitive to be run subsequent in order to complete its own
transformation, and primitives should always output valid, coherent
datasets. For example, there should not be a primitive that modifies pixel
data which is followed by a primitive which modifies the header to
reflect the change, and instead both steps should be within such a
primitive so the data is never reported to the system in an invalid or
misleading state.

Recipes can also call other recipes.  This allows refactoring between
recipes and primitives as the set of transformation evolves. A recipe
called by a higher level recipe is seen as an atomic step at the level
of the calling recipe. Coherent steps
which can be broken down into smaller coherent steps are thus probably
best addressed with a recipe calling a recipe. This feature helps
recipes to work for more types. In the end though, primitives have to be
executed so that actual python can run and manipulate the dataset. Below 
a certain level of granularity primitives become inappropriate.
Such code, insofar as it is reusable and/or needs to be encapsulated,
is written as functions in utility libraries, such as the Gemini
``gempy`` package.

Formalizing the transformation concept allows us to refactor our data
reduction approaches for unforeseen complications, new information,
new instruments, and so on, without having to necessarily change
recipes that call these transformations, or the named transformations
which the recipes themselves represent. Recipes for specific nodes in
the Astrodata Type tree can also be assigned as needed, and the fact
that recipes and primitives can be used by name interchangeably
ensures that transformations can be refactored and solved with
different levels of recipes and primitives.  This flexibility helps us 
expand and improve the available
transformations while still providing a stable interface to the user.

AstroData is intended to be useful for general Python scripting, that
is, one does not have to write code in the form of primitives to use
Astrodata. Also, the Recipe System is not
automatically imported (i.e. as a result of "import astrodata") so
that no overhead is borne by the AstroData user not making use of
automation features. A script
using AstroData benefits from the type, descriptor, validation, and
other built in data handling features of AstroData. However, such
scripts do not lend themselves to use in a well-controlled automated
system, and thus the Recipe System is provided for when there is need
for such a system in which to execute the transformation, as with the
Gemini Pipeline projects. Unconstrained python scripts lack a
consistent control and parameter interface.

When writing primitives all inputs are provided through the Reduction
Context, and depending on the control system these may come from the
unix command line, the pyraf command line, from a pipeline control
system or other software, or by the calling recipes and primitives.
Primitive functions are written as Python generators, allowing the
control system to perform some tasks for the primitive, such as
history keeping and logging, keeping lists of stackable images,
retrieving appropriate calibrations, and reporting image statistics to
a central database, etc., when the primitive "yields".

The automation system is designed to support a range of automation,
from a "dataset by dataset, fully automated" mode for pipeline
processing of data as it comes in from the telescope, through to
"interactive automation" where the user decides at what level to
initiate automation and where to intervene.

For advanced users it may be of interest that strictly
speaking primitives transform the ``ReductionContext`` object and not
only the input datasets. This context contains
references to all objects and datasets which are part of the
reduction, including the input dataset. While nearly all primitives
will access their input datasets and most will modify the datasets and
report them as outputs to the reduction context, some primitives may
calculate statistics and report these to the reduction context without
reporting pixel data outputs. In this case the stream inputs will be propagated
as inputs to the subsequent primitive. It is the Reduction Context as
a whole that is passed into the primitives as the standard and sole
argument (besides self) for the primitive. The reduction context must
be left in a coherent state upon exit from a primitive.

Below is a prototype recipe in use in our development environment for
testing. It performs some initial processing on RAW data.


.. code-block:: python
    :linenos:

    
    prepare
    overscanSub    
    overscanTrim
    biasSub
    flatField
    findshiftsAndCombine


If that recipe is generic, this means, given that
primitive sets for GMOS_IMAGE, NIRI_IMAGE, etc, implement the named
primitives in the recipe, then when the recipe system executes a line
such as biasSub , it will execute the "biasSub" member of the
appropries PrimitiveSet associate with that type. Thus, if prepare can
be implemented for both types, while biassub requires GMOS and NIRI-
specific implementations, then "prepare" can be implemented as a
shared recipe or in the GEMINI primitive set, while those that require
special implementation are implemented in the appropriate GMOS or NIRI
primitive sets within the correct part of the configuration.



Some Benefits of the Primitive Concept
``````````````````````````````````````

The use of primitives instead of scripts for reduction processes has a
side benefit besides enjoying automation features supplied by
the Recipe System. This benefit is due to the fact that the concept of
the primitive as a named transformation is bound to the spoken
language that Instrument Scientists, astronomers, data analysts and the data
software group at Gemini use to discuss data flow procedures. This
crossover between terms in our formal system and in our less formal
spoken language has promoted consistency between the two. For example,
when breaking reductions down into discrete chunks which can be
implemented and shared when possible the process helps us understand
what truly differentiates implementations of the same named
transformation. Sharing of code not only saves developers the effort
of reimplementation, but more importantly it promotes consistency and
provides locations in the system where wide ranging changes in policy
can be implemented, accommodating the inevitable evolution of reduction
software.

In short, discussing how to break down typical reduction
procedures into recipes made of reusable primitives has had the effect
of clarifying our understanding of these procedures. Sometimes the
responsibilities of tasks in our legacy system had clear boundaries,
such as for gemarith , but for other tasks, such as the "prepare" task
in each instrument's package, the boundaries of responsibility were
less clear. Adapting transformation concepts which are already in
our spoken lexicon to a more structured software environment
represented with concrete implementations, guides us to creating a
clearer definition for ``prepare``. 
Flexibility in the system allows satisfaction of any special needs
while developing truly shared transformation concepts.


Natural Emergence of Reusable Primitives
++++++++++++++++++++++++++++++++++++++++

Reusable code naturally emerges from the process above because the
work of isolating the steps in a data handling process naturally
reveals similar or identical steps present in other processes, which
can then easily be implemented at a shared level. In practice, even if
creating a recipe that is over-all very instrument and mode-specific,
there seem to emerge general purpose steps which can be of benefit in
a toolkit of primitives. New project-specific
tasks will be able to select from and reuse them freely.

Authors of primitives have several options based on the needs of the
project at hand:


#. generalize the previous attempt at a general solution to leverage
   the work already done
#. write a new generalization
#. write a version which is primarily designed to be useful as a
   primitive in the project's use case


The design of the recipes and primitives of the Recipe System is
intended to facilitate negotiating these options in an environment
with fall-backs and which does not cement you into a particular layout
of your transformations. 



Test Case at Gemini Observatory: Refactoring Python Scripts into Recipes and Primitives
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

We (GDPSG and DA teams) have performed the exercise of breaking down a
set of pre-existing scripts into recipes and primitives in the case of
some instrument monitoring scripts which are set up on a cron job.
Separate from the issue of the quality of the code being thus
preserved, the procedure for refactoring into the recipe/primitive
form turned out relatively easy and to involve the following:


#. Finding where (potential) milestone states of the data occur in the
   script being refactored. These are places where the dataset and
   headers are coherent, and any information the reduction context should
   be informed of has been prepared and is available. Note, some
   potential milestone states, when considered too fine grained will be
   bundled together as a single transformation.
#. Naming the source code between each of these milestone states, and
   identifying its input, output, and specific responsibilities.
#. Cutting and pasting (or re-entering) source from the script into a
   primitive set class, adding adapter code which fetches or stores
   information in the reduction context to and from variables the script
   uses in its legacy form. The code can be largely left as-is since
   primitives are simply python code, so long as input/output is adapted
   to the reduction context.
#. Writing a recipe is using the steps created above.


Regarding the quality of the code thus being preserved, while it was
minimal upon analysis, as is often the case it had the advantage of
being deployed and functional. It is the intent of the Recipe system
to allow rapid adaptation of code into the system, as well as to
enable more intimately and well behaved transformations to be
integrated, and for there to be iterative refactoring paths from the
former to the latter.

The primitives in the test case were developed into a separate recipe
package (not in ``astrodata_Gemini/RECIPES_Gemini``) which is added to the
Astrodata package's ``RECIPEPATH`` environment variable.

Even with lack of a formal structure to the refactoring, and the
devil-you-know approach to preserving the functioning of the code, the
process of adaptation to the recipe/primitive structure provides some
natural order and formalism in the process of identifying the *de facto*
transformations in the script. Improvement is incremental. But even in this 
case, at the very least, the
above analysis will lead to a sequential list of the steps in the
script. That alone is a good starting point for making a complete
replacement if that is necessary. Subsequent work on the recipes and
component primitives only improves the exposure of the work, the
consciousness of the ordering of operations, and merging of common
functionality into common code.

In the case of our instrument monitoring example the result of the
refactoring to the Recipe System is functional and in use. The
resulting recipes made use of some primitives from the Gemini library
of primitives, and could benefit from more refactoring allowing both
some primitives from the main package to be used (i.e. the scripts
performed, and primitives were adapted around a custom "prepare" step
on GMOS data), and also to allow several of the primitives created to
be made more robust and moved into the main package.


Recipes calling Recipes
```````````````````````

Recipes can call other recipes.
Primitives, also, can call recipes or other primitives. During
execution, the Astrodata Recipe System makes little distinction
between recipes and primitives and from the view of those invoking
recipes and primitives, recipe and primitive names are
interchangeable. For example, a user executing recipes through the reduce
command line program can just as easilly give a primitive name to the
"reduce" command as a "recipe" name, and reduce will execute the
primitive correctly. Still the general picture we tend to speak of is
one in which we have a top level recipe for standard processes such as
making a processed bias, which list the steps that the data must go
through to complete the processing named by the recipe.

It is a judgment call how fine grained the steps in a recipe should
be, and this in principle drives how fine grained primitives should be.
However, what is appropriate to view in a recipe of a certain name and
scope may not be the same granularity level which is appropriate for
specialists in the data regime being processed, as the recipe will in
general be associated with some general purpose concepts, and should
have meaning for someone with general purpose knowledge. Sometimes if
the top level recipe were to name every step which an Instrument
Analyst or Data Processing Developer found distinct and
"scientifically meaningful" this would lead to a too finely grained
list of steps, which would obscure the big picture of how the
transformation named is executed.

In this case, which is common, then the more finely grained steps
should be bundled together into recipes which then are used as single
statements in higher level recipes. The ability for recipes to call
recipes ensures steps can be named whatever is semantically
appropriate for whatever the scope of the transformation named might
be. At one extreme the recipe system can support a processing paradigm
in pipelines which invokes reduction with the most general
instructions, "do the appropriate thing for the next file", and at the
other extreme it allows users to decide what to treat as atomic
processes and when to intervene.

The fact that primitives (should) always leave datasets at some
milestone of processing provides some security for the user that they
will not perform an operation that puts the dataset in an incoherent
state. Breaking down recipes into sub-recipes and so on into
primitives truncates at the lowest level when we have primitives that,
however focused, modify the data (or reduction context) in some
significant way and leave the dataset at some milestone of reduction,
however minor a "milestone" it may be. It's also possible, especially
if a primitive is adapted from a script, that a primitive will be
monolithic, and cannot be broken down into a recipe until more finely
grained primitives are created. The interchangeability of recipes and
primitive names is meant to encourage such refactoring, as any
reusable set of primitives is considered more useful than a monolithic
primitive performing all the functions of the reusable set at once.


AstroData Lexicon
~~~~~~~~~~~~~~~~~

A lexicon is a list of words, and this is what the designer of an
Astrodata configuration creates. The set of terms adhere to a grammar
(types of elements that can be defined) and establishes a vocabulary
about dataset types, metadata, and transformations. Firstly, the
configurations define string type names, and criteria by which they
can be identified as a given type of dataset. Then they construct
names for and describe metadata one expects to be associated with
these datasets. Finally they create names for and describe
transformations that can be performed on datasets.

Datasets of particular Astrodata Types can thus
be recognized by astrodata and the other type-specific behaviors can
be assigned. For example, the "astrodata_Gemini" package is the public
configuration package defining data taken by Gemini instruments.
Descriptors for all instruments have been created, and early
implementations of primitives for GMOS_IMAGE and GMOS are available
(and under continued development).

For complete documentation of the ADCONFIG_Gemini type and descriptor
package see "Gemini AstroData Type Reference", available at
`http://gdpsg.wikis-
internal.gemini.edu/index.php?title=UserDocumentation <http://gdpsg
.wikis-internal.gemini.edu/index.php?title=UserDocumentation>`__.

The astrodata package itself has no built-in type or descriptor
definitions. It contains only the infrastructure to load such
definitions from an astrodata configuration package directory (the
path of which must appear in the ``PYTHONPATH``, ``RECIPEPATH``, or
``ADCONFIGPATH`` environment variables as a directory complying with
the "astrodata_xxx" naming convention, and containing at least one of
either ``ADCONFIG_<whatever>`` or ``RECIPES_<whatever>`` sub-packages.

Here is an part of the Gemini type hierarchy, the GMOS_IMAGE branch of
the GMOS types:

<img alt="GMOS AstroData Type Tree" style="margin:.5em;padding:.5em;
border:1px black solid" width = "90%"
src="`http://ophiuchus.hi.gemini.edu/ADTYPETREEIMAGES/GMOS_IMAGE-tree-
pd.png <http://ophiuchus.hi.gemini.edu/ADTYPETREEIMAGES/GMOS_IMAGE-
tree-pd.png>`__"/>

This diagram shows GMOS_IMAGE is a child type of the GMOS type, which
in turn is a child of the GEMINI type. The children of GMOS_IMAGE are
other types which share some or all common primitives or other
properties with GMOS_IMAGE, but which may in some cases require
special handling. The diagram shows descriptor calculator and
primitive set assignments. A descriptor calculator (a set of
descriptor functions) is assigned to GMOS, from which GMOS_IMAGE and
GMOS_SPECT inherit the same descriptors as there is nothing more
specific assigned.

The graph also shows primitive sets assigned to GEMINI, GMOS, and
GMOS_IMAGE. Since a primitive set specific to GMOS_IMAGE is present in
the configuration, it would be used for transformations applying to
GMOS_IMAGE datasets rather than the GMOS or GEMINI primitives. However
the primitive set class for GMOS_IMAGE happens to be defined in
astrodata_Gemini as a child class of the GMOS primitive set, and the
GMOS primitive set as the child of the GEMINI primitive set, so in
fact, the members can be shared unless intentionally overridden.

Primitives associated with the GEMINI Astrodata Type are generally
just bookkeeping functions which rely on features of the Recipe System
as few pixel transformations can be entirely generalized across all
Gemini datasets, though some can.

