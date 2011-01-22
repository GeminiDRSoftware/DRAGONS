


Concepts
--------


Background
~~~~~~~~~~


Dataset Abstraction
```````````````````

The AstroData class traces back to a request by Gemini Astronomers to
"handle MEFs better" in our reduction package. A "MEF" is of course a
"Multiple-Extension FITS File" and is also Gemini's standard dataset
storage format. Investigation showed that the MEF libraries were
sufficient for handling "MEFs" as such and the real meaning of the
request was for a better dataset abstraction for Gemini's datasets.
Gemini MEF, and MEFs in general, are usually meant to be coherent
collections of data. The separate pixel arrays, or extensions, are
collocated in a common MEF because they are related. The MEF
abstraction does not recognize these connections, however, and view
the MEF as a list of separate header-data units, their only relation
being collocation in the list. FITS libraries (e.g. pyfits) return
opened MEFs as objects which act as lists of Header-Data Units, aka
lists of "extensions". The libraries do not recognize the purpose of
one extension versus that of another, nor many semantic relationships
between the extensions except as part of a list and as sub-lists of
identically named extensions, adding a secondary indexing mechanism
using (EXTNAME, EXTVER) tuples.

AstroData on the other hand is designed to be configured to recognize
many internal connections that MEF does not directly encode. AstroData
detects type type of the data, and then can make sound assumptions
about what the data is and how to handle it. Any particular (python-
level) actions on the data are then performed by implementations in
the configuration space.

An additional role of the AstroData abstraction is to standardize
access to metadata. FITS allows copious metadata in each extension and
in the shared "zero'th" extension (aka "the PHU"), but it standardizes
only a small subset of what sort of information is stored there. Many
properties which are for Gemini essentially universal properties for
our datasets, across instruments and modes, are not standardized as a
result. These bits of information which are commonly required are
distributed across different header key-value pairs, in different
units, for different instruments and modes. This leads to a situation
where there is a certain bit of information which is ubiquitously
available in all datasets, but which requires instrument-mode-specific
coding to be retrieved to a particular unit and meaning. AstroData
hides the particulars by allowing functions that calculate the
metadata to be defined in the same configuration space in which the
dataset type itself is defined.

The AstroDataType system is able to look at any aspect of the dataset
to judge if it belongs in a given classification, but the intent is to
find characteristics in the MEF's PHU. Using this knowledge, AstroData
loads and applies particular arbitrarily instrument-mode-specific
methods to obtain the general behavior required by the user, not only
in the case of metadata but also in the case of transformations and
indeed, for any dataset-type-specific behavior. To first order,
Astrodata Types map to instrument-modes, and these provide a good
concrete image of what Astrodata Type are. However more abstract types
of dataset identification are also possible and make themselves
useful, such as generic types such as "IFU" vs "IMAGE", or processing
status types such as "RAW" vs "PREPARED".


Dataset Transformations
```````````````````````

The Astrodata package's "Recipe System" handles all abstractions
involved in transforming a dataset and is built on top of the
AstroData dataset abstraction. The system is called the "recipe
system" because the top level instructions for transforming data are
"recipes", text files of sequential instructions to perform. For
example the recipe "overscanCorrect" contains the following (comments
removed):

.. code-block:: python
    :linenos:

    
    prepare
    overscanCorrect
    addVARDQ
    setStackable
    averageCombine
    storeProcessedBias(clob=True)


Each of these instructions is either a "primitive", which is a python
function implemented in the configuration space for a dataset of the
given classification, or another recipe. Note that
"storeProcessedBias" primitive above takes an argument in this
example, "clob(ber)" equals "True", which tells the storage primitive
to overwrite any previous versions of the bias produced.


NOTE: Recipe System Separate from Astrodata Core
++++++++++++++++++++++++++++++++++++++++++++++++

Use of AstroData does not import any aspects of the "Recipe System",
so there is no overhead on users of AstroData borne from the Recipe
System, neither the configuration package or even the relevant
"astrodata package" modules are imported until the Recipe System is
explicitly used. Our desire with transformations was to have a system
in which high level transformations could be build of low level
transformations, and users and automation systems alike (e.g.
pipelines) could invoke these transformations at whatever level of
interactivity was appropriate for the particular class.


The Astrodata Lexicon and Configurations
````````````````````````````````````````

An Astrodata Configuration package, defining types, metadata, and
transformations, relies on a configuration which understands a lexicon
of elements which are implemented in the package in a way such that
Astrodata can load and apply the functionality involved. In the
current system there are three types of terms to be concerned with:


+ dataset classification names, aka **Astrodata Types**
+ high level metadata names, aka **Astrodata Descriptors**
+ scientifically meaningful discrete dataset transformation names, aka
  **Primitives**


Each of these have associated actions:


+ **Astrodata Type**: checks a dataset for adherence to a
  classification criteria, generally by checking key-value pairs in the
  PHU.
+ **Astrodata Descriptors**: calculates a named piece of high-level
  metadata for a particular Astrodata Type.
+ **Primitives**: performs a named transformation on a dataset of a
  particular Astrodata Type.


The "astrodata_Gemini" package contains these definitions for Gemini
datasets separated into two parts of the configuration. The first
section, in its own subdirectory in the configuration packages, is
ADCONFIG_Gemini, which defines types, descriptor functions, and other
AstroData-related features. The second section, in its own
subdirectory in the configuration package, is RECIPES_Gemini, which
defines configurations and implementations needed by the Recipe
System, such as recipes and primitives.


Astrodata Type
~~~~~~~~~~~~~~

An Astrodata Type is a named set of dataset characteristics.

Lack of a central system for type detection in our legacy package
meant that scripts and tasks in that system make extended checks on
the header data in the datasets they manipulate. Often these checks
merely verify that the right type of data is being worked on, a very
common task, yet these checks can still be somewhat complex and
brittle, for example relying on specific headers which may change when
an instrument is upgraded.

Astrodata's classification system on the other hand allows defining
dataset classifications in configuration packages such that the type
definitions are shared throughout the system. The calling code can
refer to type information by string that names the type, and any
subtitles in or changes to the means of detection are centralized,
allowing programmers to check dataset types with a single line of
code:

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


The `isType(..)` function on lines 5 and 8 above is an example of one-
line type checking. The one-line check replaces a larger set of PHU
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

It goes without saying that our scientific datasets contain (and
require) copious metadata. Significant amounts of "information about
the information" is present along with the pixel data regarding an
observation and much of it is important to data analysis processes.
The `MEF </gdpsgwiki/index.php/MEF>`__ file structure supports such
meta-data in the header units of the primary and other extension HDUs.

At first blush the problem retrieving metadata consistently is that
while the values of interest are stored in some form in the headers,
the header key names do not follow consistent conventions over all.
It's easy to assume that there is a one to one relationship between
particular metadata headers of different instrument-modes and that the
discrepancy is that the developers have chosen different header key
names. If that were the entire problem a table oriented approach could
be used and one could look up the proper header key name for a
particular named piece of metadata based on the type of dataset. This
particular key would be used to look up the information in the
headers.

This table driven approach is not workable because the situation turns
out to be more complex. Firstly, the units of the given header value
may be different for different instruments and modes. A table could be
expanded to have a column for the value's storage and return type, but
expanding the table in this way would still not be sufficient.

The decisive complications that preclude a simple table look-up
approach are two. One, the information needed to provide the named
metadata is sometimes distributed across multiple key/header values.
These require combination or computation, and the for another
instrument or mode the information is in such cases sometimes
distributed differently. Two, a correct calculation of the metadata
sometimes requires use of look-up tables that must be loaded based on
the dataset's Astrodata Type.

For metadata which complies with the more simple expectations of the
first consideration, widely shared descriptors for some metadata are
able to lookup the data based on standard names or due to simple rules
that generalize whatever variation in storage of that particular
metadata. While it is possible for a descriptor to store it's
calculated value in the header of the dataset, and return that if
called again, essentially caching the value in the header, Gemini
descriptors always recalculate, and leave such caching to the calling
program.

A descriptor is named piece of metadata, complete with proper unit and
a conceptual description (`Template:URL GEMINI DESCRIPTORS </gdpsgwiki
/index.php?title=Template:URL_GEMINI_DESCRIPTORS&action=edit&redlink=1
>`__). E.g. Any CCD based data will have an associated "gain",
relating to the electronics used to take the image. Given an AstroData
instance, ad , to get the "gain" for any supported Astrodata Type, you
would use the following source code regardless of the instrument-mode
of the dataset:

.. code-block:: python
    :linenos:

     gain = ad.gain()


Because the proper descriptors are assigned to the correct Astrodata
Types for Gemini Instruments, the line above will take into account
any type-specific peculiarities of any supported dataset. The current
ADCONFIG_Gemini configuration implementation has descriptors present
for all Gemini instruments. See "Gemini AstroData Type Reference"
(`http://www.gemini.edu/INSERTFINALGATREFURLHERE
<http://www.gemini.edu/INSERTFINALGATREFURLHERE>`__) for a list of
available descriptors for Gemini data. Note that descriptor names
themselves are not covered in the Astrodata Users Manual itself
because they are part of the type-specific configuration package.


Recipe System Primitives
~~~~~~~~~~~~~~~~~~~~~~~~

A primitive is a named transformation.

A primitive is meant to name an abstract dataset transformation for
which we will want to assign concrete implementations on a per
Astrodata Type basis. E.g. "subtractSky" is a transformation that has
meaning for a variety of wavelength regimes which involve subtracting
sky frames from the science pixels. Nevertheless, different
instruments in different modes will require different implementations
of this step, due both to differences in the data and data layout
produced by a particular instrument-mode, and also to different
reduction practices common in different wavelength regimes.

Recipe and primitive names both have rolls bridging the gap between
what the computer does and what the science user expects to be done,
which details to expose, i.e. in a name, and which details to obscure
and assume are unimportant if done "properly for the given type of
data". The primitives are thus human-recognizable steps such as come
up in a discussion among science users about data flow procedures. The
recipes are, loosely, the names of data processing work, and the
primitves are names for human-recognizable steps in that process. This
puts a constaint on how fine grained primitives should becomes, for
example at Gemini we have assumed the coherent of primitives as steps
means the data should never be incoherent or invalid scientifically
after a given step. That is, no step should require another step to
complete for its own transformation to be considered complete.

The fact that recipes can call recipes addresses the different levels
of conception of what is considered a complete step in terms of the
degree of the transformation. That is, a recipe called by a recipe at
one level is seen as an atomic step, but to experts in the mode being
processed, this recipe in turn is made of coherent steps. At bottom
primitives have to be executed so that actual python can run and
manipulate the data but below a certain level of granularity
primitives become inappropriate, and such code, insofar as it is
reuseable and/or needs to be encapsulated, appears in utility
libraries, such as the Gemini "gempy" package.

Formalizing the transformation concept allows us to refactor our data
reduction approaches due to unforeseen complications, new information,
new instruments, and so on, without having to necessarily change
recipes that call these transformations, or the named transformations
which the recipes themselves represent. Recipes for specific nodes in
the Astrodata Type tree can also be assigned as needed, and the fact
that recipes and primitives can be used by name interchangeably
ensures that transformations can be refactored and solved with
different levels of recipe and primitive. This flexibility helps us
expand and improve the available transformations while still providing
a stable interface to the user.

AstroData is intended to be useful for general python scripting, that
is, one does not have to write code in the form of primitives to use
Astrodata. And, as mentioned previously, the Recipe System is not
automatically imported (i.e. as a result of "import astrodata") so
that no overhead is borne by the AstroData user not making use of
automation features, such as when writing a script. A script using
AstroData benefits from the type, descriptor, validation, and other
built in data handling features of AstroData. However, such scripts do
not lend themselves to use in a well-controlled automated system, and
thus the Recipe System is provided for when there is need for such a
system, as with the Gemini Pipeline projects. Unconstrained python
scripts lack a consistent control and parameter interface.

When writing primitives all inputs are provided through the Reduction
Context, and depending on the control system these may come from the
unix command line, the pyraf command line, from a pipeline control
system or other software, or by calling recipes and primitives.
Primitive functions are written as python generators, allowing the
control system performs some tasks for the primitive, such as history
keeping and logging, keeping lists of stackable images, retrieving
appropriate calibrations, reporting image statistics to a central
database, etc., when the primitive "yields".

The automation system is designed to support a range of automation,
from dataset by dataset fully automated mode for pipeline processing
data as it comes from the telescope, through to "interactive
automation" where the user decides at what level to initiate
automation and where to intervene.

As users advance it may be of interest to know that primitives,
strictly speaking, transform the"Reduction Context" object and not
specifically (or merely) the input datasets. This context contains
references to all objects and datasets which are part of the
reduction, including the input datasets. It is the Reduction Context
as a whole that is passed into the primitives as the standard and sole
argument for the primitive, and which must be left in a coherent state
upon final exit. For example, a primitive to calculate "seeing
quality" will not actually modify the dataset, but it will in fact
modify the Reduction Context by reporting the calculated statistic to
the reduction context via the ReductionContext class' API.

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


Presume the above is a generic recipe. This means, given that
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

Use of primitives instead of scripts for reduction processes has a
major side benefit besides enabling automation features through the
Recipe System, due to the fact that it is bound to the language
members of data flow teams at Gemini, as well as Instrument Scientists
and PIs, use to discuss data flow procedures. This requirement has
promoted being consistent when breaking reductions down into discrete
chunks. The fact that the steps have to be implemented with common
interfaces, ensures we create conceptions of comprehensible
transformations that can be implemented.

Discussing how to break down our classical reduction procedures into
recipes made of reusable primitives has had the effect of clarifying
our understanding of these procedures. Steps with re-use potential in
other recipes should be contained as separate primitives, and this
becomes clear in the process of regularizing procedures into clear
reduction steps (i.e. into recipes). Even in very instrument and mode-
specific processing there seem to exist general purpose steps which
benefit a general purpose reusable reduction toolkit of primitive
operations.

In effect, primitives have become our natural data flow language. As
we design primitives, we end up formalizing and implementing the very
terms we use to describe our data flow. As we discuss new reduction
features, we invent terms which then are judged on the possibility of
making the abstraction and also implementing the concrete methods
given the inputs specified in the abstractions. In this way we judge
and rejudge our abstractions and approaches, finding holes in either.


Brief Aside: Test Case at Gemini Observatory Refactoring to Primitives
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

We (GDPSG and DA teams) have performed the exercise of breaking down a
set of pre-existing scripts into recipes and primitives. The procedure
turned out relatively easy:


#. finding where the natural discrete transitions occur in the scripts
   involved, these are places where the input datasets and the entire
   reduction context are in a coherent state (milestone states).
#. identifying each of these transitions as named primitives, identify
   their input and output.
#. cutting and pasting script code into a primitive and adapting the
   code to obtain whatever input is needed from the reduction context. If
   the code produced information which informed another part of the
   original script which is now in another primitive, then the code it
   adapted to store the information in the reduction context so it is
   available to the other primitive.
#. a recipe is created listing the steps so created.


The primitives in this case were developed into a separate recipe
package from RECIPES_Gemini, added to the RECIPEPATH environment
variable (they can appear in PYTHONPATH but this is not always
desirable and not necessary since they are accessed by astrodata and
not directly imported). As a stand alone package for a particular
internal purpose (Instrument Monitoring) it was not as important for
these primitives to follow idealized standards, i.e. in terms of
robustness or methodology, as it is for the general library of Gemini
primitives. Thus, instead of formal design, these primitives were
abstracted from the ad hoc design of the scripts that had been doing
the work. Even so, since the original, ad hoc, source code is hidden
within named steps, the primitives, the recipe is a good high level
description of the scripts original algorithmic process, information
which was far from evident perusing the original scripts themselves.
Subsequent work on the recipes only improves the exposure and ordering
of the processes involved.

In the case of our instrument monitoring example the result of the
refactoring to the Recipe System is functional and in use. The
resulting recipes made use of some primitives from the Gemini library
of primitives, and could use more with some refactoring (i.e. they
performed their own "prepare" step on GMOS data). Furthermore, several
of the primitives created from the scripts proved of probable general
interest, such as a primitive which can retrieve data from the GSA.


Recipes calling Recipes
```````````````````````

Recipes can in fact call other recipes as well as primitive, as can
primitives. During execution the Astrodata Recipe System makes little
distinction between recipes and primitives or from the caller's
perspective. Calling code can give a primitive name to the "reduce"
command as a "recipe" and reduce will execute the primitive directly.
Still the general picture is that we tend to have top level recipes
for standard processes such as making a processed bias, which list the
steps that the data must go through to complete the processing named
by the recipe.

It is a judgment call how fine grained this list should be, and what
is appropriate to view in a recipe of a certain scope may not be the
same granularity appropriate for specialists in the data regime being
processed. Sometimes if the top level recipe were to name every step
which an Instrument Analyst of Data Processing Developer found
distinct, then the more finely grained steps should be bundled
together into a recipe which can be called as a single step from a
higher level recipe. The ability for recipes to call recipes ensures
steps can be named whatever is semantically appropriate for whatever
level of processing is of interest to the user in a particular
situation. This means that at one extreme the recipe system can
support a processing paradigm in pipelines which invoke reduction with
the most general instructions, "do the appropriate thing for the next
file", and at the other extreme allows users to decide what to treat
as atomic processes and when to intervene.

The fact that primitives (should) always leave datasets at some
milestone of processing provides some security for the user that they
will not perform an operation that puts the dataset in an incoherent
state.

Breaking down recipes into sub-recipes and so on into primitives
truncates at the lowest level when we have primitives that, however
focused, modify the data (or reduction context) in some significant
way and leave the dataset at some milestone of reduction, however
minor. It's also possible, especially if a primitive is adapted from a
script, that a primitive will be monolithic, and cannot be broken down
into a recipe until more finely grained primitives are created. The
interchangeability of recipes and primitive names is meant to
encourage such refactoring, as any reusable set of primitives is
considered more useful than a monolithic primitive performing all the
functions of the reusable set at once.


AstroData Lexicon
~~~~~~~~~~~~~~~~~

A lexicon is a list of words, and this is what the designer of an
Astrodata configuration creates at the top level of abstraction.
Firstly, they words that identify types of dataset. Then they
construct words that describe metadata one expects to be associated
with these datasets, and finally they create words that describe
transformations that can be performed on datasets.

Astrodata Types sufficiently defined can be recognized by astrodata
and thus the other behaviors can be inferred. For example, the
"astrodata_Gemini" package is the public configuration package
defining data from Gemini instruments. Descriptors for all instruments
have been created, and early implementations of primitives for
GMOS_IMAGE and GMOS are available (and under development).

For complete documentation of the ADCONFIG_Gemini type and descriptor
package see {{GATREFNAME}, available at
`http://www.gemini.edu/INSERTFINALGATREFURLHERE
<http://www.gemini.edu/INSERTFINALGATREFURLHERE>`__.

The astrodata package itself has no built-in type or descriptor
definitions. It contains only the infrastructure to load such
definitions from an astrodata configuration package directory (which
appears in the PYTHONPATH, RECIPEPATH, or ADCONFIGPATH environment
variables as a directory complying with the "astrodata_xxx" naming
convention, and containing at least one ADCONFIG_xyz or RECIPES_xyz
sub-package.

Here is an part of the Gemini type hierarchy, the GMOS_IMAGE branch of
the GMOS types:


.. figure:: images_types/GMOS_IMAGE-tree-pd.png
    :scale: 90
    :figwidth: 5.4in
    :figclass: align-center
    
    GMOS AstroData Type Tree
    


This diagram shows GMOS_IMAGE is a child type of the GMOS type, which
in turn is a child of the GEMINI type. The children of GMOS_IMAGE are
other types which share common primitives or other properties with
GMOS_IMAGE, but which may in some cases require special handling. The
diagram shows descriptor calculator and primitive set assignments. A
descriptor calculator (a set of descriptor functions) is assigned to
GMOS, from which GMOS_IMAGE inherits as there is nothing more specific
assigned and GMOS_SPECT shares the GMOS descriptors which work in both
cases.

The graph also shows primitive sets assigned to GEMINI, GMOS, and
GMOS_IMAGE. Since a primitive set specific to GMOS_IMAGE is present in
the configuration, it would be used for transformations applying to
GMOS_IMAGE datasets. Note, the primitive set class for GMOS_IMAGE
defined in astrodata_Gemini uses the GMOS primitive set as a parent
class, and only overwrites those primitives which require special
handling in the GMOS_IMAGE case. Thus GMOS_IMAGE still uses GMOS (and
GEMINI) primitives, and the assignment paradigm allows overriding just
those which need special handling.

Primitives associated with the GEMINI Astrodata Type are generally
just bookkeeping functions as few transformations can be generalized
across all Gemini datasets, though some can. Some are planned to be
moved to a still more general, MEF, type.

