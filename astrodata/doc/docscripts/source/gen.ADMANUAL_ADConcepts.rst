


Concepts
--------


Background
~~~~~~~~~~


Dataset Abstraction
```````````````````

The AstroData class traces back to a request by Gemini Astronomers to
"handle MEFs better" in our reduction package. A "MEF" is of course a
"Multiple-Extension FITS File", and is also Gemini's standard dataset
storage format. Investigation showed that the MEF libraries were
sufficient for handling "MEFs" as such, and the real meaning of the
request was for a better dataset abstraction. FITS libraries (e.g.
pyfits) return opened MEFs as lists of Header-Data Units, aka lists of
"extensions". The libraries do not recognize semantic relationships
between the extensions except as part of a list (though adding a
secondary listing mechanism using (EXTNAME, EXTVER) tuples. AstroData
on the other hand can be configured to recognise internal connections
that MEF does not directly encode.

An additional role of the abstraction is to standardise access to
metadata. FITS allows copious metadata in each extension and in the
shared "zero'th" extension (aka "the PHU"), but it standardizes only a
small subset of what is stored there. Many properties which are, for
Gemini, universal properties for our datasets across instruments and
modes are thus not standardised and are distributed across different
header key-value pairs. This leads to ubiquitous information being
available, in all datasets but requiring dataset-specific coding for
retrieval. AstroData hides the particulars behind getting a particular
bit of dataset metadata behind a common AstroData interface.

AstroData begins by detecting the type of dataset, ideally on
characteristics of the file available from the PHU, but able to look
at any aspect of the datase4t. Then, using this knowledge, AstroData
can load and apply particular, potentially instrument-specific,
methods to obtain the general behaviour required by the user. To first
order Astrodata Types map to instrument-modes, but more rich types of
dataset identification are possible (such as generic types such as
"IFU" vs "IMAGE").


Dataset Transformations
```````````````````````

The Astrodata package's "Recipe System" handles all abstractions
involved in transforming a dataset and is built on top of the
AstroData dataset abstraction. Note, use of AstroData does not import
any aspects of the "Recipe System", so there is no overhead from the
Recipe System borne due to use only of the AstroData abstraction. Our
desire with transformations was to have a system in which high level
transformations could be build of low level transformations, and users
and automation systems alike (e.g. pipelines) could invoke these
transformations at whatever level of interactivity was appropriate for
the particular class.


The Astrodata Lexicon and Configurations
````````````````````````````````````````

An Astrodata Configuration package, defining types, metadata, and
transformations, relies on a lexicon consisting of three types of
elements, which are implemented in the package in a way such that
Astrodata can load and apply the functionality involved. In the
current system there are three types of terms to be concerned with:


+ dataset classification names, aka **Astrodata Types**
+ high level metadata names, aka **Astrodata Descriptors**
+ scientifically meaningful discrete dataset transformation names, aka
  **Primitives**


Each of these have associated actions:


+ Astrodata Type: checks a dataset for adherence to a classification
  criteria, generally by checking PHU key-value pairs.
+ Astrodata Descriptors: calculates a particular, named, piece of
  high-level metadata for a particular Astrodata Type.
+ Primitives: performs a standard, named, transformation on a dataset
  of a particular Astrodata Type.


The "astrodata_Gemini" package contains these definitions for Gemini
datasets separated into two parts of the configuration. First,
ADCONFIG_Gemini, which defines types, descriptor functions, and any
other AstroData related features. Second, RECIPES_Gemini, which
defines configurations and implementations needed by the Recipe
System, such as primitives.


Astrodata Type
~~~~~~~~~~~~~~

Lack of a central system for type detection in our legacy package
means that scripts and tasks in that system make extended checks on
the header data in the datasets they manipulate. Often these checks
merely verify that the right type of data is being worked on, and yet
they can still be somewhat complex, at the least multi-lines, hard
coded to particular hearder values, and can vary from task to task
even when the same check is intended.

Thus, how a dataset classification is recognised is not presumed to be
consistent throughout the legacy package. Astrodata's classification
system, on the other hand, allows defining dataset classifications in
configuration packages such that the type definitions are shared
throughout the system. This centralizes the meaning of a particular
type and also the official heuristics for detecting it. This allows
programmers to make such checks in a single line of code:

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
type, such as better checks, or incorporating new instruments and
modes, and also gain additional sophistication such as type-hierarchy
relationships which are simply not present in the legacy approach.

The most general of benefits to a clean type system is the ability to
assign type-specific behaviors and still provide the using programmer
with a consistent interface to the type of functionality involved.


Astrodata Descriptors
~~~~~~~~~~~~~~~~~~~~~

It goes without saying that our scientific datasets contain (and
require) copious metadata. Significant amounts of "information about
the information" is present regarding an observation and much of it is
important to a data analysis process. The `MEF
</gdpsgwiki/index.php/MEF>`__ file structure supports such meta-data
in the headers of the primary and extension HDUs. One might presume,
as we did, the problem is that the header have different names, and
that a table driven solution could work, such that when the user
needs, for example, the gain value(s) associated with a dataset, the
problem is the value is stored in header values with differing key
names. One could look the correct name for a particular concept of
metadata up, for a given instrument, and do a header lookup based on
that key.

This approach is not workable. Firstly, the units of the given value
are different. To return the values in different units would break the
point of unifying access to the data, though one could add unit
information to the key-name lookup table, and either convert the value
to a standard unit or at least report the unit to the user. However,
expanding the table approach this way would still not be sufficient.

The reason no table lookup can be a general solution is that the
desired and expected metadata is sometime distributed across multiple
header key/value pairs. As the distribution and meaning of the header
cards in which the information is located therefore in general
requires arbitrary computation to combine. Secondly a correct
calculation sometimes requires use of lookup tables that are not in
the dataset at all, and must be looked up using type information about
the dataset.

For these reasons only a function-based system was deemed general
enough. Thus part of the ADCONFIG_Gemini configuration package
contains functions that can calculate given metadata. These functions
can be shared by branches of the type tree for which the metadata is
calculated (or looked up) identically, and also particular functions
can be assigned to instrument-modes which require a special means of
calculation due to instrument specific behaviors or merely a different
arrangement of raw metadata in the MEF headers.

We call the high-level metadata "Astrodata Descriptors", or just
"descriptors". The descriptor is a concept, both the name and the
meaning of the name, including details such as the units in which the
metadata is returned. Behind this name and concept are implementations
attached to branches of the type-hierarchy which then share the same
calculation method, when that is possible. The configuration includes
tables which assign "descriptor calculators" to Astrodata Types, and
at runtime the correct implementation is looked up using these tables
defined in the configuration itself. Given an AstroData instance, ad ,
to get the "gain" metadata for any supported datatype, you would use
the following code, regardless of the instrument-mode of the dataset:

.. code-block:: python
    :linenos:

     gain = ad.gain()


Because the proper descriptors are assigned to the correct Astrodata
Types, the line above will work for any supported datatype, taking
into account any type-specific peculiarity. The current
ADCONFIG_Gemini configuration implementation has descriptors for all
Gemini instruments. See "Gemini AstroData Type Reference"
(`http://www.gemini.edu/INSERTFINALGATREFURLHERE
<http://www.gemini.edu/INSERTFINALGATREFURLHERE>`__) for a list of
available descriptors for Gemini data.


Recipe System Primitives
~~~~~~~~~~~~~~~~~~~~~~~~

A primitive is meant to name a particular abstract dataset
transformation both in name and idea. The means of implementing the
transformation can and will sometimes be different for different types
of dataset, but the meaning of the transformation should apply in all
cases. E.g. "subtractSky" has the same idea and name for MIR and
Optical data, but will be implemented differently due to different
practices in different wavelength regimes.

The concept of a "recipe" is important to the Recipe System, and is
just a list of "primitives". As with primitives and other Astrodata
features, recipes can be written for particular branches of the type
hierarchy, however, since they are lists of primitives, which already
normalize transformations which may in fact have different
implementations, a "recipe" should tend to be a more general purpose
element, shared higher up the type hierarchy, with instrument specific
behavior happening not in the recipe but during the dispatch to
particular primitives in the recipe. Recipes can contain recipes and
internally recipes are converted "to primitives" at run time, so the
level at which specific behaviour is invoked can be carefully
controlled.

Formalizing the transformation concepts allows us to refactor the
solutions due to unforeseen complications, new information and
instruments, and so on, without having to necessarily change recipes
that call these transformations. This helps us expand and improve the
available transformations while still providing a stable interface to
the user.

AstroData is intended to be useful in general python scripting so one
does not have to write code in the form of primitives, and the Recipe
System is not automatically imported (i.e. as a result of "import
astrodata"). A script using AstroData benefits from the type,
descriptor and other built in data handling features of AstroData but
such scripts do not lend themselves to use in a well-controlled
automated system. They lack a consistent control and parameter
interface. If one wants to take advantage of the automation systems
within the Recipe System, such code is wrapped in primitives which
provides a consistent input/output interface the both the user and the
primitive author.

The automation system is designed to support a range of automation,
from dataset by dataset automation for a pipeline processing data as
it comes from the telescope, through to a more "interactive"
automation where the user decides at what level to initiate
automation.

The primitives themselves are implemented as python "generators", a
type of function from which the programmer can "yield" control such
that the function can be subsequently reentered at the point of the
yield. This ability allows communication and cooperative control to
take place "while" a primitive executes, in a cooperatively
multitasking manner. This allows the controlling system to perform
some services for the reduction (like retrieving calibrations from a
potentially remote source), during "yield" statements.

The astrodata package itself in no way enforces any rules about the
complication or nature of the transformations performed by the
primitive, but such standards are meant to be part of a particular
configuration. In the astrodata_Gemini configuration the general
intention has been that primitives represent transformations which are
arguably "scientifically meaningful". The name of a primitive should
bear some meaning in an offline conversation about data flow.

For examples, some example primitive names are "subtractSky" and
"biasCorrect" which have meaning in conversation about dataflow,
regardless of how they are performed on a particular dataset.
Arbitrarily complex material differences in the dataset may require
very different implementations, but so long as the step is performed
properly for the type processed, the differences are not represented
in the primitive names (and thus not in the recipes invoking them).

Within the primitive there is pure python code and significant
software engineering artifacts, but in the name of the primitives, and
thus in recipes, only a reference to the scientific concepts of the
named primitives exists. There are no explicit conditionals or
variables in recipes. However, the correct implementation for a given
primitive is ensured to be run, and thus there is implicit conditional
behavior in recipes, based on Astrodata Type. Thus recipes are said to
"adapt" to the dataset type being processed at that point in the
recipe, as determined by the dataset being processed at that state of
processing.

As users advance it may be of interest to mention that primitives,
strictly speaking, transform a"Reduction Context" object, not
specifically (or merely) the input datasets. This context contains
references to all objects and datasets which are part of the
reduction, part of which are the input files and is passed into the
primitives as the standard and sole argument for all primitive
generators.

For example, a primitive to calculate "seeing quality" will not
actually modify the dataset, but it will in fact modify the Reduction
Context by reporting the calculated statistic to context via the
ReductionContext class' API.

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



Benefit of the Primitive Concept
````````````````````````````````

Use of primitives instead of scripts for reduction processes has a
major side benefit besides enabling automation features through the
Recipe System, which is that it promotes breaking reductions down into
discrete chunks, each of which is a comprehensible transformation of
one valid dataset into another valid dataset. Discussing how to break
down and name our classical reduction procedures into reusable recipes
and primitives has had the effect of clarifying our understanding of
these procedures.

Steps with re-use potential in other recipes should be contained as
separate primitives, as should those which may be of use if used
directly by a user in isolation of other transformations. In our
experience so far, mapping a general processing script into a recipe,
which means breaking the process down into discrete primitive steps,
leads to discovery of reusable concepts that enrich our data flow
language even if the script had a rather specific purpose. That is,
even in very mode-specific processing there exist general purpose
steps which benefit from a library of mode-general standard
transformations.

In effect, primitives have become our natural data flow language. As
we design primitives, we end up formalizing and implementing the very
terms we use to describe our data flow. When a new primitive is
defined, a new term is created that we use to describe our data flow.

Note: We have performed the exercise of breaking down a set of pre-
existing scripts into recipes and primitives. It turned out relatively
easy to find where the discrete transitions occur in the scripts, and
identify them as primitives. These primitives were developed in a
separate recipe package from RECIPES_Gemini, added to the RECIPEPATH
environment variable (they can appear in PYTHONPATH but this is not
always desirable and not necessary since they are accessed by
astrodata and not directly imported. As a stand alone package for a
particular purpose (Instrument Monitoring) it was not as important to
create idealized primitives as it is for the standardized Gemini
primitives which are intended to be reusable and generally
comprehensible. Instead of formal design, these primitives had been
abstracted from the ad hoc design of the scripts. However, since the
ad hoc source code is hidden within the primitives, the recipe still
is a good high level description of the scripts original algorithmic
shape, and subsequent to the conversion lends itself to careful and
conscious refactoring as deemed worthwhile.

In the case of our instrument monitoring example case, the result of
the refactoring to the Recipe System is functional and in use, and
there was no serious need to change the method used in a significant
way to benefit from the Gemini library of primitives. Also, several of
the primitives created proved of probable general interest (i.e.
retrieving data from the GSA automatically), and would be temptingly
simple to generalize.


Recipes calling Recipes
```````````````````````

Recipes can in fact call recipes, as can primitives. The result is we
tend to have top level recipes which represent the most abstract view
of the data transformation and describe steps most data go through. At
the lowest level we have primitives which represent the most concrete
steps we want want to consider as "arguably scientific", or at least
consider "data flow language" rather than "pure python". In between
can be recipes and primitives with a varying degree of mode-
specificity.

Ultimately, at the lowest level, within the primitives, is of course,
pure python. However, this python code can still be written in a
generic way, and be assigned to a high level type. The result is that
recipes and primitives both can appear anywhere in the type hierarchy
with respect to mode-specificity, with recipes still conceptually at a
higher level, but technically possibly below the "level" of a
particular primitive. Recipes and be refactored into primitives and
vice versa fairly easilly, and design of the system and the
particulars of the development goals in a given case thus can drive
what is a recipe vs what is a primitive in addition to the general
guidline that recipes are used for transformations defined higher in
the type hierarchy.


AstroData Lexicon
~~~~~~~~~~~~~~~~~

These three concepts, starting with Astrodata Type, and then adding
Astrodata Descriptors and Primitives which are assigned to the types,
contribute in a lexicon of terms about the datasets recognized by the
configuration, e.g. in the case of Gemini, as defined in
astrodata_Gemini our astrodata configuration package. Type-specific
behaviors are assigned to branches (or leaves) of the type-tree
hierarchies and will apply to those types below them in the hierarchy
unless overridden by a still more-type-specific assignment. While I
have described two general features which require type-dependent
implementations, the system is arranged such that it is relatively
easy to look up (and therefore assign) any feature or property based
on Astrodata Type in the same manner. Other behaviors can and will
also be assigned this way in the future, for example the Astrodata
Structures feature, currently a prototype implementation, which
provides hierarchical representation of datasets as well as performing
validation functionality. For complete documentation of the
ADCONFIG_Gemini type and descriptor package see {{GATREFNAME},
available at `http://www.gemini.edu/INSERTFINALGATREFURLHERE
<http://www.gemini.edu/INSERTFINALGATREFURLHERE>`__.

The astrodata package itself has no built in type or descriptor
definitions. It contains only the infrastructure to load such
definitions from an astrodata configuration package directory (which
appears in the PYTHONPATH or RECIPEPATH environment variables as a
directory following the "astrodata_xxx" naming convention, and which
also by convention contains the specific "ADCONFIG_xxx" and
optionally, the "RECIPE_xxx" sub-packages). Here is an example of part
of the GMOS type tree graph, specifically for the GMOS_IMAGE branch,
from the current Gemini classification library:


.. figure:: images_types/GMOS_IMAGE-tree-pd.png
    :scale: 90
    :figwidth: 5.4in
    :figclass: align-center
    
    GMOS AstroData Type Tree
    


This graph shows GMOS_IMAGE is a child type of the GMOS type, which in
turn is a child of the GEMINI type. The children of GMOS_IMAGE are
other types which are children of GMOS_IMAGE. The graph shows a
descriptor calculator and primitive set assignments and shows a
descriptor assigned to GMOS which GMOS_IMAGE inherits, as there is
nothing more specific assigned. The graph shows primitive sets
assigned to GEMINI, GMOS, and GMOS_IMAGE. A primitive set specific for
GMOS_IMAGE is present so this is what would be used for
transformations to GMOS_IMAGE objects. Note, the primitive set class
used internally for GMOS_IMAGE uses the GMOS primitive set as a parent
class, and only over writes those primitives which require special
handling in the GMOS_IMAGE case, thus GMOS_IMAGE still uses GMOS and
GEMINI primitives, and the assignment is to allow overriding special
cases in which it is required.

The GEMINI primitives are generally just bookkeeping functions as few
transformations can be generalized across all Gemini datasets, though
some can.

