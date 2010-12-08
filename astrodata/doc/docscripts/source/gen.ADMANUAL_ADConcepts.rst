


Concepts
--------


Background
~~~~~~~~~~

The astrodata system traces back to a request by Gemini Astronomers to
"handle MEFs better" in our reduction package. A "MEF" is of course a
"Multiple-Extension FITS File", and is also Gemini's standard dataset
storage format. Investigation showed that the MEF libraries were
sufficient for handling "MEFs" as such, and the real problem was the
MEFs are presented by fits libraries (e.g. pyfits) as lists of Header-
Data Units, aka "extensions", and relationships between the extensions
are not recognized. Further, while FITS allows copious meta-data, it
standardizes only a subset of what are for Gemini, universal
properties for our datasets. This leads to the ubiquitous information
being available in all datasets, but requiring dataset-specific
retrieval, e.g. via different meta-data key names.

AstroData on the other hand is intended to recognize the connections
MEF does not directly encode. AstroData begins by detecting the type
of dataset, then, using this knowledge, it applies the particular
methods to obtain the general behavior for that particular dataset's
instrument-mode. From AstroData's point of view, the data in a MEF is
not merely incidentally collocated in the MEF, it's presumed to be
there as part of a bundle of data associated with a complex
observation. Thus, to first order, these dataset types tend to map to
instrument-modes and contain all the associated information needed to
make full use of the science data.

Ample header data exists in the datasets to perform this function once
the dataset's type is characterized. Given such definitions in an
AstroData configuration (i.e. in astrodata_Gemini/ADCONFIG_Gemini),
the AstroData class will look at data in type-specific ways when
needed, but maintain a consistent, type-independent interface for the
AstroData user. When code can be shared among types the behavior is
assigned to more general types those datasets share (e.g. "GMOS_IMAGE"
and "GMOS_IFU" share the "GMOS" type).

To use the package astrodata you will develop or use a lexicon of
terms which describe your data and the processing you want to do. In
the current system there are three types of terms to be concerned
with:


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



Astrodata Type
~~~~~~~~~~~~~~

Lack of a central system for type detection in our legacy package
meant that scripts and tasks had to make extended checks on the header
data in the datasets they manipulate. Often these checks merely verify
that the right type of data is being worked on, but are multi-lined
and can vary from script to script even when essentially the same
check is intended. Thus, how a dataset is recognized as belonging to a
particular instrument-mode is not presumed to be consistent throughout
the legacy package. Astrodata's classification system, on the other
hand, allows defining dataset classifications in configuration
packages centralizing the meaning of a particular mode and also the
official heuristics for detecting it. This allows users to make such
checks in a single line of code:

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
header checks which would otherwise have to be used. These checks are
centralized within AstroData, which provides an interface for
accessing type information that is presented as string names in the
instance's "types list". Users benefit in a forward-compatible way
from any future improvements to the named type, and also gain
additional sophistication such as type-hierarchy relationships which
are simply not present in the legacy approach.


Astrodata Descriptors
~~~~~~~~~~~~~~~~~~~~~

It goes without saying that our scientific datasets require copious
metadata. Significant amounts of "information about the information"
is present regarding an observation. The `MEF
</gdpsgwiki/index.php/MEF>`__ file structure supports such meta-data
in the headers of the primary and extension HDUs. Based on the idea
that the information is thus available in the headers and merely
hidden under differently named keys we first investiaged a table
driven solution. In such a solution a standard key name is used in the
code to look up the actual metadata key name for a given particular
instrument type, via a lookup table.

The problems with this approach turned out to be multiple. First there
is the problem that the desired and expected metadata is sometime
distributed across multiple header key/value pairs, and may require
arbitrary computation to combine. Secondly it sometimes requires
lookup tables containing information associated with, but not present
in, the target dataset. Both these, as well as other cases, preclude a
simple table driven system. They require the flexibility of a function
that can perform arbitrary manipulations and calculations. Thus the
high-level metadata "lookup" invokes functions written to calculate
the value from a combination of low-level metadata and configuration
information.

We call the high-level metadata "Astrodata Descriptors", or just
"descriptors". The descriptor is a concept, both the name and the
meaning of the name, including details such as the units in which the
metadata is returned. Behind this name and concept are implementations
attached to branches of the type-hierarchy which then share the same
calculation method. At run-time, the correct implementation is looked
up by assigning particular descriptor implementations to the correct
Astrodata Type. The assignment is made within python dictionaries
stored in the configuration directories. Given an AstroData instance,
ad , to get the "gain" metadata for any supported datatype, you would
use the following code:

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

Primitives name a particular kind of dataset transformation, the idea
and name of the transformation. A "recipe" is just a list of such
"primitives". As with descriptors it is expected that the particular
instrument-specific steps required to perform the transformations can
be special to particular branches (or leaves) of the type hierarchy.
One approach to performing the intended transformation might likely
apply to just a single instrument or instrument-mode, while others
will apply more broadly. We want to write generalized algorithms as
often as possible due to the advantages of such, but also to
compartmentalize code which is notably different in approach even
though the concept of the transformation is the same.

Formalizing the transformation concepts allows us to refactor the
solutions due to unforeseen complications, new information and
instruments, and so on, without having to necessarily change recipes
that call these transformations. This helps configurations to be
expanded and improved while still providing a stable interface to the
user.

AstroData is intended to be useful in general python scripting, one
does not have to write code in the form of primitives. Such scripts
benefit from the type, descriptor and other built in data handling
features of AstroData. However, such scripts lend themselves to use in
a well-controlled automated system. They lack a consistent control and
parameter interface. Thus to be used in the astrodata automation
features such code is wrapped in primitives to give a consistent
input/output interface, and allow interaction with the control system.

The automation system is designed to support a range of automation, to
dataset by dataset automation for a pipeline processing data as it
comes from the telescope, through to a more "interactive" automation
where the user decides at what level to initiate automation.

The "primitive" functions are implemented as python generators, which
can "yield" control, "returning" the current context, and then later
be reentered. This ability allows communication and cooperative
control while a primitive executes, allowing the controlling system to
perform some services for the reduction (like retrieving calibrations
from a potentially remote source).

The astrodata package itself in no way enforces any rules about the
complication or nature of the transformations performed by the
primitive, such standards are meant to be part of a particular
configuration. In the Astrodata Configuration the general intention
has been that primitives represent transformations which are arguably
"scientifically meaningful". The name of the transformation should not
merely name operations which are performed, nor purely infrastructural
purposes, but rather the operation as a whole should relate to a
general idea about data processing.

For examples typical primitive names are "subtractSky" or
"biasCorrect" which have meaning in conversation about dataflow,
regardless of how they are performed. Arbitrarily complex differences
may require very different implementations, but so long as the step is
performed properly for the type processed, the differences are not
represented at the recipe level.

Within the primitive there is pure python code and significant
software engineering artifacts, but in the name of the primitives, and
thus in recipes, which are sequential lists of primitive names, only a
reference to the scientific concepts named exists. There are no
explicit conditionals or variables in these recipes. However, the
correct implementation for a given primitive is ensured to be run,
including if previous primitives have changed the type during their
transformation of the data. Thus there is implicit conditional
behavior based on Astrodata Type, and recipes are said to "adapt" to
the dataset type, dynamically as necessary as the data is processed.

For advanced users it may be of note to mention that primitives,
strictly speaking, transform a"Reduction Context" object, not
specifically or just the input datasets. This context contains
references to all objects and datasets which are part of the
reduction, part of which are the input files. For example, a primitive
to calculate "seeing quality" will not actually modify the dataset,
but it will modify the Reduction Context, reporting the calculated
statistic to the context.

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

Use of primitives instead of scripts for reduction processes has a
major side benefit besides enabling automation features of the
astrodata package. It promotes breaking reductions down into discrete
chunks, which is generally a good thing. Discussing how to break down
and name our classical reduction procedures into reusable recipes and
primitives has had the effect of clarifying our understanding of these
procedures. Steps with re-use potential in other recipes should be
contained as separate primitives, as should those which may be of
direct use to users that would want to be able to call them in
isolation of other behaviors. Thus, even if breaking down a recipe
intended for one specific instrument-mode, one discovers reusable
concepts that enrich our data flow language.

Essentially primitives have become our data flow language, as we
design primitives, we end up formalizing and implementing terms we use
to describe our data flow, and when a new primitive is defined, a new
term is created that we use to describe our data flow.

Note: We have performed the exercise of breaking down a set of pre-
existing scripts into recipes and primitives. It turned out relatively
easy to find where the discrete transitions occur in the scripts, and
identify them as primitives. These primitives were developed in a
separate recipe package from RECIPES_Gemini, added to the RECIPEPATH.
As a stand alone package for a particular purpose (Instrument
Monitoring) it was not as important to create idealized primitives for
this application as it is for the Gemini primitives which are intended
to be reusable and generally comprehensible. Instead of formal design,
these primitives had been abstracted from the ad hoc design of the
scripts. However, since the ad hoc source code is hidden within the
primitives, the recipe still is a good high level description of the
scripts original algorithmic shape, but not its specific
implementation where, generally, the more ad hoc and get-the-job-done
code will be found.

The result is the transformation begins possibility of graceful
refactoring; If the de facto methods that emerge in the recipe are
sound, then perhaps the pieces can be cleaned up and used as is. On
the other hand, if it is not optimal, the reusable parts of the
script, now in reusable primitive form, can be used in a new approach.

In the case of our instrument monitoring example case, the result is
functional and in use, and there was no serious need to change the
method seen in the recipe or the implementation of the primitives
themselves, in order to perform their specific job, and benefit from
the Gemini library of primitives. Also, several of the primitives
created prove of probably general interest (i.e. retrieving data from
the GSA automatically), and would be temptingly simple to generalize.


Recipes calling Recipes
```````````````````````

Recipes can in fact call recipes. Thus there are levels of these data
flow terms, starting with very high level concepts like "prepare",
made of lower level primitives such as "biasSubtract", and even lower
level primitives like "validateHeaders". The result is we tend to have
top level recipes which represent the most abstract view of the data,
and describe steps most data go through and act as generally
applicable instructions. At the lowest level we have primitives which
represent the most concrete steps we want want to consider as
"arguably scientific", or at least consider "data flow language"
rather than "pure python".

Ultimately, at the lowest level, within the primitives, is of course,
pure python. This code still can be written in a general or generic
way, allowing us to share code at high levels in the type hierarchy,
using features of AstroData such as descriptors which also hide
incidental differences between instrument-modes.


AstroData Lexicon
~~~~~~~~~~~~~~~~~

These three concepts, starting with Astrodata Type, and then Astrodata
Descriptors and Primitives to be assigned to these types, results in a
lexicon of terms about the data configured, e.g. in the case of
Gemini, as defined in astrodata_Gemini, our astrodata configuration
package. Type-specific behaviours are assigned to branches (or leaves)
of the type-tree hierarchies and will apply to those types below them
in the hierarchy unless overriden by a more-type-specific assignment
of the same sort. While I have described two general features which
require type-dependent implementations, the system is arranged such
that it is relatively easy to look up (and therefore assign) any
feature or property based on Astrodata Type in the same manner. Other
behaviours can and will also be assigned this way in the future, for
example the Astrodata Structures feature, currently a prototype, which
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
also by conventions contains the specific "ADCONFIG_xxx" and
"RECIPE_xxx sub-packages). Here is an example type tree graph for
GMOS, from the current Gemini classification library:


.. figure:: images_types/GMOS-tree-pd.png
    :scale: 90
    :figwidth: 5.4in
    :figclass: align-center
    
    GMOS AstroData Type Tree
    


This graph shows GMOS is a child type of GEMINI, and all other
displayed instrument-modes arranged as children of GMOS since they are
indeed modes of GMOS and "are" GMOS data. Any dataset which "is" a
type named at some particular node of the tree will also "be" every
parent type from that node to the top of the hierarchy (i.e. all GMOS
data is GEMINI data). The graph shows a descriptor calculator is
assigned to GMOS. The member of this Calculator class are the specific
descriptor functions which will calculate all high-level metadata for
all GMOS types. The graph also shows a set of primitives is assigned
to the GEMINI type, and another assigned to the GMOS_IMAGE type, thus
all types shown will run GEMINI, generic, primitives unless overridden
by a more specific primitive set. Thus as currently defined all modes
but GMOS_IMAGE have only generic primitives available. This is due to
work on primitives being ongoing, and eventually GMOS_SPECT will have
a primitive set. Note, that in a case such as this, the GMOS_IMAGE
primitive is the only one loaded for GMOS_IMAGE data, and the generic
primitives are thus not available (!). However, since the GEMINI
primitive set is class, the GMOS_IMAGE primitive set can and does
refer to it as a parent, allowing the GMOS_IMAGE primitive developer
to specifically inherit the more generic behavior not through the
Astrodata Type Tree, but using python OOP. This provides a great deal
of flexibility to share code in multiple ways as appropriate to a
particular case.

GEMINI primitives are generally just bookkeeping functions as few
transformations can be generalised across all Gemini datasets, though
some can. Thus lacking a more specific primitive set means,
essentially, that instrument is not fully supported, and the generic
transformations we've defined cannot yet be executed on those types of
dataset.

