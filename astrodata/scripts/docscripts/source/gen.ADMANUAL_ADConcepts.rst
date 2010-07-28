


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
are not recognised. Further, while FITS allows copious meta-data, it
standardises only a subset of what are for Gemini, universal
properties for our datasets. This leads to the information being
available in all datasets, but to be discovered differently, e.g. via
different meta-data key names.

AstroData on the other hand is intended to recognise the connections
MEF does not directly encode. AstroData begins by detecting the type
of dataset then using this knowledge, applies the particular methods
to obtain the general behavior for that particular dataset's
instrument-mode. The data in a MEF is not incidentally co-located in
the MEF, it's presumed to be part of a bundle of data associated with
a potentially complex observation. Thus, to first order, these dataset
types therefore map to instrument-modes and all the associated
information needed to make full use of the science data.

Ample header data exists in the datasets to perform this function
given a definition of how a type is detected once characterised. Given
such definitions in an AstroData configuration (i.e. in
astrodata_Gemini/ADCONFIG_Gemini), AstroData the class will look at
data in format-specific ways when needed, but maintain a consistent
interface for the AstroData user. When code can be shared among types,
then the behavior is assigned to more general types those modes share
(e.g. "GMOS_IMAGE" and "GMOS_IFU" share the "GMOS" type).

To use astrodata the package you will develop or use a lexicon of
terms which describe your data and the processing you want to do. In
the current system there are three types of terms to be concerned
with:


+ dataset classification names, aka **AstroDataTypes**
+ high level metadata names, aka **Descriptors**
+ scientifically meaningful discrete dataset transformation names, aka
  **Primitives**


Each of these have associated actions:


+ Astrodata Type: checks a dataset for adherence to a classification
  criteria, generally by checking PHU key-value pairs.
+ Astrodata Descriptors: calculates a particular, named, bit of high-
  level metadata for a particular Astrodata Type.
+ Primitives: performs a standard, named, dataset transformation on a
  dataset of a particular Astrodata Type.



Astrodata Type
~~~~~~~~~~~~~~

Lack of a central system for type detection in our legacy package
meant that scripts and tasks had to make extended checks on the header
data in the datasets they manipulate. Often these checks merely verify
that the right type of data is being worked on, but are multi-line and
can vary from script to script. Thus, how a dataset is recognised as
belonging to a particular instrument-mode is not consistent.
Astrodata's classification system allows defining dataset
classifications in configuration packages, centralizing the meaning of
a particular mode and the official heuristics for detecting it:

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
centralised within AstroData, which provides an interface for
accessing type information. Users benefit in a forward-compatible way
from any future improvements to the named type, and also gain
additional sophistication such as type-hierarchy relationships which
are simply not present in the legacy approach.


Astrodata Descriptors
~~~~~~~~~~~~~~~~~~~~~

It goes without saying that our scientific datasets require copious
meta-data. Significant amounts of "information about the information"
is present regarding an observation. The `MEF
</gdpsgwiki/index.php/MEF>`__ file structure supports such meta-data,
of course in the extension and primary headers. Thus there was an
assumptive notion that the information we want is in the headers, and
it is merely named differently. So at first we hoped the differences
in storage of meta-data could be eliminated with a table driven
solution, as it has been for isolated tools and limited subsets of
meta-data. That is, we would provide universal meta-data names which
would be mapped to the actual header key name based on the dataset
type. The high level names can be thought of as relating to high level
meta-data which is stored in the low-level meta-data as it actually
appears in the dataset as stored.

The problem with this approach turned out to be that the desired and
expected meta-data is sometime distributed across multiple header
key/value pairs, can require lookup tables keyed by arbitrary factors
(such as the date of the exposure), and that the information can be
distributed in different header units. For a simple example, the
situation with the "filter_name" high level meta-data is that while
most instruments name their filter wheels FILTER1, FILTER2, and so on,
there are, one, a different number of filters in any given instrument,
and two, no actual requirement they be named this way (it's merely a
natural convention). In some instruments filters share wheels with
grisms, in some they do not. These issues cannot be sorted in general
by data structure based approach of acceptable simplicity, and require
the flexibility of a function to perform arbitrary manipulations and
calculation. Thus the high-level named meta-data is implemented by
functions associated to datasets based on their AstroDataType.

We name the high-level metadata "Astrodata Descriptors", or just
"descriptors". The descriptor is a name and description of the data
represented (description and the units used). Behind this name are
implementations attached to branches of the type-hierarchy that can
share the same calculation methods, and the correct implementation is
looked up by assigning these descriptor implementation to the correct
AstroDataType. Given AstroData instance ad , to get the "gain" meta-
data you would use the following code::

.. code-block:: python
    :linenos:

     gain = ad.gain()


Because descriptors are assigned to AstroData Types that line will
work for any supported datatype, taking into account any type-specific
peculiarity. The current ADCONFIG_Gemini configuration implementation
has descriptors for all Gemini instruments. See "Gemini AstroData Type
Reference" (`http://www.gemini.edu/INSERTFINALGATREFURLHERE
<http://www.gemini.edu/INSERTFINALGATREFURLHERE>`__) for a list of
available descriptors for Gemini data.

Descriptors are implemented and called as functions, and their
implementation are bundled together in a common class which will be
assigned to the appropriate type in an index file in the ADCONFIG
configuration directories. When descriptors can be written broadly,
i.e. a single function can return the value for all GEMINI datasets,
these descriptor functions can be assigned to the more a bundle of
descriptors intended for this more general type, and the
implementations will be shared by the types below it in the hierarchy
unless overriden. Also, when the high-level meta-data does in fact
correspond directly to a single key-value pair within the low level
meta-data, the infrastructure will look the value up using the
standard key for that dataset type, and there is no need to write a
function to calculate the value, though the user syntax is still
presented as a function call.


Recipe System Primitives
~~~~~~~~~~~~~~~~~~~~~~~~

Primitives name a certain kind of dataset transformation. As with
descriptors it is expected that the particular instrument-specific
steps required to complete the transformations are arbitrarily unique
to particular branches (or leaves) of the type hierarchy. That is, one
approach to performing the intended transformation may likely apply to
just a single instrument or instrument-mode, while others will apply
more broadly. We want to write generalised algorithms as often as
possible due to the advantages of supporting one cod ebase rather than
multiple, different but similar, code bases. The Astrodata system
allows arbitrary granularity to assign transformations, and other
features, to any type, and also to define types based on whatever
information you desire (though they should be quickly completed
tests). This approach provides access to "refactoring paths" in the
type tree, so that decisions about implementing generic or mode-
specific algorithms can be continually re-evaluated and revised as the
package matures.

Writing regular python scripts using only the type and descriptor
features of AstroData, rather than primitives, is acceptable and will
benefit the script by providing normalisation in terms of type
checking and meta-data access, and still provide other AstroData
features. However, such scripts are impossible to control as part of a
well-controlled automated system, so to be included in such systems,
for example used in pipeline data reduction, the code must be written
as a primitive (or recipe which is just a sequential list of
primitives).

By abstracting transformations into well defined steps, and allowing
the configuration to define layers of recipes and primitives (recipes
and primitives can both also call recipes and primitives as part of
their operation), we expose to the user a range of interaction levels,
at the highest we imagine non-interactive pipeline processing, with
everything needed arranged such that it can be looked up depending on
the dataset it is fed. At the lowest level, concise, atomic,
operations are provided by the lowest level of primitives. By
assigning sets of transformations based on astrotype AstroData Type
inside a bundling class, the PrimitiveSet, we provide the
infrastructure with enough information to initiate and control
reductions.

The "primitive" functions are implemented as python generators, which
can "yield" control, and then be reentered. This ability allows
communication and cooperative control at a granularity more fine than
the transformation as a whole. They take a single argument, the
ReductionContext, in which all input and output are stored, allowing
the infrastructure to observe changes in this structure and perform
validation, history and other services.

The astrodata package itself in no way enforces any rules about the
"size" or nature of the transformations performed by the dataset, but
the intention is that the top level primitives carry names and
represent transformations which are arguably "scientifically
meaningful". The name of the transformation should not merely name
operations which are performed, but the operation as a whole should
relate to a general idea about data processing. For example, one
creates primitives such as "subtractSky" and "biasCorrect" which have
meaning in conversation about dataflow regardless of the different
methodologies a particular instrument in a particular mode might
employ. This allows us to build a concept of a "recipe", which is
simply a sequential list of primitives, and argue it too is
"scientifically meaningful".

There are no explicit conditionals in these recipes, but there is
implicit conditional behaviour insofar as the actual primitive
implementation called at a given step will depend on the
classification of the data given as input, as determined during
execution. Thus recipes are implicitly conditional on AstroData Type.

For advanced users it may be of note to mention that primitives,
strictly speaking, transform the ReductionContext, and not implicitly
the input datasets it maintains for the reduction. For example, a
primitive to calculate seeing quality will not modify the dataset, but
it will modify the ReductionContext, reporting the statistic. Below is
a sample recipe, in use in our development environment for testing.


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

Use of primitives has the benefit of promoting breaking reductions
down into discrete chunks which is generally a good thing. Discussing
how to break down and name our reductions as recipes and primitives
leads to what you might call a natural granularity. Steps with re-use
potential in other recipes should be contained as separate primitives,
as should those which may be of direct use to users that would want to
be able to call them in isolation of other behaviours. Also, software
engineering concerns advocate isolating code which is different in
character in separate units, isolating functionality, which in this
case means separate primitives. Thus the decision about what
transformation is atomic, and if it is truly atomic or calls other
nested recipes or primitives, is driven by a mix of design and user
interface concerns.

We have performed the exercise of breaking down a set of scripts into
primitives and it is relatively easy to find where these transitions
occur in the scripts. This is because reduction scripts generally
actually do have to occur sequentially in the script, either if
written with planning and foresight or even if written ad hoc on the
fly, just as they do in the recipe. It is simply more natural to, for
example, to fix corrupt headers at one point in ones script, and them
move onto some other check or fix, than it is to intersperse lines of
each. Thus such scripts tend to naturally have implicit recipes.
Refactoring them as recipes and primitives simply makes this implicit
sequential structure explicit. It also seems to be the case that most
true branching flow control in reduction scripts keys on the type of
data, which is addressed by the implicit type-specificity of primitive
execution. This allows the recipe to be presented sequentially, which
conceptually from the data flow standpoint, it is, but implemented in
a type-dependant ways, which is necessary for more concrete realities
like how the pixels and meta-data is laid on in that particular
instrument and mode.

The result is we tend to have top level primitives or recipes which
represent the most abstract view of the data, steps all data go
through. At the lowest level we have primitives which represent the
most concrete steps we want want to consider as "arguably scientific",
or at least consider "data flow language" rather than "python". Within
primitives themselves is always python code, which by definition is
the most low level and particular, below this "data flow" language,
refering to concrete facts, like the interpreting the a four
dimensional dataset produced by a particular instrument mode.

Though these then are the smallest transformations, and from a recipe-
execution perspective the "atomic" steps of transformation, within the
primitive there is another stack of normalized interface helping the
coder, now in python, to write generalized code using the type and
high-level metadata features of AstroData, as well as its other useful
dataset-handling features.


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
    :scale: 90%
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

