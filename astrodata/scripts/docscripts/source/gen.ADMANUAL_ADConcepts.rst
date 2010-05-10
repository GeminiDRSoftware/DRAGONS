


Concepts
--------


Background
~~~~~~~~~~

The astrodata system is a product of a Gemini Astronomers request to
"handle MEFs better". Investigation showed that the MEF libraries were
sufficient for handling "MEFs" as such, and the real problem was the
MEFs are presented by fits libraries (e.g. pyfits) as lists of Header-
Data Units, aka "extensions", and relationships between the extensions
are not recognised. AstroData on the other hand is intended to
recognise those connections and present the data in a MEF as a related
whole, for example interpreting the set as some particular dataset
type.

Ample header data exists in the datasets to perform this function
given a definition of how the types are characterized, and what
unified interface one wants to present. Given the configuration
AstroData will look at data in a format-specific ways when needed, but
maintain a consistent interface for the AstroData user.

To use astrodata you will develop a lexicon of terms or use an
existing set of definitions, e.g. Gemini's. In the current system
there are three types of terms to be concerned wits:


+ dataset classification names, aka **AstroDataTypes**
+ high level metadata names, aka **Descriptors**
+ scientifically meaningful discrete dataset transformation names, aka
  **Primitives**


Each of these have associated actions:


+ AstroDataType: check a dataset for adherence to a criteria of
  classification, generally by checking PHU key-value pairs.
+ Descriptors: calculate of a type of high-level metadata in a
  particular variable type, for a particular AstroDataType, generally
  from one or more low-level metadata in the dataset headers.
+ Primitives: perform a dataset discrete, transformation for a
  particular AstroDatatype.



AstroDataType
~~~~~~~~~~~~~

Lack of a central system meant that scripts and tasks have to make
extended checks of their own on the header data of the datasets they
are manipulating. Often these are merely to verify the right type of
data is being worked on. Thus AstroData includes a classification
system where types are defined in configuration packages and can be
checked in a single line once the AstroData instance is created:

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


The `isType(..)` function on lines 5 and 8 above are examples of one
line type checking, the one line check replaces a larger set of phu
checks which determine the instrument and mode, and with astrodata are
centralized in AstroDataType Library. The advantage in addition to
saving specific header checks, both the verbosity and the need to
remain familiar with arbitrary specific data is the centralization of
the checks. Thus if we need to enhance the check, handle special cases
sch as Instrument Upgrades, or otherwise rework our classification
heuristics, we can do this in a shared configuration.


AstroData Descriptors
~~~~~~~~~~~~~~~~~~~~~

It goes without saying that our scientific data requires copious
metadata to track significant information about the information. And
the `MEF </gdpsgwiki/index.php/MEF>`__ file structure supports, as all
must, such metadata. At first we hoped a table driven solution would
work such that a user would make a request based on a standardized
common key, and the actual header key name would be looked up and the
information retrieved.

The problem with this approach, upon inspection, is that our desired
and expected metadata is sometime distributed across multiple headers
key/value pairs, and also different header units. For a simple
example, the situation with the "filtername" metadata is that while
most instruments name their filter wheels FILTER1, FILTER 2, and so
on, there are a different number of filters in any given instrument,
and no requirement they be named this way (it's merely a natural
convention). More complicated still is the situation we face when an
instrument has been upgraded, requiring different lookup tables to
make a calculation, such as with GMOS and the "gain" descriptor. In
this case the calculation must be conditional on date, and also it
looks at multiple headers and as is the case with many such
calculations uses an adjustable lookup table.

To address this we invented descriptors to represent high level
metadata so that the AstroData user given AstroData instance ad simply
types::

.. code-block:: python
    :linenos:

     gain = ad.gain()


Because descriptors are assigned to AstroData Types, this ensure than
any date will have the correct tables used for GMOS, and indeed, that
the line will work for any supported datatype. Current ADCONFIG_Gemini
has descriptors for all Gemini instruments. See "Gemini AstroData Type
Reference" (`http://www.gemini.edu/INSERTFINALGATREFURLHERE
<http://www.gemini.edu/INSERTFINALGATREFURLHERE>`__ for a list of
available descriptors for Gemini data.

Thus descriptors are implemented as python functions. The model for
descriptors support the most general case of arbitrary calculation of
"high-level" metadata, "descriptors" via groups of python functions
(bundled as a class) assigned to particular types, using the metadata
known to be present in the given subclass of datasets. When
descriptors can be written broadly, as for all GEMINI datasets, then
they can be assigned to a general type. When the high-level metadata
does in fact correspond to a single key-value pair of low level
metadata, the infrastructure can look the value up and the descriptor
calculator class can indicate this merely by not implementing that
descriptor or calling the infrastructure standard key lookup function.


Recipe System Primitives
~~~~~~~~~~~~~~~~~~~~~~~~

Primitives as a term name a certain kind of data transformation, and
as with descriptors it is expected that the particular, instrument-
specific steps required to complete the transformations are
potentially unique to arbitrary degree, that is, perhaps just a single
instrument mode or telescope configuration. One might will want to
write general algorithms as often as possible, but need arbitrary
granularity to include special steps.

While writing orginary programs and scripts is sufficent and doesn't
impost inherent organization difficulties doing manual header checks
(AstroData Type) and normalizing metadata (Descriptors) do, they are
impossible to control and organized by an automated system. We know we
wanted and had to support automation in the package, and that the
package would be deployed within pipelines, possbily varied, and would
have to be well controllable.

The primitive allows this. By abstracting transformations into well
defined steps and once again assigning sets of related steps to
AstroData Types, we give the infrastructure the power to initiate and
control reductions. Primitive Sets are associated as members of the
same class, and are python generators so they can yield to the control
system as appropriate.

The granularity of the transformations is of course in no way enforced
by the system, but at Gemini the intention is that the names of
primitves be arguably "scientifically meaningful" so one creates
primitves such as "subtractSky" and "biasCorrect", and specific pixel
manipulation is done within the primitives. This allows us to build a
concept of a "recipe" also as arguably "scientifically meaningful" as
merely a list of primitives executed in order. There are no explicite
conditionals, but there is an implicit conditional insofar as the
actual primitive implementation called will depend on the
classification of the data at that step in the recipe, after
processing by previous steps.

Take the following recipe:

.. code-block:: python
    :linenos:

    
    prepare
    overscanSub    
    overscanTrim
    biasSub
    flatField
    findshiftsAndCombine


The above is a generic recipe, given primitive sets for GMOS_IMAGE,
NIRI_IMAGE, etc, when the recipe system executes a line such as
biasSub, it will execute the "biasSub" member implemented for that
type. Thus, if biasSub can be implemented for both types, while
prepare requires specific implementations, then those primitives which
have unique implementation will be called for the appropriate dataset
type, and those which are generic will also be applied. The two
implementation of Primitive sets could share a biasSub implementation
through, say, an IMAGE related primitive set for IMAGE-generic
operations.

Use of primitives also promotes breaking code down into discrate
chunks, specifically those in which identified types of data can be
conveyed to a subsequent primitives. We have found this helps
discussion about recipes maintain this "arguably scientific" centered
argument, while software engineering issues are isolated to
primitives. Use of AstroData within the primitives further defers the
need to worry about incidental differences between data, and focus on
the steps needed in principle, based on the scientific purpose and
abilities of the given instrument in the given mode and telescope
configuration.

