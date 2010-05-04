


Concepts
--------


Background
~~~~~~~~~~

The astrodata system is a product of a Gemini Astronomers request to
"handle MEFs better". Investigation showed that the MEF libraries were
sufficient for handling "MEFs" as such, and the real problem was the
MEFs are inherently lists of Header-Data Units, aka (extensions), and
relationships between the extensions are not recognized by the system.

Ample header data to find related extensions and maintain
relationship, but the FITS library does not track these, since indeed
it does not need to. Thus we needed a higher level abstraction to
visualize our MEF files as single datasets which may consist of
multiple extensions.

To use astrodata you will develop a lexicon of datatypes (or use an
existing set of definitions, e.g. Gemini's) with the following types
of terms:


+ dataset classifications, aka AstroDataTypes
+ high level metadata names, aka Descriptors
+ dataset transformation names, aka Primitives


Each of these have associated actions:


+ AstroDataTypes' action: checking for adherence to a criteria of
  classification, generally by checking PHU values.
+ Descriptors' actions: calculation of a type of metadata for a
  particular AstroDataType, generally from metadata (possibly
  distributed through multiple extension headers and requiring dataset-
  specific algorithms).
+ Primtives' actions: performing the dataset transformation for a
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


The `isType(..)` function is on lines 5 and 8 above are examples, the
one line check replaces a larger set of phu checks which determine the
instrument and mode, and with astrodata are centralized in
AstroDataType Library.


AstroData Descriptors
~~~~~~~~~~~~~~~~~~~~~

The problem we face in metadata is...

