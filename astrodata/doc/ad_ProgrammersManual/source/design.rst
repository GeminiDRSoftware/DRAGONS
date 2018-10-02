.. design.rst

.. _design:

**************
General Design
**************

As astronomical instruments have become more complex, there
has been an increasing need for bespoke reduction packages and pipelines to
deal with the specific needs of each instrument. Despite this
complexity, many of the reduction steps can be very similar and the overall
effort could be reduced significantly by sharing code. In practice, however,
there are often issues regarding the manner in which the data are stored
internally. The purpose of AstroData is to provide a uniform interface to the data
and metadata, in a manner that is independent both of the specific instrument
and the way the data are stored on disk, thereby facilitating this code-sharing.
It is *not* a new astronomical data format.

One of the main features of AstroData is the use of *descriptors*, which
provide a level of abstraction between the metadata and the code accessing it.
Somebody using the AstroData interface who wishes to know the exposure time
of a particular astronomical observation represented by the ``AstroData`` object
``ad`` can simply write ``ad.exposure_time()`` without needing to concern
themselves about how that value is stored internally, for example, the name
of the FITS header keyword. These are discussed further in :ref:`ad_descriptors`.

AstroData also provides a clearer representation of the relationships
between different parts of the data produced from a single astronomical
observation. Modern astronomical instruments often contain multiple
detectors that are read out separately and the multi-extension FITS (MEF)
format used by many institutions, including Gemini Observatory, handles
the raw data well. In this format, each detector's data and metadata is
assigned to its own extension,
while there is also a separate extension (the Primary Header Unit,
or PHU) containing additional metadata that applies to the entire
observation. However, as the data are processed, more data and/or
metadata may be added whose relationship is obscured by the limitations
of the MEF format. One example is the creation and propagation of information
describing the quality and uncertainty of the scientific data: while
this was a feature of
Gemini IRAF\ [#iraf]_, the coding required to implement it was cumbersome
and AstroData uses the ``astropy.nddata.NDData`` class,
as discussed in :ref:`containers`. This makes the relationship between these
data much clearer, and AstroData creates a syntax that makes readily apparent the
roles of other data and metadata that may be created during the reduction
process.

An ``AstroData`` object therefore consists of one or more self-contained
"extensions" (data and metadata) plus additional data and metadata that is
relevant to all the extensions. In many data reduction processes, the same
operation will be performed on each extension (e.g., subtracting an overscan
region from a CCD frame) and an axiom of AstroData is that iterating over
the extensions produces AstroData "slices" which retain knowledge of the
top-level data and metadata. Since a slice has one (or more) extensions
plus this top-level (meta)data, it too is an ``AstroData`` object and,
specifically, an instance of the same subclass as its parent.


A final feature of AstroData is the implementation of very high-level metadata.
These data, called ``tags``, facilitate a key part of the Gemini data reduction
system, DRAGONS, by linking the astronomical data to the recipes
required to process them. They are explained in detail in :ref:`ad_tags` and the
Recipe System Programmers Manual\ [#rsprogman]_.

.. note::

   AstroData and DRAGONS have been developed for the reduction of data from
   Gemini Observatory, which produces data in the FITS format that is still the
   most widely-used format for astronomical data. In light of this, and the
   limited resources in the Science User Support Department, we have only
   *developed* support for FITS, even though the AstroData format is designed
   to be independent of the file format. In some cases, this has led to
   uncertainty and internal disagreement over where precisely to engage in
   abstraction and, should AstroData support a different file format, we
   may find alternative solutions that result in small, but possibly
   significant, changes to the API.


.. [#iraf] `<https://www.gemini.edu/sciops/data-and-results/processing-software/description>`_

.. [#rsprogman] PIPE-USER-108_RSProgManual