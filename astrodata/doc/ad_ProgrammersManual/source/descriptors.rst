.. descriptors.rst

.. _ad_descriptors:

***********
Descriptors
***********

Descriptors are just regular methods that translate metadata from the raw
storage (eg. cards from FITS headers) to values useful for the user,
potentially doing some processing in between. They exist to:

* Abstract the actual organization of the metadata. Eg. ``AstroDataGemini``
  takes the detector gain from a keyword in the FITS PHU, where
  ``AstroDataNiri`` overrides this to provide a hard-coded value.

  More complex implementations also exist. In order to determine the gain
  of a GMOS observations, ``AstroDataGmos`` uses the observation date
  (provided by a descriptor) to select a particular lookup table, and
  then uses the values of other descriptors to select the correct entry
  in the table.
* Provide a common interface to a set of instruments. This simplifies user
  training (no need to learn a different API for each instrument), and
  facilitates the reuse of code for pipelines, etc.
* Also, since FITS header keywords are limited to 8 characters, for simple
  keyword → value mappings, they provide a more meaningful and readable name.

Descriptors **should** be decorated using
``astrodata.core.astro_data_descriptor``. The only function of this decorator
is to ensure that the descriptor is marked as such: it does not alter its input
or output in any way. This lets the user to explore the API of an AstroData
object via the ``descriptors`` property.

Descriptors **can** be decorated with ``astrodata.core.returns_list`` to
eliminate the need to code some logic. Some descriptors return single values,
while some return lists, one per extension. Typically, the former are
descriptors that refer to the entire observation (and, for MEF files,
are usually extracted from metadata in the PHU, such as ``airmass``), while
the latter are descriptors where different extensions might return different
values (and typically come from metadata in the individual HDUs, such as
``gain``). A list is returned even if there is only one extension in the
AstroData object, as this allows code to be written generically to
iterate over the ``AstroData`` object and the descriptor return, without
needing to know how many extensions there are. The ``returns_list``
decorator ensures that the descriptor returns an appropriate object
(value or list), using the following rules:

* If the ``AstroData`` object is not a single slice:

  * If the undecorated descriptor returns a list, an exception is raised
    if the list is not the same length as the number of extensions.
  * If the undecorated descriptor returns a single value, the decorator
    will turn this into a list of the correct length by copying this value.

* If the ``AstroData`` object is a single slice and the undecorated
  descriptor returns a list, only the first element is returned.

An example of the use of this decorator is the ``AstroDataNiri`` ``gain``
descriptor, which reads the value from a lookup table and simply returns it.
A single value is only appropriate if the AstroData object is singly-sliced
and the decorator ensures that a list is returned otherwise.