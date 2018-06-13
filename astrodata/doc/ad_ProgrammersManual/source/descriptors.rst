.. descriptors.rst

.. _ad_descriptors:

***********
Descriptors
***********

The descriptors are just regular methods that translate metadata from the raw
storage (eg. cards from FITS headers) to values useful for the user,
potentially doing some processing in between. They exist to:

* Abstract the actual organization of the metadata. Eg. ``AstroDataGemini``
  takes the detector gain from a keyword in the FITS PHU, where
  ``AstroDataNiri`` overrides this to provide a hard-coded value.

  Then again, ``AstroDataGmos`` uses the observation date to choose one out of
  several lookup tables that are then indexed by read speed setting, gain
  setting, and amplifier name… and does this per each individual detector in
  the mosaic. No only that, the read speed and gain settings, and the amplifier
  are obtained using descriptors, too.
* Provide a common interface to a set of instruments. This simplifies user
  training (no need to learn a different API for each instrument), and the
  facilitates the reuse of code for pipelines, etc.
* Also, for simple keyword → value mappings, they provide a more meaningful and
  readable name, typically.

Descriptors **should** be decorated using
``astrodata.core.astro_data_descriptor``. The only function of this decorator
is to ensure that the descriptor is marked as such: it does not alter its input
or output in any way. This lets the user to explore the API of an AstroData
object by calling ``astrodata.core.descriptor_list``.

Descriptors **can** be decorated with ``astrodata.core.returns_list``. This is
useful with descriptors that are supposed to return lists of values. This is a
common case for decorators on instances that may have more than one extension.
AstroData complicates a little bit the logic for descriptors when there is
slicing, and ``returns_list`` may help in those cases, altering the output as
needed. The algorithm for the decorator is the following:

* If the astrodata object has been sliced as a single object, a single value
  will be returned (as opposed to a list). If the descriptor returns a list,
  the decorator will return only its first element.
* Else:

  * If the descriptor returns a list, it must match the number of extensions,
    which is always equal to ``len(self)``. If the lengths do not match, an
    exception will be raised.
  * Else, the decorator will consider the result a single value, and will
    return a list of length ``len(self)``, where each element is a copy of the
    one returned by the original descriptor.
