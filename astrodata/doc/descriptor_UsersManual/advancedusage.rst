.. advancedusage:

.. _Advanced_Descriptor_Usage:

*************************
Advanced Descriptor Usage
*************************

DescriptorValue
---------------

When a descriptor is called, what is actually returned is a ``DescriptorValue``
(``DV``) object::

  >>> from astrodata import AstroData
  # Load the fits file into AstroData
  >>> ad = AstroData("N20091027S0137.fits")
  # Get the airmass value using the airmass descriptor
  >>> airmass = ad.airmass()
  >>> print airmass
  1.327
  >>> print type(airmass)
  <class 'astrodata.Descriptors.DescriptorValue'>

Each descriptor has a default Python type defined::

  >>> print ad.airmass().pytype
  <type 'float'>

If any of the following operations are applied to the ``DV`` object, the ``DV``
object is automatically cast to the default Python type for that descriptor::

  +   -   *   /   //   %   **   <<   >>   ^   <   <=   >   >=   ==

For example::

  >>> print type(airmass*1.0)
  <type 'float'>

The value of the descriptor can be retrieved with the default Python type by
using ``as_pytype()``::

  >>> elevation = ad.elevation().as_pytype()
  >>> print elevation
  48.6889222222
  >>> print type(elevation)
  <type 'float'>

The ``as_pytype()`` member function of the ``DV`` object should only be 
required when the ``DV`` object can not be automatically cast to it's default 
Python type. For example, when it is necessary to use the actual value of a
descriptor (e.g., string, float, etc.) rather than a ``DV`` object as a key of
a Python dictionary, the ``DV`` object can be cast to it's default Python type
using ``as_pytype()``::

  >>> my_key = ad.gain_setting.as_pytype()
  >>> my_value = my_dict[my_key]

If an alternative Python type is required for whatever reason, the ``DV`` 
object can be cast to the appropriate Python type as follows::

  >>> elevation_as_int = int(ad.elevation())
  >>> print elevation_as_int
  48
  >>> print type(elevation_as_int)
  <type 'int'>

When using operations with a ``DV`` object and a numpy object, care must be
taken. Consider the following cases::

  >>> ad[0].data / ad.gain()
  >>> ad[0].data / ad.gain().as_pytype()
  >>> ad[0].data / 12.345

All the above commands return the same result (assuming that ad.gain() =
12.345). However, the first command is extremely slow while the second and
third commands are fast. In the first case, since both operands have overloaded
operators, the operator from the operand on the left will be used. For some
reason, the ``__div__`` operator from the numpy object loops over each pixel in
the numpy object and uses the ``DV`` object as an argument, which is very time
consuming. Therefore, the ``DV`` object should be cast to an appropriate
Python type before using it in an operation with a numpy object. 

In the case where a descriptor returns multiple values (one for each pixel data
extension), a Python dictionary is used to store the values, where the key of
the dictionary is the ("``EXTNAME``", ``EXTVER``) tuple::

  >>> print ad.gain()
  {('SCI', 2): 2.3370000000000002, ('SCI', 1): 2.1000000000000001, 
  ('SCI', 3): 2.2999999999999998}

For those descriptors that describe a concept applying specifically to the
pixel data extensions within a dataset but has the same value for each pixel
data extension will "act" as a single value that has a Python type defined by
the default Python type of that descriptor::

  >>> xbin = ad.detector_x_bin()
  >>> print xbin
  2
  >>> print type(xbin)
  <class 'astrodata.Descriptors.DescriptorValue'>
  >>> print xbin.pytype
  <type 'int'>

If the original value of the descriptor is required, it can be retrieved by
using ``get_value()``::

  >>> xbin = ad.detector_x_bin().get_value()
  >>> print xbin
  {('SCI', 2): 2, ('SCI', 1): 2, ('SCI', 3): 2}
  >>> print type(xbin)
  <type 'dict'>
  >>> print xbin[("SCI", 1)]
  2

DescriptorUnits
---------------

The DescriptorUnits (``DU``) object provides a way to access and update the
units for a given descriptor. This feature is not yet implemented, but
development is ongoing.
