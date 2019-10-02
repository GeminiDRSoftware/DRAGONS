.. headers.rst

.. _headers:

********************
Metadata and Headers
********************

**Try it yourself**

Download the data package (:ref:`datapkg`) if you wish to follow along and run the
examples.  Then ::

    $ cd <path>/ad_usermanual/playground
    $ python

You need to import Astrodata and the Gemini instrument configuration package.

::

    >>> import astrodata
    >>> import gemini_instruments

Astrodata Descriptors
=====================

We show in this chapter how to use the Astrodata Descriptors.  But first
let's explain what they are.

Astrodata Descriptors provide a "header-to-concept" mapping that allows the
user to access header information from a unique interface, regardless of
which instrument the dataset is from.  Like for the Astrodata Tags, the
mapping is coded in a configuration package separate from core Astrodata.
For Gemini instruments, that package is named ``gemini_instruments``.

For example, if the user is interested to know the effective filter used
for an observation, normally one needs to know which specific keyword or
set of keywords to look at for that instrument.  However, once the concept
of "filter" is coded as a Descriptor, the user only needs to call the
``filter_name()`` descriptor to retrieve the information.

The Descriptors are closely associated with the Astrodata Tags.  In fact,
they are implemented in the same ``AstroData`` class as the tags.  Once
the specific ``AstroData`` class is selected (upon opening the file), all
the tags and descriptors for that class are defined.  For example, all the
descriptor functions of GMOS data, ie. the functions that map a descriptor
concept to the actual header content, are defined in the ``AstroDataGmos``
class.

This is all completely transparent to the user.  One simply opens the data
file and all the descriptors are ready to be used.

.. note::
    Of course if the Descriptors have not be implemented for that specific
    data, they will not work.  They should all be defined for Gemini data.
    For other sources, the headers can be accessed directly, one keyword at
    a time.  This type of access is discussed below.  This is also useful
    when the information needed is not associated with one of the standard
    descriptors.

To get the list of descriptors available for an ``AstroData`` object::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    >>> ad.descriptors
    ('airmass', 'amp_read_area', 'ao_seeing', ...
      ...)

Most Descriptor names are readily understood, but one can get a short
description of what the Descriptor refers to by calling the Python help
function.  For example::

    >>> help(ad.airmass)
    >>> help(ad.filter_name)

The full list of standard descriptors is available in the Appendix
:ref:`descriptors`.

Accessing Metadata
==================

Accessing Metadata with Descriptors
-----------------------------------
Whenever possible the Descriptors should be used to get information from
headers.  This allows for maximum re-usability of the code as it will then
work on any datasets with an ``AstroData`` class.

Here are a few examples using Descriptors::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')

    >>> #--- print a value
    >>> print('The airmass is : ', ad.airmass())

    >>> #--- use a value to control the flow
    >>> if ad.exposure_time() < 240.:
    ...     print('This is a short exposure.')
    ... else:
    ...     print('This is a long exposure.')
    ...

    >>> #--- multiply all extensions by their respective gain
    >>> for ext, gain in zip(ad, ad.gain()):
    ...     ext *= gain
    ...

    >>> #--- do arithmetics
    >>> fwhm_pixel = 3.5
    >>> fwhm_arcsec = fwhm_pixel * ad.pixel_scale()

The return values for Descriptors depend on the nature of the information
being requested and the number of extensions in the ``AstroData`` object.
When the value has words, it will be string, if it is a number
it will be a float or an integer.
The dataset used in this section has 4 extensions.  When the descriptor
value can be different for each extension, the descriptor will return a
Python list.

::

    >>> ad.airmass()
    1.089
    >>> ad.gain()
    [2.03, 1.97, 1.96, 2.01]
    >>> ad.filter_name()
    'open1-6&g_G0301'

Some descriptors accept arguments.  For example::

    >>> ad.filter_name(pretty=True)
    'g'

A full list of standard descriptors is available in the Appendix
:ref:`descriptors`.


Accessing Metadata Directly
---------------------------
Not all header content is mapped to Descriptors, nor should it.  Direct access
is available for header content falling outside the scope of the descriptors.

One important thing to keep in mind is that the PHU (Primary Header Unit) and
the extension headers are accessed slightly differently.  The attribute
``phu`` needs to used for the PHU, and ``hdr`` for the extension headers.

Here are some examples of direct header access::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')

    >>> #--- Get keyword value from teh PHU
    >>> ad.phu['AOFOLD']
    'park-pos.'

    >>> #--- Get keyword value from a specific extension
    >>> ad[0].hdr['CRPIX1']
    511.862999160781

    >>> #--- Get keyword value from all the extension in one call.
    >>> ad.hdr['CRPIX1']
    [511.862999160781, 287.862999160781, -0.137000839218696, -224.137000839219]



Whole Headers
-------------
Entire headers can be retrieved as ``fits`` ``Header`` objects::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')
    >>> type(ad.phu)
    <class 'astropy.io.fits.header.Header'>
    >>> type(ad[0].hdr)
    <class 'astropy.io.fits.header.Header'>

In interactive mode, it is possible to print the headers on the screen as
follow ::

    >>> ad.phu
    SIMPLE  =                    T / file does conform to FITS standard
    BITPIX  =                   16 / number of bits per data pixel
    NAXIS   =                    0 / number of data axes
    ....

    >>> ad[0].hdr
    XTENSION= 'IMAGE   '           / IMAGE extension
    BITPIX  =                   16 / number of bits per data pixel
    NAXIS   =                    2 / number of data axes
    ....



Updating, Adding and Deleting Metadata
======================================
Header cards can be updated, added to, or deleted from the headers.  The PHU
and the extensions headers are again accessed in a mostly identical way
with ``phu`` and ``hdr``, respectively.

::

    >>> ad = astrodata.open('../playdata/N20170609S0154.fits')

Add and update a keyword, without and with comment::

    >>> ad.phu['NEWKEY'] = 50.
    >>> ad.phu['NEWKEY'] = (30., 'Updated PHU keyword')

    >>> ad[0].hdr['NEWKEY'] = 50.
    >>> ad[0].hdr['NEWKEY'] = (30., 'Updated extension keyword')

Delete a keyword::

    >>> del ad.phu['NEWKEY']
    >>> del ad[0].hdr['NEWKEY']



Adding Descriptors [Advanced Topic]
===================================
For proper and complete instructions on how to create Astrodata Descriptors,
the reader is invited to refer to the Astrodata Programmer Manual.  Here we
provide a simple introduction that might help some readers better understand
Astrodata Descriptors, or serve as a quick reference for those who have
written Astrodata Descriptor in the past but need a little refresher.

The Astrodata Descriptors are defined in an ``AstroData`` class.  The
``AstroData`` class specific to an instrument is located in a separate
package, not in ``astrodata``.  For example, for Gemini instruments, all the
various ``AstroData`` classes are contained in the ``gemini_instruments``
package.

An Astrodata Descriptor is a function within the instrument's ``AstroData``
class.  The descriptor function is distinguished from normal functions by
applying the ``@astro_data_descriptor`` decorator to it.  The descriptor
function returns the value(s) using a Python type, ``int``, ``float``,
``string``, ``list``; it depends on the value being returned.  There is no
special "descriptor" type.

Here is an example of code defining a descriptor::

    class AstroDataGmos(AstroDataGemini):
        ...
        @astro_data_descriptor
        def detector_x_bin(self):
            def _get_xbin(b):
                try:
                    return int(b.split()[0])
                except (AttributeError, ValueError):
                    return None

            binning = self.hdr.get('CCDSUM')
            if self.is_single:
                return _get_xbin(binning)
            else:
                xbin_list = [_get_xbin(b) for b in binning]
                # Check list is single-valued
                return xbin_list[0] if xbin_list == xbin_list[::-1] else None

This descriptor returns the X-axis binning as a integer when called on a
single extension, or an object with only one extension, for example after the
GMOS CCDs have been mosaiced.   If there are more than one extensions, it
will return a Python list or an integer if the binning is the same for all
the extensions.

Gemini has defined a standard list of descriptors that should be defined
one way or another for each instrument to ensure the re-usability of our
algorithms.  That list is provided in the Appendix :ref:`descriptors`.

For more information on adding to Astrodata, see the Astrodata Programmer
Manual.
