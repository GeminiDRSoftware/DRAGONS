# AstroData Informal Reference Guide

This cheat sheet is a temporary measure until proper documentation has
been written. It lists a number of common tasks, and how to achieve them
using AstroData.

## Importing the modules

To use AstroData you need to import the core modules:

    >>> import astrodata

This provides you access to file-loading factories, the AstroData class,
and basic FITS (AstroDataFits) processing capabilities. Opening Gemini
files in a meaningful way, though, requires registering a number of
derivative classes with the system. To do this, we:

    >>> import gemini_instruments

## File manipulation

New AstroData doesn't keep open hdulists. Instead, it reads the data and
keeps it in-memory, organized as NDData objects. This means that there's
no concept of opening a FITS file in "read-write" or "update" modes.

To load a file, use the `astrodata.open` factory:

    >>> ad = astrodata.open('/path/to/file.fits')

To write it back:

    >>> ad.write(clobber=True)
    >>> ad.write(filename='foobar.fits')

Both `filename` and `clobber` are optional. If an `AstroData` instance has
been created by opening a file (eg. with the factory), and you try to write
the file without specifying `filename`, the original path will be used (ie.
you'll be overwriting the file you loaded). `clobber` works the same way as
with PyFITS: if it's `False` (default), an exception will be rised when
trying to overwrite an existing file.

## Obtaining information from an instance

The classic `ad.info()` gives an overall view of the file. This example comes
from a processed image and you can see how objects that are associated to an
extension appear under the science frame, with no index.

    >>> ad.info()
	Tags: ACQUISITION GEMINI GMOS IMAGE NORTH OVERSCAN_SUBTRACTED OVERSCAN_TRIMMED
	    PREPARED PROCESSED_SCIENCE SIDEREAL

	Pixels Extensions
	Index  Content                  Type              Dimensions     Format
	[ 0]   science                  NDDataRef         (2304, 3108)   float32
		  .uncertainty          StdDevUncertainty (2304, 3108)   float32
		  .mask                 ndarray           (2304, 3108)   int16
		  .OBJCAT               Table             (10, 43)       n/a
		  .OBJMASK              NDDataRef         (2304, 3108)   int16

	Other Extensions
		       Type        Dimensions
	.REFCAT        Table       (8, 14)

We can also ask for the tags:

    >>> ad.tags
    set(['PROCESSED_SCIENCE', 'GEMINI', 'NORTH', 'SIDEREAL', 'OVERSCAN_TRIMMED', 'IMAGE', 'OVERSCAN_SUBTRACTED', 'GMOS', 'ACQUISITION', 'PREPARED'])

## Accessing extensions

When it comes to extensions, the `AstroData` objects behave as a sequence. We can
use `len`, for example:

    >>> len(ad)
    6

This returns the number of first-level `NDData` instances contained by our `ad`
object. If this comes from a raw file, the number is the same as of extensions.
For a processed file, this is the number of `SCI` frames which. Top level tables
as MDF are not included in the count either, as they're referred by name:

    >>> ad.REFCAT
    <Table length=8>
      Id    Cat_Id    RAJ2000    DEJ2000    umag  umag_err   gmag  gmag_err   rmag  rmag_err   imag  imag_err   zmag  zmag_err
    int32   str32     float64    float64  float32 float32  float32 float32  float32 float32  float32 float32  float32 float32
    ----- ---------- ---------- --------- ------- -------- ------- -------- ------- -------- ------- -------- ------- --------
        1 0020122537 273.287093 58.936429     nan      nan  14.603    0.035   14.23    0.057  14.093    0.028     nan      nan
        2 0020122537 273.144454  58.86838     nan      nan  14.073    0.014  13.727     0.08  13.516    0.029     nan      nan
    ...

You can iterate over an `AstroData` instance:

    for ext in ad:
        # do something with the extension

The order is always predictable: first extensions with `EXTVER`
(sorted by this field); then pixel images that had no `EXTVER`, which
are assigned one.

`AstroData` objects support the same kind of slicing than other
sequences. Eg.:

    ad[2]                   # Returns the 3rd extension in the list
    ad[:3]                  # Returns the first 3

## Accessing the headers

To get info on the PHU:

    >>> ad.phu.keywords
    set(['TIMESYS', 'P2APMDEC', 'EXTEND', 'SIMPLE', ...])
    >>> ad.phu.show()
    SIMPLE  =                    T / Fits standard
    BITPIX  =                   16 / Bits per pixel
    NAXIS   =                    0 / Number of axes
    EXTEND  =                    T / File may contain extensions
    ORIGIN  = 'NOAO-IRAF FITS Image Kernel July 2003' / FITS file originator
    DATE    = '2016-10-06T05:31:51' / Date FITS file was generated
    IRAF-TLM= '2016-10-06T05:32:31' / Time of last modification
    ...

    >>> 'TIMESYS' in ad.phu
    True

To get values from the PHU:

    # This may raise a KeyError
    >>> ad.phu.KEYWORD
    'foo'
    # The following won't raise a KeyError
    >>> ad.phu.get('EXISTSNO')
    None
    >>> ad.phu.get('EXISTSNO', 'default_value')
    'default_value'

To set values:

    >>> ad.phu.KEYWORD = 'bar'
    >>> ad.set('KEYWORD', 'bar')

To get/set comments:

    >>> ad.phu.get_comment('KEYWORD')
    '..........'
    >>> ad.phu.set_comment('KEYWORD', 'comment')

    # Convenience method to set value and comment at the same time:
    >>> ad.phu.set('KEYWORD', value='value', comment='a comment')

## Accessing the data

NB: of the following, whenever something refers to the extensions, it will
    always return a *list*... except if the object has been sliced with a
    single integer: in that case we return a single value

If I want to access the underlying `NDData` instance that contains the
extension and everything related to it...

    >>> ad.nddata

If I want to access the `SCI`, `VAR`, or `DQ` frames I'll use, respectively:

    >>> ad.data
    >>> ad.variance
    >>> ad.mask

All of them are arrays (or `None`, if not assigned), and can be read and set.
Setting only works on single slices, though:

    >>> ad.data = array([.....])
    Traceback (most recent call last):
      ...
    ValueError: Trying to assign to a non-sliced AstroData object

`variance` is actually stored as a `NDUncertainty` instance, which you can
retrieve using:

    >>> ad.uncertainty

**When setting `data`, `variance`, or `mask` individually, none of their
values are checked for consistency. You will only realize that you introduced
and improper value when trying to operate, save the date, etc.**

There's a method that will let you set all the related data at the same time,
testing for consistency:

    >>> ad.reset(data=..., variance=..., mask=...)

## Arithmetics


