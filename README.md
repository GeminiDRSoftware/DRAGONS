AstroData Reborn
================

This project is a prototype to test the ideas drafted in the design document
for AstroData's new incarnation. What follows is a short tutorial on using
`AstroData` objects. Read to the end if you want a more detailed explanation
on what's going on.

Using the package
-----------------

There are two main packages:

 - `astrodata`, with core functionality, mostly independent of Geminy stuff
 - `instruments`, with all Gemini related classes

In short: you need to import both.

    >>> import astrodata
    >>> import instruments

The next step would be instantiating an `AstroData` object. This we do by
using a factory function:

    >>> ad = astrodata.open("/path/to/my/file.fits")

Descriptors and keywords
~~~~~~~~~~~~~~~~~~~~~~~~

To get a descriptor value, simply call it as a method:

    >>> ad.coadds()
    1

Some descriptors may accept arguments, like `stripID` or `pretty`. Descriptors
that extract information from the PHU will return a single value. Those that
work with the extensions will return a list of values, in the same order as
the extensions themselves.

You an access the raw keyword values in the PHU and the extension headers
directly, too, by using the following syntax:

    >>> ad.phu.TELESCOP
    'Gemini-North'
    >>> ad.ext.EXTVER
    [1, 2, 3, 4]
    # Set a keyword
    >>> ad.phu.SOMEKEYW = value
    >>> 'FOOBAR' in ad.phu
    True
    >>> 'BAR' in ad.ext
    (True, False, True)

NB: For the time being, when dealing with non-raw files, *only* SCI extensions
will be considered by this high-level interface.

Retrieving the value of a non-existing keyword will raise a `KeyError`. If this
happens with `.ext`, the exception will contain an additional `missing_at`
member that lists the indices for the extensions that missed the keyword.
If you want a value no matter what, yo can use `.get`, like you'd do on a
dictionary:

    >> ad.phu.get('RAOFFSET', 0)

The specified default value -`None`, if it's not provided,- will be substituted
for the missing one(s).

One can also get/set the comment for a keyword:

    >>> ad.phu.get_comment('LNRS')
    'Number of non-destructive read pairs'
    >>> ad.phu.set_comment('LNRS', 'Something else')

Data and tables
~~~~~~~~~~~~~~~

To access the data, we use the `nddata` property:

    >>> ad = astrodata.open('data/S20060131S0122.fits')
    >>> ad.phu.INSTRUME
    'GMOS-S'
    >>> ad.nddata
    [NDData([....]), NDData([....]), NDData([....])]

which returns a list. The number of elements on this list will be the same
as the extensions, for raw images; or the amount of `SCI` in case of
prepared ones.

Every `NDData` object has a `meta` attribute which is a dictionary of
"extra" attributes. All of them will have at least one element in the
`meta`: the HDU

`NDData` objects coming from prepared files may have additional components:

  * `meta['ver']` will contain the EXTVER value for this SCI extension
  * `nddata.mask` will be set if there's an EXTVER matching `DQ` extension
  * `nddata.uncertainty` will be set if there's an EXTVER matching `VAR` extension
  * For every extension matching the EXTVER that is not covered by the previous
    cases, there will be an additional attribute assigned to the SCI `NDData`
    object, named after the EXTNAME of the additional extension (`nddata.OBJMASK`,
    `nddata.OBJCAT`, ...), where:
    - If it's an IMAGE, the new attribute will be an `NDData`, itself;
    - If it's a TABLE, then the new attribute will be an instance of `astropy.table.Table`

Finally, if there's an MDF extension, it will be an attribute of the main AstroData
object (`ad.MDF`). The same is true for REFCAT extensions (only one copy will be returned).
Note that REFCATs have EXTVER matching an SCI but, at the moment, all REFCATs in a file are
identical copies. Thus, we expose them as a property of the AstroData object, instead of
associating them to their respective SCI `NDData`s.

Details: why do you need to import both `astrodata` and `instruments`??
-----------------------------------------------------------------------

Regarding the `AstroData` hierarchy, the `astrodata` package defines just a
couple of very generic ones classes: `AstroData` itself (which is abstract),
and `AstroDataFits`, which gives the machinery to read and access FITS files,
but has no knowledge about telescope or instrument details.. The hierarchy
is extended by additional packages (such as 3rd party ones).

Following the proposed design, to make things as loosely coupled as possible,
`astrodata` is *totally* unaware of additional packages, and it won't
perform any "self-discovery" trying to find other modules and packages that
may apply. Instead, the additional functionality *must be* actively
registered by the user. This may be as easy as importing the desired
packages. Indeed, you'll need this somewhere in your script:

The `instruments` package itself imports `astrodata` and registers
anything that could be needed for later use, as can be seen in the following
example:

    >>> import astrodata
    >>> ad = astrodata.open("data/N20120505S0564.fits")
    >>> type(ad)
    <class 'astrodata.fits.AstroDataFits'>
    >>> import instruments
    >>> ad = astrodata.open("data/N20120505S0564.fits")
    >>> type(ad)
    <class 'instruments.generic.AstroDataNiri'>

By importing just `astrodata`, the only available type is `AstroDataFits`
(which is mostly useless, because it is too generic).
