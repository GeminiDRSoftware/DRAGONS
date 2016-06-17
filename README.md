AstroData Reborn
================

This project is a prototype to test the ideas drafted in the design document
for AstroData's new incarnation.

What to import
--------------

There are two main packages:

 - `astrodata`, with core functionality, mostly independent of Geminy stuff
 - `instruments`, with all Gemini related classes

Regarding the `AstroData` hierarchy, the `astrodata` package defines just a
couple of very generic ones: `AstroData` itself (which is abstract), and
`AstroDataFits`. The hierarchy is extended by additional packages (such as
3rd party ones).

Following the proposed design, to make the design as loosely coupled as
possible, `astrodata` is *totally* unaware of additional packages, and it
doesn't perform any "self-discovery" trying to find other modules and
packages that may apply. Instead, the additional functionality *must be*
actively registered by the user. This may be as easy as importing the
desired packages. Indeed, you'll need this somewhere in your script:

    import astrodata
    import instruments

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

As can be seen in this example, by importing just `astrodata`, the only
available type is `AstroDataFits` (which is mostly useless, because it
is too generic)
