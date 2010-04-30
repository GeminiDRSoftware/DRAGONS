


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
visualize our MEF files as, while consisting of multiple extensions,
single datasets.

Lack of a central system meant that scripts and tasks have to make
extended checks of their own on the header data of the datasets they
are manipulating. Often these are merely to verify the right type of
data is being worked on. Thus AstroData includes a classification
system where types are defined in configuration packages and can be
checked in a single line once the AstroData instance is created:

::

    
    from astrodata.AstroData import AstroData
    
    ad = AstroData("N20091027S0134.fits")
    
    if ad.isType("GMOS_IMAGE"):
       gmos_specific_function(ad)


