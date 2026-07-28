
Adds headers to the AD object giving some statistics of the unmasked
pixel values
By default, adds:
PIXMEAN - the arithmetic mean of the pixel values
PIXSTDV - the standard deviation of the pixel values
PIXMED - the median of the pixel values.

Parameters
----------

adinputs: list of :class:`~astrodata.AstroData`

prefix: Prefix for header keywords. Maximum of 4 characters, defaults
        to PIX.
