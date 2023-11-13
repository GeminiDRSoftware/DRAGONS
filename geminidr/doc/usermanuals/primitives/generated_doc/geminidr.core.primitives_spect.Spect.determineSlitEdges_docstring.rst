
Finds the edges of the illuminated regions of the CCD and stores the
Chebyshev polynomials used to fit them in a SLITEDGE table.

Parameters
----------
adinputs : list of :class:`~astrodata.AstroData`
    Science data as 2D spectral images.
suffix : str
    Suffix to be added to output files.
spectral_order : int, Default : 3
    Fitting order in the spectral direction (minimum of 1).
debug : bool, Default: False
    Generate plots of several aspects of the fitting process.

Returns
-------
list of :class:`~astrodata.AstroData`
    Science data as 2D spectral images with a `SLITEDGE` table attached
    to each extension.
