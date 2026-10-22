
Finds the edges of the illuminated regions of the CCD and stores the
Chebyshev polynomials used to fit them in a SLITEDGE table.

The primitive works by determining the locations of plausible slit
edges from fitting peaks to the first derivative of a spatial cut
across the image. These are then matched to predicted pairs of slit
edges (ensuring the handedness of the edges by assigning positive
and negative weights accordingly). The edges are traced in the
dispersion direction of the first-derivative image and a Chebyshev
polynomial fit to the data. If only one edge of a pair is found,
the other edge is assumed to be a parallel trace separated by the
expected slit width.

The polynomial model for each slit edge is placed in a SLITEDGE
table.

Parameters
----------
adinputs : list of :class:`~astrodata.AstroData`
    Science data as 2D spectral images.
suffix : str
    Suffix to be added to output files.
spectral_order : int
    Fitting order in the spectral direction (minimum of 1).
min_snr : float
    Minimum signal-to-noise ratio of peaks to be considered as slit
    edges
edges1, edges2 : list
    List (of matching length) of the pixel locations of the edges of
    illuminated regions in the image. `edges1` should be all the top or
    left edges, `edges2` the bottom or right edges.
search_radius : float
    Distance (in pixels) within which to search for the edges of
    illuminated regions.
debug_plots : bool
    Generate plots of several aspects of the fitting process.
debug_max_missed : int
    The maximum number of steps that can be missed before the trace is
    lost. The default value is set per instrument/mode, but can be
    changed if needed.
debug_max_shift : float
    The maximum perpendicular shift (in pixels) between rows/columns.
    The default value is set per instrument/mode, but can be changed if
    needed.
debug_step : int
    The number of rows/columns per step. The default value is set per
    instrument/mode, but can be changed if needed.
debug_nsum : int
    The number of rows/columns to sum each step. The default value is
    set per instrument/mode, but can be changed if needed.

Returns
-------
list of :class:`~astrodata.AstroData`
    Science data as 2D spectral images with a `SLITEDGE` table attached
    to each extension.
