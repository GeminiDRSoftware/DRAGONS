
Finds the edges of the illuminated regions of the detector and stores
the Chebyshev polynomials used to fit them in a SLITEDGE table.

The primitive works by determining the locations of plausible slit
edges from fitting peaks to the first derivative of a spatial cut
across the image. These are then matched to predicted pairs of slit
edges (ensuring the handedness of the edges by assigning positive
and negative weights accordingly). The edges are traced in the
dispersion direction of the first-derivative image and a Chebyshev
polynomial fit to the data. If only one edge of a pair is found,
the other edge is assumed to be a parallel trace separated by the
expected slit width.

The expected location of the slits, to use as initial conditions,
are found in a Mask Definition File (MDF) attached to each signal
extension, or they are given as input to the `edges1` and `edges2`
parameters.  (See primitive `addMDF`.)

The polynomial model for each slit edge is placed in a SLITEDGE
table.  The format is one column for each coefficient of the
polynomial, a slit ID column, an edge identifier
(0 for left/bottom, 1 for right/top), and a column for the spectral
order.   One row per edge.

Parameters
----------
adinputs : list of :class:`~astrodata.AstroData`
    Image with illuminated slits.
suffix : str
    Suffix to be added to output files.
nsum : int
    Number of rows/columns to sum when searching for peaks.
min_snr : float
    Minimum signal-to-noise ratio of peaks to be considered as slit
    edges.
spectral_order : int
    Order of the polynomial to fit to the edges. The fit is along
    the spectral direction (minimum of 1).
edges1, edges2 : list
    Expected pixel locations of the edges of the illuminated slit(s).
    If None, a Mask Definition File (MDF) must be attached to each
    signal extensions of the input file.  (See primitive addMDF.)
    If pixel locations are provided, the lists for `edges1` and
    `edges2` must be of equal length.
    `edges1` refer to bottom/left edges; `edges2` are the top/right
    edges.
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

Returns
-------
list of :class:`~astrodata.AstroData`
    Inputs with a `SLITEDGE` table attached  to each astrodata
    extension.
