# Copyright(c) 2006,2016-2025 Association of Universities for Research in Astronomy, Inc.

"""
The astroTools module contains astronomy specific utility functions
"""

from copy import deepcopy
import os
import numpy as np
from scipy import interpolate, optimize

from astropy import units as u
from astropy import stats
from astropy.coordinates import Angle
from astropy.modeling import models, fitting


class Magnitude:
    # Wavelength (nm) and AB-Vega offsets
    # Optical data from Bessell et al. (1998; A&A 333, 231)
    # MK Filter set info from Tokunaga & Vacca (2005; PASP 117, 421)
    # except Y from Hewett et al. (2006; MNRAS 367, 454)
    VEGA_INFO = {
        "U": (366., 0.768),
        "B": (438., -0.122),
        "V": (545., 0.),
        "R": (641., 0.184),
        "I": (798., 0.442),
        "Y": (1030.5, 0.634),
        "J": (1250., 0.943),
        "H": (1644., 1.38),
        "K'": (2121., 1.84),
        "Kp": (2121., 1.84),
        "Ks": (2149., 1.96),
        "K": (2198., 1.90),
        "L'": (3754., 2.94),
        "Lp": (3754., 2.94),
        "M'": (4702., 3.40),
        "Mp": (4702., 3.40),
    }

    def __init__(self, valuestr, abmag=False):
        """
        Create a magnitude object from a string

        Parameters
        ----------
        valuestr: str
            the string representation of the magnitude
        """
        try:
            self._filter, value = valuestr.split("=")
            self._value = float(value)
        except (IndexError, ValueError):
            raise ValueError("Magnitude string must be of the form 'filter=value'")
        if self._filter not in self.VEGA_INFO:
            raise ValueError(f"Filter {self._filter} not recognized")
        self.abmag = abmag

    def __str__(self):
        return f"{self._filter}={self._value}"

    def wavelength(self, units=None):
        """
        Return the wavelength of the filter. This is a Quantity if no units
        are specified, otherwise a float.

        Parameters
        ----------
        units: astropy.units.Unit/None
            units of the wavelength

        Returns
        -------
        astropy.units.Quantity/float: the wavelength
        """
        w = self.VEGA_INFO[self._filter][0] * u.nm
        if units is None:
            return w
        return w.to(units).value

    def flux_density(self, units=None):
        """
        Return the flux density of the filter. This is a Quantity if no units
        are specified, otherwise a float.

        Parameters
        ----------
        units: astropy.units.Unit/None
            units of the flux density

        Returns
        -------
        astropy.units.Quantity/float: the flux density
        """
        mag = self._value + (0 if self.abmag else self.VEGA_INFO[self._filter][1])
        fluxden = 3630 * u.Jy / 10 ** (0.4 * mag)
        if units is None:
            return fluxden
        return fluxden.to(units, equivalencies=u.spectral_density(self.wavelength())).value

    def properties(self, wave_units=None, fluxden_units=None):
        """Convenience method to return wavelength and flux density"""
        return self.wavelength(units=wave_units), self.flux_density(units=fluxden_units)


def array_from_list(list_of_quantities, unit=None):
    """
    Convert a list of Quantity objects to a numpy array. The elements of the
    input list must all be converted to the same units.

    Parameters
    ----------
    list_of_quantities: list
        Quantities objects that all have equivalencies

    Returns
    -------
    array: array representation of this list
    """
    if unit is None:
        unit = list_of_quantities[0].unit
    values = [x.to(unit).value for x in list_of_quantities]
    # subok=True is needed to handle magnitude/log units
    return u.Quantity(np.array(values), unit, subok=True)


def boxcar(data, operation=np.ma.median, size=1):
    """
    "Smooth" a 1D array by applying a boxcar filter along it. Any operation
    can be performed, as long as it can take a sequence and return a single
    value.

    Parameters
    ----------
    data: 1D ndarray
        the data to be maninpulated
    operation: callable
        function for the boxcar to use
    size: int
        the boxcar width will be 2*size+1

    Returns
    -------
    1D ndarray: same shape as data, after the boxcar operation
    """
    try:
        boxarray = np.array([operation(data[max(i-size, 0):i+size+1])
                             for i in range(len(data))])
    except (ValueError, TypeError):  # Handle things like np.logical_and
        boxarray = np.array([operation.reduce(data[max(i-size, 0):i+size+1])
                             for i in range(len(data))])
    return boxarray


def calculate_pixel_edges(centers):
    """
    Calculate the world coordinates of the edges of pixels in a 1D array,
    given their centers. This is achieved by fitting a cubic spline to the
    centers and evaluating it at half-pixel locations.

    Parameters
    ----------
    centers: array-like, shape (N,)
        locations of centers of pixels

    Returns
    -------
    edges: array, shape (N+1,)
        locations of edges of pixels
    """
    spline = interpolate.CubicSpline(np.arange(len(centers)), centers)
    return spline(np.arange(len(centers)+1) - 0.5)


def calculate_scaling(x, y, sigma_x=None, sigma_y=None, sigma=3, niter=2):
    """
    Determine the optimum value by which to scale inputs so that they match
    a set of reference values

    Parameters
    ----------
    x: array
        values to be scaled (inputs)
    y: array
        values to be scaled to (references)
    sigma_x: array/None
        standard deviations on each input value
    sigma_y: array/None
        standard deviations on each reference value
    sigma: float/None
        sigma_clipping value (None => no clipping)
    niter: int
        number of clipping iterations

    Returns
    -------
    factor: float
        the best-fitting scaling factor
    """
    x, y = np.asarray(x), np.asarray(y)
    if sigma_x is None and sigma_y is None:
        weights = None
        init_guess = (x * y).sum() / (x * x).sum()
    elif sigma_x is None:
        weights = 1 / np.asarray(sigma_y)
        init_guess = (x * y *weights**2).sum() / (x * x *weights**2).sum()
    elif sigma_y is None:
        weights = 1 / np.asarray(sigma_x)
        init_guess = np.square(y * weights).sum() / (x * y *weights**2).sum()
    else:
        # Calculus has failed me here, I don't think this is linear
        fun = lambda f, x, y, sx, sy: np.square((f * x - y) / (f*f*sx*sx + sy*sy)).sum()
        result = optimize.minimize(fun, [1.], args=(x, y, sigma_x, sigma_y))
        init_guess = result.x[0]
        # Assume the initial guess is pretty good and so the weights are these
        # We really want scipy.odr (Orthogonal Distance Regression) but we
        # have to choose between that and outlier removal, and outlier
        # removal is more important
        weights = 1 / np.sqrt(sigma_y ** 2 + (init_guess * sigma_x)**2)

    if sigma is None:
        return init_guess

    m_init = models.Scale(init_guess)
    fit_it = fitting.FittingWithOutlierRemoval(
        fitting.LinearLSQFitter(), outlier_func=stats.sigma_clip, niter=niter,
        sigma=sigma)
    m_final, _ = fit_it(m_init, x, y, weights=weights)
    return float(m_final.factor.value)  # don't force upcasting in N2 arith.


def divide0(numerator, denominator):
    """
    Perform division, replacing division by zero with zero. This expands
    on the np.divide() function by having to deal with cases where either
    the numerator and/or denominator might be scalars, rather than arrays,
    and also deals with cases where they might be integer types.

    Parameters
    ----------
    numerator: float/array-like
        the numerator for the division
    denominator: float/array-like
        the denominator for the division

    Returns
    -------
    The quotient, with instances where the denominator is zero replace by zero
    """
    try:
        is_int = np.issubdtype(denominator.dtype, np.integer)
    except AttributeError:
        # denominator is a scalar
        if denominator == 0:
            try:
                return np.zeros(numerator.shape)
            except AttributeError:
                # numerator is also a scalar
                return 0
        else:
            return numerator / denominator
    else:
        dtype = np.float32 if is_int else denominator.dtype
        try:
            out_shape = numerator.shape
        except:
            out_shape = denominator.shape
        else:
            # both are arrays so the final shape will be the one with the
            # higher dimensionality (if they're broadcastable)
            if len(out_shape) < len(denominator.shape):
                out_shape = denominator.shape

        return np.divide(numerator, denominator, out=np.zeros(out_shape, dtype=dtype),
                         where=abs(denominator) > np.finfo(dtype).tiny)


def fit_spline_to_data(data, mask=None, variance=None, k=3):
    """
    Fit a spline to data, weighting by variance. If no variance is supplied,
    it is computed from the pixel-to-pixel variations in the data and an
    additional component is added based on how rapidly the data are varying.
    This is important to prevent "overfitting" of the slopes of features.

    Parameters
    ----------
    data: array
        data to which a spline is to be fitted
    mask: array/None
        mask values for each data point
    variance: array/None
        variance of each data point
    k: int
        order of the spline

    Returns
    -------
    callable spline object: the fitted spline
    """
    if variance is None:
        y = np.ma.masked_array(data, mask=mask)
        sigma1 = std_from_pixel_variations(y, subtract_linear_fits=True)
        diff = np.diff(y)
        # 0.1 is a fudge factor that seems to work well
        #sigma2 = 0.1 * (np.r_[diff, [0]] + np.r_[[0], diff])
        # Average from 2 pixels either side; 0.05 corresponds to 0.1 before
        # since this is the change in a 4-pixel span, instead of 2 pixels
        diff2 = np.r_[diff[:2], y[4:] - y[:-4], diff[-2:]]
        sigma2 = 0.05 * diff2
        w = 1. / np.sqrt(sigma1 * sigma1 + sigma2 * sigma2)
        if mask is not None:
            mask = mask | w.mask
    else:
        w = divide0(1.0, np.sqrt(variance))

    # TODO: Consider outlier removal
    iterations = 0
    sigma_limit = 10 # spline residuals within this limit don't trigger refitting
    change = 0.90
    iters_number = 50
    if mask is None:
        x = np.arange(data.size)
        interp_x = (x[:-1] + x[1:]) / 2 # get values between data point locations
        interp_data = (data[:-1] + data[1:]) / 2
        interp_w = (w[:-1] + w[1:]) / 2
        while iterations < iters_number:
            spline = interpolate.UnivariateSpline(x, data,  w=w, k=k)
            residuals = spline(interp_x) - interp_data
            if not ((residuals - ((1 / interp_w) * sigma_limit)).max() > 0).any():
                break
            else:
                interp_w *= change
                iterations += 1
    else:
        x = np.arange(data.size)[mask == 0]
        interp_x = (x[:-1] + x[1:]) / 2
        interp_data = (data[mask == 0][:-1] + data[mask == 0][1:]) / 2
        interp_w = (w[mask == 0][:-1] + w[mask == 0][1:]) / 2
        while iterations < iters_number:
            spline = interpolate.UnivariateSpline(x, data[mask == 0],
                                                  w=w[mask == 0], k=k)
            residuals = spline(interp_x) - interp_data
            if not ((residuals - ((1 / interp_w) * sigma_limit)).max() > 0).any():
                break
            else:
                interp_w *= change
                iterations += 1

    return spline


def std_from_pixel_variations(array, separation=5, subtract_linear_fits=True,
                              **kwargs):
    """
    Estimate the standard deviation of pixels in an array by measuring the
    pixel-to-pixel variations. Since the values might be correlated over small
    scales (e.g., if the data have been smoothed), pixels are compared not
    with their immediate neighbors but with pixels a few locations away.

    Subtracting a linear fit from subgroups of pixels helps recover the standard
    deviation from pixel arrays that are dominated by signal, where the large-
    scale correlations would otherwise overwhelm the pixel-to-pixel variation.
    For arrays without much signal, the differences between both options are
    generally minor.

    Parameters
    ----------
    array: array-like
        the data from which the standard deviation is to be estimated
    separation: int
        separation between pixels being compared
    subtract_linear_fits: bool
        subtract linear fit from each sub-group of separation+3 pixels
    kwargs: dict
        kwargs to be passed directly to astropy.stats.sigma_clipped_stats

    Returns
    -------
    float: the estimated standard deviation
    """
    _array = np.asarray(array).ravel()
    if subtract_linear_fits:
        # Make mini-arrays of groups of separation+3 pixels
        data = np.vstack([_array[i:i+separation+3]
                         for i in range(_array.size-separation-2)]).T
        # Fit and subtract a least-squares linear function to each mini-array
        xrange = np.arange(separation+3)
        A = np.vstack([xrange, np.ones(xrange.size)]).T
        m, c = np.linalg.lstsq(A, data, rcond=None)[0]
        corr_data = data - (m * xrange[:, np.newaxis] + c)
        diffs = (corr_data[1] - corr_data[-2])
    else:
        diffs = _array[separation:] - _array[:-separation]
    ok = ~(np.isnan(diffs) | np.isinf(diffs))  # Stops AstropyUserWarning
    return stats.sigma_clipped_stats(diffs[ok], **kwargs)[2] / np.sqrt(2)


def cartesian_regions_to_slices(regions):
    """
    Convert a sample region(s) string, consisting of a comma-separated list
    of (colon-or-hyphen-separated) pixel ranges into Python slice objects.

    These ranges may describe either multiple 1D regions or a single higher-
    dimensional region (with one range per axis), a distinction which is not
    important here. The ranges are specified in 1-indexed Cartesian pixel
    co-ordinates, inclusive of the upper limit, and get converted to a tuple
    of Python slice objects (in reverse order), suitable for indexing a
    :mod:`numpy` array. If regions is None or empty, the resulting slice will
    select the entire array. Single indices may be used (eg. '1,2'), or '*'
    for the whole axis, lower and/or upper limits may be omitted to use the
    remainder of the range (eg. '10:,:') and/or an optional step may be
    specified using colon syntax (eg. '1:10:2' or '1-10:2').
    """
    if not regions:
        return (slice(None, None, None),)

    if not isinstance(regions, str):
        raise TypeError('region must be a string or None, not \'{}\''
                        .format(regions))

    origin = 1

    slices = []
    ranges = parse_user_regions(regions, allow_step=True)

    for limits in ranges[::-1]:           # reverse Cartesian order for Python
        nlim = len(limits)
        if nlim == 1:
            lim = int(limits[0])-origin
            sliceobj = slice(lim, lim+1)
        else:
            # Adjust only the lower limit for 1-based indexing since Python
            # ranges are exclusive:
            sliceobj = slice(*(int(lim)-adj if lim else None
                               for lim, adj in zip(limits, (origin, 0, 0))))
        slices.append(sliceobj)

    return tuple(slices)


def parse_user_regions(regions, dtype=int, allow_step=False):
    """
    Parse a string containing a list of sections into a list of tuples
    containing the same information

    Parameters
    ----------
    regions : str
        comma-separated list of regions of form start:stop:step
    dtype : dtype
        string values will be coerced into this dtype, raising an error if
        this is not possible
    allow_step : bool
        allow a step value in the ranges?

    Returns
    -------
    list of slice-like tuples with 2 or 3 values per tuple
    """
    if not regions:
        return [(None, None)]
    elif not isinstance(regions, str):
        raise TypeError(f"regions must be a string or None, not '{regions}'")

    if isinstance(dtype, np.dtype):
        dtype = getattr(np, dtype.name)

    ranges = []
    for range_ in regions.strip("[]").split(","):
        range_ = range_.strip()
        if range_ == "*":
            ranges.append((None, None))
            continue
        try:
            values = [dtype(x) if x else None
                      for x in range_.replace("-", ":", 1).split(":")]
            assert len(values) in (1, 2, 2+allow_step)
            if len(values) > 1 and values[0] is not None and values[1] is not None and values[0] > values[1]:
                values[0], values[1] = values[1], values[0]
        except (ValueError, AssertionError):
            raise ValueError(f"Failed to parse sample regions '{regions}'")
        ranges.append(tuple(values))
    return ranges


def create_mask_from_regions(points, regions=None):
    """
    Produce a boolean mask given an array of x-values and a list of unmasked
    regions. The regions can be specified either as slice objects (interpreted
    as pixel indices in the standard python sense) or (start, end) tuples with
    inclusive boundaries.

    Parameters
    ----------
    points : `numpy.ndarray`
        Input array
    regions : list, optional
        valid regions, either (start, end) or slice objects

    Returns
    -------
    mask : boolean `numpy.ndarray`
    """
    mask = np.ones_like(points, dtype=bool)
    if regions:
        for region in regions:
            if isinstance(region, slice):
                mask[region] = False
            else:
                x1 = min(points) if region[0] is None else region[0]
                x2 = max(points) if region[1] is None else region[1]
                if x1 > x2:
                    x1, x2 = x2, x1
                mask[np.logical_and(points >= x1, points <= x2)] = False
    return mask


def get_center_of_projection(wcs):
    """
    Determines the location of the center of projection from a gWCS object

    Parameters
    ----------
    wcs: gWCS object

    Returns
    -------
    ra, dec: location of pole
    """
    for m in wcs.forward_transform:
        if isinstance(m, models.RotateNative2Celestial):
            return (m.lon.value, m.lat.value)


def find_outliers(data, sigma=3, cenfunc=np.median):
    """
    Examine a list for individual outlying points and flag them.
    This operates better than sigma-clip for small lists as it sequentially
    removes a single point from the list and determined whether it is an
    outlier based on the statistics of the remaining points.

    Parameters
    ----------
    data: array-like
        array to be searched for outliers
    sigma: float
        number of standard deviations for rejection
    cenfunc: callable (np.median/np.mean usually)
        function for deriving average

    Returns
    -------
    mask: bool array
        outlying points are flagged
    """
    mask = np.zeros_like(data, dtype=bool)
    if mask.size < 3:  # doesn't work with 1 or 2 elements
        return mask

    for i in range(mask.size):
        omitted_data = list(deepcopy(data))
        omitted_data.pop(i) # Remove one point
        average = cenfunc(omitted_data)
        stddev = np.std(omitted_data)
        if abs(data[i] - average) > sigma * stddev:
            mask[i] = True
    return mask


def get_corners(shape):
    """
    This is a recursive function to calculate the corner indices
    of an array of the specified shape.

    :param shape: length of the dimensions of the array
    :type shape: tuple of ints, one for each dimension

    """
    if not type(shape) == tuple:
        raise TypeError('get_corners argument is non-tuple')

    if len(shape) == 1:
        corners = [(0,), (shape[0]-1,)]
    else:
        shape_less1 = shape[1:len(shape)]
        corners_less1 = get_corners(shape_less1)
        corners = []
        for corner in corners_less1:
            newcorner = (0,) + corner
            corners.append(newcorner)
            newcorner = (shape[0]-1,) + corner
            corners.append(newcorner)

    return corners


def get_spline3_extrema(spline):
    """
    Find the locations of the minima and maxima of a cubic spline.

    Parameters
    ----------
    spline: a callable spline object

    Returns
    -------
    minima, maxima: 1D arrays
    """
    derivative = spline.derivative()
    try:
        knots = derivative.get_knots()
    except AttributeError:  # for BSplines
        knots = derivative.t

    minima, maxima = [], []
    # We take each pair of knots and map to interval [-1,1]
    for xm, xp in zip(knots[:-1], knots[1:]):
        ym, y0, yp = derivative([xm, 0.5*(xm+xp), xp])
        # a*x^2 + b*x + c
        a = 0.5 * (ym+yp) - y0
        b = 0.5 * (yp-ym)
        c = y0
        for root in np.roots([a, b, c]):
            if np.isreal(root) and abs(root) <= 1:
                x = 0.5 * (root * (xp-xm) + (xp+xm))  # unmapped from [-1, 1]
                if 2*a*root + b > 0:
                    minima.append(x)
                else:
                    maxima.append(x)
    return np.array(minima), np.array(maxima)


def spherical_offsets_by_pa(coord1, coord2, position_angle=0):
    """
    Calculates the spherical offsets between two sky coordinates relative to
    a specific position angle.

    Parameters
    ----------
    coord1: astropy.coordinates.SkyCoord object
        initial position
    coord2: astropy.coordinates.SkyCoord object
        offset position
    position_angle: float
        position angle (in degrees) of slit

    Returns
    -------
    dist_para, dist_perp: floats
        the offsets (in arcseconds) parallel and perpendicular to the slit
        between the two coordinates
    """
    frame = coord1.skyoffset_frame(rotation=Angle(position_angle, unit='deg'))
    offset_coord = coord2.transform_to(frame)
    # coord1 is (0, 0) in the new frame of course
    dist_para = offset_coord.lat.deg * 3600
    dist_perp = offset_coord.lon.deg * 3600
    return dist_para, dist_perp


def transpose_if_needed(*args, transpose=False, section=slice(None)):
    """
    This function takes a list of arrays and returns them (or a section of them),
    either untouched, or transposed, according to the parameter.

    Parameters
    ----------
    args : sequence of arrays
        The input arrays.
    transpose : bool
        If True, return transposed versions.
    section : slice object
        Section of output data to return.

    Returns
    -------
    list of arrays
        The input arrays, or their transposed versions.
    """
    return list(None if arg is None
                else arg.T[section] if transpose else arg[section] for arg in args)


def weighted_sigma_clip(data, weights=None, sigma=3, sigma_lower=None,
                        sigma_upper=None, maxiters=5):
    """
    Perform sigma-clipping on a dataset, accounting for different relative
    weights of the data.

    Parameters
    ----------
    data: array/masked_array
        the data
    weights: array/None
        relative weights of the data
    sigma: float/None
        number of standard deviations to clip if clipping symmetrically
    sigma_lower: float/None
        number of standard deviations for lower clip (non-symmetric)
    sigma_upper: float/None
        number of standard deviations for upper clip (non-symmetric)
    maxiters: int
        maximum number of iterations to perform

    Returns
    -------
    np.ma.masked_array: data with mask indicating clipped points
    """
    if sigma_lower is None or sigma_upper is None:
        sigma_lower = sigma_upper = sigma
    if weights is None:
        weights = np.ones_like(data)

    if isinstance(data, np.ma.masked_array):
        good = ~data.mask
        data = data.data
    else:
        good = np.ones_like(data, dtype=bool)

    niter = 0
    while True:
        avg = (np.average(data[good], weights=weights[good]) if niter > 0
               else np.median(data[good]))
        ngood = good.sum()
        if ngood <= 1 or niter == maxiters:
            break
        std = np.sqrt(np.sum(weights[good] * (data[good] - avg) ** 2) /
                      np.sum(weights[good]))
        good[np.logical_or(data < avg - sigma_lower * std,
                           data > avg + sigma_upper * std)] = False
        if good.sum() == ngood:
            break
        niter += 1

    return np.ma.masked_array(data, mask=~good)


def clipped_mean(data):
    num_total = len(data)
    mean = data.mean()
    sigma = data.std()

    if num_total < 3:
        return mean, sigma

    num = num_total
    clipped_data = data
    clip = 0
    while num > 0.5 * num_total:
        # CJS: edited this as upper limit was mean+1*sigma => bias
        clipped_data = data[(data < mean + 3*sigma) & (data > mean - 3*sigma)]
        num = len(clipped_data)

        if num > 0:
            mean = clipped_data.mean()
            sigma = clipped_data.std()
        elif clip == 0:
            return mean, sigma
        else:
            break

        clip += 1
        if clip > 10:
            break

    return mean, sigma


# The following functions and classes were borrowed from STSCI's spectools
# package, currently under development.  They might be able to be
# replaced with a direct import of spectools.util if/when it is available

IRAF_MODELS_MAP = {1.: 'chebyshev',
                   2.: 'legendre',
                   3.: 'spline3',
                   4.: 'spline1'}
INVERSE_IRAF_MODELS_MAP = {'chebyshev': 1.,
                           'legendre': 2.,
                           'spline3': 3.,
                           'spline1': 4.}

def get_records(fname):
    """
    Read the records of an IRAF database file ionto a python list

    Parameters
    ----------
    fname: string
           name of an IRAF database file

    Returns
    -------
        A list of records
    """
    filehandle = open(fname)
    dtb = filehandle.read()
    filehandle.close()
    records = []
    recs = dtb.split('begin')[1:]
    records = [Record(r) for r in recs]
    return records

def get_database_string(fname):
    """
    Read an IRAF database file

    Parameters
    ----------
    fname: string
          name of an IRAF database file

    Returns
    -------
        the database file as a string
    """
    f = open(fname)
    dtb = f.read()
    f.close()
    return dtb

class Record:
    """
    A base class for all records - represents an IRAF database record

    Attributes
    ----------
    recstr: string
            the record as a string
    fields: dict
            the fields in the record
    taskname: string
            the name of the task which created the database file
    """
    def __init__(self, recstr):
        self.recstr = recstr
        self.fields = self.get_fields()
        self.taskname = self.get_task_name()

    def aslist(self):
        reclist = self.recstr.split('\n')
        reclist = [l.strip() for l in reclist]
        out = [reclist.remove(l) for l in reclist if len(l) == 0]
        return reclist

    def get_fields(self):
        # read record fields as an array
        fields = {}
        flist = self.aslist()
        numfields = len(flist)
        for i in range(numfields):
            line = flist[i]
            if line and line[0].isalpha():
                field = line.split()
                if i+1 < numfields:
                    if not flist[i+1][0].isalpha():
                        fields[field[0]] = self.read_array_field(
                                             flist[i:i+int(field[1])+1])
                    else:
                        fields[field[0]] = " ".join(s for s in field[1:])
                else:
                    fields[field[0]] = " ".join(s for s in field[1:])
            else:
                continue
        return fields

    def get_task_name(self):
        try:
            return self.fields['task']
        except KeyError:
            return None

    def read_array_field(self, fieldlist):
        # Turn an iraf record array field into a numpy array
        fieldline = [l.split() for l in fieldlist[1:]]
        # take only the first 3 columns
        # identify writes also strings at the end of some field lines
        xyz = [l[:3] for l in fieldline]
        try:
            farr = np.array(xyz)
        except:
            print("Could not read array field %s" % fieldlist[0].split()[0])
        return farr.astype(np.float64)

class IdentifyRecord(Record):
    """
    Represents a database record for the longslit.identify task

    Attributes
    ----------
    x: array
       the X values of the identified features
       this represents values on axis1 (image rows)
    y: int
       the Y values of the identified features
       (image columns)
    z: array
       the values which X maps into
    modelname: string
        the function used to fit the data
    nterms: int
        degree of the polynomial which was fit to the data
        in IRAF this is the number of coefficients, not the order
    mrange: list
        the range of the data
    coeff: array
        function (modelname) coefficients
    """
    def __init__(self, recstr):
        super().__init__(recstr)
        self._flatcoeff = self.fields['coefficients'].flatten()
        self.x = self.fields['features'][:, 0]
        self.y = self.get_ydata()
        self.z = self.fields['features'][:, 1]
####here - ref?
        self.zref = self.fields['features'][:, 2]
        self.modelname = self.get_model_name()
        self.nterms = self.get_nterms()
        self.mrange = self.get_range()
        self.coeff = self.get_coeff()

    def get_model_name(self):
        return IRAF_MODELS_MAP[self._flatcoeff[0]]

    def get_nterms(self):
        return self._flatcoeff[1]

    def get_range(self):
        low = self._flatcoeff[2]
        high = self._flatcoeff[3]
        return [low, high]

    def get_coeff(self):
        return self._flatcoeff[4:]

    def get_ydata(self):
        image = self.fields['image']
        left = image.find('[')+1
        right = image.find(']')
        section = image[left:right]
        if ',' in section:
            yind = image.find(',')+1
            return int(image[yind:-1])
        else:
            return int(section)
        #xind = image.find('[')+1
        #yind = image.find(',')+1
        #return int(image[yind:-1])

class FitcoordsRecord(Record):
    """
    Represents a database record for the longslit.fitccords task

    Attributes
    ----------
    modelname: string
        the function used to fit the data
    xorder: int
        number of terms in x
    yorder: int
        number of terms in y
    xbounds: list
        data range in x
    ybounds: list
        data range in y
    coeff: array
        function coefficients

    """
    def __init__(self, recstr):
        super().__init__(recstr)
        self._surface = self.fields['surface'].flatten()
        self.modelname = IRAF_MODELS_MAP[self._surface[0]]
        self.xorder = self._surface[1]
        self.yorder = self._surface[2]
        self.xbounds = [self._surface[4], self._surface[5]]
        self.ybounds = [self._surface[6], self._surface[7]]
        self.coeff = self.get_coeff()

    def get_coeff(self):
        return self._surface[8:]

class IDB:
    """
    Base class for an IRAF identify database

    Attributes
    ----------
    records: list
             a list of all `IdentifyRecord` in the database
    numrecords: int
             number of records
    """
    def __init__(self, dtbstr):
        lst = self.aslist(dtbstr)
        self.records = [IdentifyRecord(rstr) for rstr in self.aslist(dtbstr)]
        self.numrecords = len(self.records)

    def aslist(self, dtb):
        # return a list of records
        # if the first one is a comment remove it from the list
        record_list = dtb.split('begin')
        try:
            rl0 = record_list[0].split('\n')
        except:
            return record_list
        if len(rl0) == 2 and rl0[0].startswith('#') and not rl0[1].strip():
            return record_list[1:]
        elif len(rl0) == 1 and not rl0[0].strip():
            return record_list[1:]
        else:
            return record_list

class ReidentifyRecord(IDB):
    """
    Represents a database record for the onedspec.reidentify task
    """
    def __init__(self, databasestr):
        super().__init__(databasestr)
        self.x = np.array([r.x for r in self.records])
        self.y = self.get_ydata()
        self.z = np.array([r.z for r in self.records])


    def get_ydata(self):
        y = np.ones(self.x.shape)
        y = y * np.array([r.y for r in self.records])[:, np.newaxis]
        return y


# This class pulls together fitcoords and identify databases into
# a single entity that can be written to or read from disk files
# or pyfits binary tables
class SpectralDatabase:
    def __init__(self, database_name=None, record_name=None,
                 binary_table=None):
        """
        database_name is the name of the database directory
        on disk that contains the database files associated with
        record_name.  For example, database_name="database",
        record_name="image_001" (corresponding to the first science
        extention in a data file called image.fits
        """
        self.database_name = database_name
        self.record_name = record_name
        self.binary_table = binary_table

        self.identify_database = None
        self.fitcoords_database = None

        # Initialize from database on disk
        if database_name is not None and record_name is not None:

            if not os.path.isdir(database_name):
                raise OSError('Database directory %s does not exist' %
                              database_name)

            # Read in identify database
            db_filename = "%s/id%s" % (database_name, record_name)
            if not os.access(db_filename, os.R_OK):
                raise OSError("Database file %s does not exist " \
                              "or cannot be accessed" % db_filename)

            db_str = get_database_string(db_filename)
            self.identify_database = IDB(db_str)

            # Read in fitcoords database
            db_filename = "%s/fc%s" % (database_name, record_name)
            if not os.access(db_filename, os.R_OK):
                raise OSError("Database file %s does not exist " \
                              "or cannot be accessed" % db_filename)

            db_str = get_database_string(db_filename)
            self.fitcoords_database = FitcoordsRecord(db_str)

        # Initialize from pyfits binary table in memory
        elif binary_table is not None:

            # Get record_name from header if not passed
            if record_name is not None:
                self.record_name = record_name
            else:
                self.record_name = binary_table.header["RECORDNM"]

            # Format identify information from header and table
            # data into a database string
            db_str = self._identify_db_from_table(binary_table)
            self.identify_database = IDB(db_str)

            # Format fitcoords information from header
            # into a database string
            db_str = self._fitcoords_db_from_table(binary_table)
            self.fitcoords_database = FitcoordsRecord(db_str)

        else:
            raise TypeError("Both database and binary table are None.")

    def _identify_db_from_table(self, tab):

        # Get feature information from table data
        features = tab.data
        nrows = len(features)
        nfeat = features["spectral_coord"].shape[1]
        ncoeff = features["fit_coefficients"].shape[1]

        db_str = ""

        for row in range(nrows):

            feature = features[row]

            # Make a dictionary to hold information gathered from
            # the table.  This structure is not quite the same as
            # the fields member of the Record class, but it is the
            # same principle
            fields = {}
            fields["id"] = self.record_name
            fields["task"] = "identify"
            fields["image"] = "%s[*,%d]" % (self.record_name,
                                            feature["spatial_coord"])
            fields["units"] = tab.header["IDUNITS"]

            zip_feature = np.array([feature["spectral_coord"],
                                    feature["fit_wavelength"],
                                    feature["ref_wavelength"]])
            fields["features"] = zip_feature.swapaxes(0, 1)

            fields["function"] = tab.header["IDFUNCTN"]
            fields["order"] = tab.header["IDORDER"]
            fields["sample"] = tab.header["IDSAMPLE"]
            fields["naverage"] = tab.header["IDNAVER"]
            fields["niterate"] = tab.header["IDNITER"]

            reject = tab.header["IDREJECT"].split()
            fields["low_reject"] = float(reject[0])
            fields["high_reject"] = float(reject[1])
            fields["grow"] = tab.header["IDGROW"]

            # coefficients is a list of numbers with the following elements:
            # 0: model number (function type)
            # 1: order
            # 2: x min
            # 3: x max
            # 4 on: function coefficients
            coefficients = []

            model_num = INVERSE_IRAF_MODELS_MAP[fields["function"]]
            coefficients.append(model_num)

            coefficients.append(fields["order"])

            idrange = tab.header["IDRANGE"].split()
            coefficients.append(float(idrange[0]))
            coefficients.append(float(idrange[1]))

            fit_coeff = feature["fit_coefficients"].tolist()
            coefficients.extend(fit_coeff)
            fields["coefficients"] = np.array(coefficients).astype(np.float64)


            # Compose fields into a single string
            rec_str = "%-8s%-8s %s\n" % \
                      ("begin", fields["task"], fields["image"])
            for field in ["id", "task", "image", "units"]:
                rec_str += "%-8s%-8s%s\n" % ("", field, str(fields[field]))
            rec_str += "%-8s%-8s %d\n" % \
                       ("", "features", len(fields["features"]))
            for feat in fields["features"]:
                rec_str += "%16s%10f %10f %10f\n" % \
                           ("", feat[0], feat[1], feat[2])
            for field in ["function", "order", "sample",
                          "naverage", "niterate", "low_reject",
                          "high_reject", "grow"]:
                rec_str += "%-8s%s %s\n" % ("", field, str(fields[field]))
            rec_str += "%-8s%-8s %d\n" % ("", "coefficients",
                                         len(fields["coefficients"]))
            for coeff in fields["coefficients"]:
                rec_str += "%-8s%-8s%E\n" % ("", "", coeff)
            rec_str += "\n"

            db_str += rec_str

        return db_str

    def _fitcoords_db_from_table(self, tab):

        # Make a dictionary to hold information gathered from
        # the table.  This structure is not quite the same as
        # the fields member of the Record class, but it is the
        # same principle
        fields = {}

        fields["begin"] = self.record_name
        fields["task"] = "fitcoords"
        fields["axis"] = tab.header["FCAXIS"]
        fields["units"] = tab.header["FCUNITS"]

        # The surface is a list of numbers with the following elements:
        # 0: model number (function type)
        # 1: x order
        # 2: y order
        # 3: cross-term type (always 1. for fitcoords)
        # 4. xmin
        # 5: xmax
        # 6. xmin
        # 7: xmax
        # 8 on: function coefficients
        surface = []

        model_num = INVERSE_IRAF_MODELS_MAP[tab.header["FCFUNCTN"]]
        surface.append(model_num)

        xorder = tab.header["FCXORDER"]
        yorder = tab.header["FCYORDER"]
        surface.append(xorder)
        surface.append(yorder)
        surface.append(1.)

        fcxrange = tab.header["FCXRANGE"].split()
        surface.append(float(fcxrange[0]))
        surface.append(float(fcxrange[1]))
        fcyrange = tab.header["FCYRANGE"].split()
        surface.append(float(fcyrange[0]))
        surface.append(float(fcyrange[1]))

        for i in range(int(xorder)*int(yorder)):
            coeff = tab.header["FCCOEF%d" % i]
            surface.append(coeff)

        fields["surface"] = np.array(surface).astype(np.float64)

        # Compose fields into a single string
        db_str = "%-8s%s\n" % ("begin", fields["begin"])
        for field in ["task", "axis", "units"]:
            db_str += "%-8s%-8s%s\n" % ("", field, str(fields[field]))
        db_str += "%-8s%-8s%d\n" % ("", "surface", len(fields["surface"]))
        for coeff in fields["surface"]:
            db_str += "%-8s%-8s%E\n" % ("", "", coeff)

        return db_str

    def write_to_disk(self, database_name=None, record_name=None):

        # Check for provided names; use names from self if not
        # provided as input
        if database_name is None and self.database_name is None:
            raise TypeError("No database_name provided")
        elif database_name is None and self.database_name is not None:
            database_name = self.database_name
        if record_name is None and self.record_name is None:
            raise TypeError("No record_name provided")
        elif record_name is None and self.record_name is not None:
            record_name = self.record_name

        # Make the directory if needed
        if not os.path.exists(database_name):
            os.mkdir(database_name)

        # Timestamp
        import datetime
        timestamp = str(datetime.datetime.now())

        # Write identify files
        id_db = self.identify_database
        if id_db is not None:
            db_filename = "%s/id%s" % (database_name, record_name)
            db_file = open(db_filename, "w")
            db_file.write("# "+timestamp+"\n")
            for record in id_db.records:
                db_file.write("begin")
                db_file.write(record.recstr)
            db_file.close()

        # Write fitcoords files
        fc_db = self.fitcoords_database
        if fc_db is not None:
            db_filename = "%s/fc%s" % (database_name, record_name)
            db_file = open(db_filename, "w")
            db_file.write("# "+timestamp+"\n")
            db_file.write(fc_db.recstr)
            db_file.close()

    def as_binary_table(self, record_name=None):

        # Should this be lazy loaded?
        import astropy.io.fits as pf

        if record_name is None:
            record_name = self.record_name

        # Get the maximum number of features identified in any
        # record.  Use this as the length of the array in the
        # wavelength_coord and fit_wavelength fields
        nfeat = max([len(record.x)
                     for record in self.identify_database.records])

        # The number of coefficients should be the same for all
        # records, so take the value from the first record
        ncoeff = self.identify_database.records[0].nterms

        # Get the number of rows from the number of identify records
        nrows = self.identify_database.numrecords

        # Create pyfits Columns for the table
        column_formats = [{"name":"spatial_coord", "format":"I"},
                          {"name":"spectral_coord", "format":"%dE"%nfeat},
                          {"name":"fit_wavelength", "format":"%dE"%nfeat},
                          {"name":"ref_wavelength", "format":"%dE"%nfeat},
                          {"name":"fit_coefficients", "format":"%dE"%ncoeff},]
        columns = [pf.Column(**fmt) for fmt in column_formats]

        # Make the empty table.  Use the number of records in the
        # database as the number of rows
        table = pf.new_table(columns, nrows=nrows)

        # Populate the table from the records
        for i in range(nrows):
            record = self.identify_database.records[i]
            row = table.data[i]
            row["spatial_coord"] = record.y
            row["fit_coefficients"] = record.coeff
            if len(row["spectral_coord"]) != len(record.x):
                row["spectral_coord"][:len(record.x)] = record.x
                row["spectral_coord"][len(record.x):] = -999
            else:
                row["spectral_coord"] = record.x
            if len(row["fit_wavelength"]) != len(record.z):
                row["fit_wavelength"][:len(record.z)] = record.z
                row["fit_wavelength"][len(record.z):] = -999
            else:
                row["fit_wavelength"] = record.z
            if len(row["ref_wavelength"]) != len(record.zref):
                row["ref_wavelength"][:len(record.zref)] = record.zref
                row["ref_wavelength"][len(record.zref):] = -999
            else:
                row["ref_wavelength"] = record.zref

        # Store the record name in the header
        table.header.update("RECORDNM", record_name)

        # Store other important values from the identify records in the header
        # These should be the same for all records, so take values
        # from the first record
        first_record = self.identify_database.records[0]
        table.header.update("IDUNITS", first_record.fields["units"])
        table.header.update("IDFUNCTN", first_record.modelname)
        table.header.update("IDORDER", first_record.nterms)
        table.header.update("IDSAMPLE", first_record.fields["sample"])
        table.header.update("IDNAVER", first_record.fields["naverage"])
        table.header.update("IDNITER", first_record.fields["niterate"])
        table.header.update("IDREJECT", "%s %s" %
                            (first_record.fields["low_reject"],
                             first_record.fields["high_reject"]))
        table.header.update("IDGROW", first_record.fields["grow"])
        table.header.update("IDRANGE", "%s %s" %
                            (first_record.mrange[0], first_record.mrange[1]))

        # Store fitcoords information in the header
        fc_record = self.fitcoords_database
        table.header.update("FCUNITS", fc_record.fields["units"])
        table.header.update("FCAXIS", fc_record.fields["axis"])
        table.header.update("FCFUNCTN", fc_record.modelname)
        table.header.update("FCXORDER", fc_record.xorder)
        table.header.update("FCYORDER", fc_record.yorder)
        table.header.update("FCXRANGE", "%s %s" %
                            (fc_record.xbounds[0], fc_record.xbounds[1]))
        table.header.update("FCYRANGE", "%s %s" %
                            (fc_record.ybounds[0], fc_record.ybounds[1]))
        for i in range(len(fc_record.coeff)):
            coeff = fc_record.coeff[i]
            table.header.update("FCCOEF%d" % i, coeff)
####here -- comments

        return table
