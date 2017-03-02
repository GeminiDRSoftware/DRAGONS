import numpy as np
from astropy.modeling.fitting import (_validate_model,
                                      _fitter_to_model_params,
                                      _model_to_fit_params, Fitter,
                                      _convert_input)
from scipy import optimize, spatial
from datetime import datetime

def match_sources(incoords, refcoords, radius=2.0, priority=[]):
    """
    Match two sets of sources that are on the same reference frame. In general
    the closest match will be used, but there can be a priority list that will
    take precedence.

    Parameters
    ----------
    incoords: 2xN array
        input source coords (transformed to reference frame)
    refcoords: 2xM array
        reference source coords
    radius:
        maximum separation for a match
    priority: list of ints
        items in incoords that should have priority, even if a closer
        match is found

    Returns
    -------
    int array of length N:
        index of matched sources in the reference list (-1 means no match)
    """
    matched = np.full((len(incoords[0]),), -1, dtype=int)
    tree = spatial.cKDTree(zip(*refcoords))
    dist, idx = tree.query(zip(*incoords), distance_upper_bound=radius)
    for i in range(len(refcoords[0])):
        inidx = np.where(idx==i)[0][np.argsort(dist[np.where(idx==i)])]
        for ii in inidx:
            if ii in priority:
                matched[ii] = i
                break
        else:
            # No first_allowed so take the first one
            if len(inidx):
                matched[inidx[0]] = i
    return matched

def _landstat(landscape, updated_model, x, y):
    """
    Compute the statistic for transforming coordinates onto an existing
    "landscape" of "mountains" representing source positions. Since the
    landscape is an array and therefore pixellated, the precision is limited.

    Parameters
    ----------
    landscape: 2D array
        synthetic image representing locations of sources in reference plane
    updated_model: Model
        transformation (input -> reference) being investigated
    x, y: float arrays
        input x, y coordinates

    Returns
    -------
    float:
        statistic representing quality of fit to be minimized
    """
    xt, yt = updated_model(x, y)
    sum = np.sum(landscape[iy,ix] for ix,iy in zip((xt-0.5).astype(int),
                                                   (yt-0.5).astype(int))
                  if ix>=0 and iy>=0 and ix<landscape.shape[1]
                                     and iy<landscape.shape[0])
    return -sum  # to minimize

def _stat(tree, updated_model, x, y, sigma, maxsig):
    """
    Compute the statistic for transforming coordinates onto a set of reference
    coordinates. This uses mathematical calulations and is not pixellated like
    the landscape-array methods.

    Parameters
    ----------
    tree: KDTree
        a KDTree made from the reference coordinates
    updated_model: Model
        transformation (input -> reference) being investigated
    x, y: float arrays
        input x, y coordinates
    sigma: float
        standard deviation of Gaussian (in pixels) used to represent each source
    maxsig: float
        maximum number of standard deviations of Gaussian extent

    Returns
    -------
    float:
        statistic representing quality of fit to be minimized
    """
    f = 0.5/(sigma*sigma)
    maxsep = maxsig*sigma
    xt, yt = updated_model(x, y)
    #start = datetime.now()
    dist, idx = tree.query(zip(xt, yt), k=5, distance_upper_bound=maxsep)
    sum = np.sum(np.exp(-f*d*d) for dd in dist for d in dd)
    #print (datetime.now()-start).total_seconds(), updated_model.offset_0.value, updated_model.offset_1.value, sum
    return -sum  # to minimize

class KDTreeFitter(Fitter):
    """
    Fitter class that uses minimization (the method can be passed as a
    parameter to the instance) to determine the transformation to map a set
    of input coordinates to a set of reference coordinates.
    """
    def __init__(self):
        super(KDTreeFitter, self).__init__(optimize.minimize,
                                             statistic=_stat)

    def __call__(self, model, x, y, ref_coords, sigma=5.0, maxsig=4.0,
                 **kwargs):
        model_copy = _validate_model(model, ['bounds'])

        # Starting simplex step size is set to be 5% of parameter values
        # Need to ensure this is larger than the convergence tolerance
        # so move the initial values away from zero if necessary
        try:
            xtol = kwargs['options']['xtol']
        except KeyError:
            pass
        else:
            for p in model_copy.param_names:
                pval = getattr(model_copy, p).value
                if abs(pval) < xtol / 0.05:
                    getattr(model_copy, p).value = np.sign(pval) * xtol / 0.049

        tree = spatial.cKDTree(zip(*ref_coords))
        # avoid _convert_input since tree can't be coerced to a float
        farg = (model_copy, x, y, sigma, maxsig, tree)
        p0, _ = _model_to_fit_params(model_copy)

        result = self._opt_method(self.objective_function, p0, farg,
                                  **kwargs)
        fitted_params = result['x']
        _fitter_to_model_params(model_copy, fitted_params)
        return model_copy

class BruteLandscapeFitter(Fitter):
    """
    Fitter class that employs brute-force optimization to map a set of input
    coordinates onto a set of reference coordinates by cross-correlation
    over a "landscape" of "mountains" representing the reference coords
    """
    def __init__(self):
        super(BruteLandscapeFitter, self).__init__(optimize.brute,
                                              statistic=_landstat)

    def mklandscape(self, coords, sigma, maxsig, landshape):
        landscape = np.zeros(landshape)
        lysize, lxsize = landscape.shape
        hw = int(maxsig * sigma)
        xgrid, ygrid = np.mgrid[0:hw * 2 + 1, 0:hw * 2 + 1]
        rsq = (ygrid - hw) ** 2 + (xgrid - hw) ** 2
        mountain = np.exp(-0.5 * rsq / (sigma * sigma))
        for x, y in zip(*coords):
            mx1, mx2, my1, my2 = 0, hw * 2 + 1, 0, hw * 2 + 1
            lx1, lx2 = int(x - 0.5) - hw, int(x - 0.5) + hw + 1
            ly1, ly2 = int(y - 0.5) - hw, int(y - 0.5) + hw + 1
            if lx2 < 0 or lx1 >= lxsize or ly2 < 0 or ly1 >= lysize:
                continue
            if lx1 < 0:
                mx1 -= lx1
                lx1 = 0
            if lx2 > lxsize:
                mx2 -= (lx2 - lxsize)
                lx2 = lxsize
            if ly1 < 0:
                my1 -= ly1
                ly1 = 0
            if ly2 > lysize:
                my2 -= (ly2 - lysize)
                ly2 = lysize
            try:
                landscape[ly1:ly2, lx1:lx2] += mountain[my1:my2, mx1:mx2]
            except ValueError:
                print(y, x, landscape.shape)
                print(ly1, ly2, lx1, lx2)
                print(my1, my2, mx1, mx2)
        return landscape

    def __call__(self, model, x, y, ref_coords, sigma=5.0, maxsig=4.0,
                 **kwargs):
        model_copy = _validate_model(model, ['bounds'])
        landscape = self.mklandscape(ref_coords, sigma, maxsig,
                                    (int(np.max(y)),int(np.max(x))))
        farg = (model_copy,) + _convert_input(x, y, landscape)
        p0, _ = _model_to_fit_params(model_copy)

        # TODO: Use the name of the parameter to infer the step size
        ranges = [slice(*(model_copy.bounds[p]+(min(sigma, 0.1*np.diff(model_copy.bounds[p])[0]),)))
                  for p in model_copy.param_names]

        fitted_params = self._opt_method(self.objective_function,
                                         ranges, farg, finish=None, **kwargs)
        _fitter_to_model_params(model_copy, fitted_params)
        return model_copy

def find_offsets(xin, yin, xref, yref, range=(-300,300), subpix=1,
                 sigma=10.0, maxsig=4.0):
    """
    Identify the translational offsets that map (xin, yin) to (xref, yref)
    by cross-correlation

    Parameters
    ----------
    xin, yin: float array
        Input coordinates
    xref, yref: float array
        Reference coordinates
    range: int, int
        range over which to search for offsets
    subpix: int
        factor by which to subdivide input pixels
    sigma: float
        rms of Gaussian mountain in input pixels
    maxsig: float
        size of Gaussian mountain in standard deviations

    Returns
    -------
    int, int
        x and y offsets
    """
    hw = int(maxsig * sigma * subpix)
    xgrid, ygrid = np.mgrid[0:hw * 2 + 1, 0:hw * 2 + 1]
    rsq = (ygrid - hw) ** 2 + (xgrid - hw) ** 2
    midrange = 0.5*(range[0]+range[1])
    cpix = int((midrange - range[0]) * subpix)
    size = cpix * 2 + 1
    landscape = np.zeros((size, size))
    mountain = np.exp(-0.5 * rsq / (sigma * sigma))
    xoff = np.array([xr-xi for xr in xref for xi in xin])
    yoff = np.array([yr-yi for yr in yref for yi in yin])

    x1 = (subpix*(xoff-midrange) + cpix - hw).astype(int)
    x2 = x1 + hw*2+1
    y1 = (subpix*(yoff-midrange) + cpix - hw).astype(int)
    y2 = y1 + hw*2+1
    for xx1, xx2, yy1, yy2 in zip(x1, x2, y1, y2):
        if xx1>=0 and xx2<size and yy1>=0 and yy2<size:
            landscape[yy1:yy2, xx1:xx2] += mountain

    y, x = np.unravel_index(np.argmax(landscape), landscape.shape)
    y = float(y-cpix) / subpix
    x = float(x-cpix) / subpix
    return x, y