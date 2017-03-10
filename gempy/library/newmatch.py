import numpy as np
import math
from astropy.modeling.fitting import (_validate_model,
                                      _fitter_to_model_params,
                                      _model_to_fit_params, Fitter,
                                      _convert_input)
from astropy.modeling import models, FittableModel, Parameter

from scipy import optimize, spatial
from datetime import datetime

from ..utils import logutils

class Shift2D(FittableModel):
    """2D translation"""
    inputs = ('x', 'y')
    outputs = ('x', 'y')
    x_offset = Parameter(default=0.0)
    y_offset = Parameter(default=0.0)

    @property
    def inverse(self):
        inv = self.copy()
        inv.x_offset = -self.x_offset
        inv.y_offset = -self.y_offset
        return inv

    @staticmethod
    def evaluate(x, y, x_offset, y_offset):
        return x+x_offset, y+y_offset

class Scale2D(FittableModel):
    """2D scaling"""
    def __init__(self, factor, param_scale=1.0, **kwargs):
        self._param_scale = param_scale
        super(Scale2D, self).__init__(factor, **kwargs)

    inputs = ('x', 'y')
    outputs = ('x', 'y')
    factor = Parameter(default=1.0)

    @property
    def inverse(self):
        inv = self.copy()
        inv.factor = self._param_scale**2/self.factor
        return inv

    def evaluate(self, x, y, factor):
        return x*factor/self._param_scale, y*factor/self._param_scale

class Rotate2D(FittableModel):
    """Rotation; Rotation2D isn't fittable"""
    def __init__(self, factor, param_scale=1.0, **kwargs):
        self._param_scale = param_scale
        super(Rotate2D, self).__init__(factor, **kwargs)

    inputs = ('x', 'y')
    outputs = ('x', 'y')
    angle = Parameter(default=0.0, getter=np.rad2deg, setter=np.deg2rad)

    @property
    def inverse(self):
        inv = self.copy()
        inv.angle = -self.angle
        return inv

    def evaluate(self, x, y, angle):
        if x.shape != y.shape:
            raise ValueError("Expected input arrays to have the same shape")
        orig_shape = x.shape or (1,)
        inarr = np.array([x.flatten(), y.flatten()])
        s, c = math.sin(angle/self._param_scale), math.cos(angle/self._param_scale)
        x, y = np.dot(np.array([[c, -s], [s, c]], dtype=np.float64), inarr)
        x.shape = y.shape = orig_shape
        return x, y

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
    start = datetime.now()
    dist, idx = tree.query(zip(xt, yt), k=5, distance_upper_bound=maxsep)
    sum = np.sum(np.exp(-f*d*d) for dd in dist for d in dd)
    #print (datetime.now()-start).total_seconds(), updated_model.parameters, sum
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

    def __call__(self, model, in_coords, ref_coords, sigma=5.0, maxsig=4.0,
                 **kwargs):
        model_copy = _validate_model(model, ['bounds', 'fixed'])

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
                if abs(pval) < 20*xtol and 'offset' in p:
                    getattr(model_copy, p).value = 20*xtol if pval == 0 \
                        else (np.sign(pval) * 20*xtol)

        tree = spatial.cKDTree(zip(*ref_coords))
        # avoid _convert_input since tree can't be coerced to a float
        x, y = in_coords
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
        """
        Populates an array with Gaussian mountains at specified coordinates.
        Used to allow rapid goodness-of-fit calculations for cross-correlation.

        Parameters
        ----------
        coords: 2xN float array
            coordinates of sources
        sigma: float
            standard deviation of Gaussian in pixels
        maxsig: float
            extent (in standard deviations) of each Gaussian
        landshape: 2-tuple
            shape of array

        Returns
        -------
        float array:
            the "landscape", populated by "mountains"
        """
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

    def __call__(self, model, in_coords, ref_coords, sigma=5.0, maxsig=4.0,
                 **kwargs):
        model_copy = _validate_model(model, ['bounds'])
        x, y = in_coords
        landscape = self.mklandscape(ref_coords, sigma, maxsig,
                                    (int(np.max(y)),int(np.max(x))))
        farg = (model_copy,) + _convert_input(x, y, landscape)
        p0, _ = _model_to_fit_params(model_copy)

        # TODO: Use the name of the parameter to infer the step size
        bounds = [model_copy.bounds[p] for p in model_copy.param_names]
        ranges = [slice(*(b[p]+(min(0.5*sigma, 0.1*np.diff(b[p])[0]),)))
                  if b[p] is not None and np.diff(b[p])[0] > 0
                  else b[p] for b in bounds]
        # Ns=1 limits the fitting along an axis where the range is not a slice
        # object: this is those were the bounds are equal (i.e. fixed param)
        fitted_params = self._opt_method(self.objective_function, ranges,
                                         farg, Ns=1, finish=None, **kwargs)
        _fitter_to_model_params(model_copy, fitted_params)
        return model_copy

def fit_brute_then_simplex(model, xin, xout, sigma=5.0, tolerance=0.001,
                           verbose=True):
    """
    Finds the best-fitting mapping to convert from xin to xout, using a
    two-step approach by first doing a brute-force scan of parameter space,
    and then doing simplex fitting from this starting position.
    Handles a fixed parameter by setting the bounds equal to the value.

    Parameters
    ----------
    model: FittableModel
        initial model guess
    xin: array-like
        input coordinates
    xout: array-like
        coordinates to fit to
    sigma: float
        size of the mountains for the BruteLandscapeFitter
    tolerance: float
        accuracy of parameters in final answer
    verbose: boolean
        output model and time info?

    Returns
    -------
    Model: the best-fitting mapping from xin -> xout
    """
    log = logutils.get_logger(__name__)
    start = datetime.now()

    # Since optimize.brute can't handle "fixed" parameters, we have to unfix
    # them and control things by setting the bounds to a zero-width interval
    for p in model.param_names:
        pval = getattr(model, p).value
        if getattr(model, p).fixed:
            getattr(model, p).bounds = (pval, pval)
            getattr(model, p).fixed = False

    # Brute-force grid search using an image landscape
    fit_it = BruteLandscapeFitter()
    m = fit_it(model, xin, xout, sigma=sigma)
    if verbose:
        log.stdinfo(_show_model(m, "Coarse model in {:.2f} seconds".
                                format((datetime.now() - start).total_seconds())))

    # Re-fix parameters in the intermediate model, if they were fixed
    # in the original model
    for p in m.param_names:
        if np.diff(getattr(model, p).bounds)[0] == 0:
            getattr(m, p).fixed = True

    # More precise minimization using pairwise calculations
    fit_it = KDTreeFitter()
    # We don't care about how much the function value changes (ftol), only
    # that the position is robust (xtol)
    final_model = fit_it(m, xin, xout, method='Nelder-Mead',
                     options={'xtol': tolerance, 'ftol': 100.0})
    if verbose:
        log.stdinfo(_show_model(final_model, "Final model in {:.2f} seconds".
                                format((datetime.now() - start).total_seconds())))
    return final_model

def fit_brute_repeatedly(model, xin, xout, sigma=5.0, tolerance=0.001,
                           reduction=0.5, verbose=True):
    """
    Finds the best-fitting mapping to convert from xin to xout, using repeated
    brute-force fitting over decreasing regions of parameter space

    Parameters
    ----------
    model: FittableModel
        initial model guess
    xin: array-like
        input coordinates
    xout: array-like
        coordinates to fit to
    sigma: float
        size of the mountains for the BruteLandscapeFitter
    tolerance: float
        accuracy of parameters in final answer
    reduction: float
        factor by which to reduce search size at each step
    verbose: boolean
        output model and time info?

    Returns
    -------
    Model: the best-fitting mapping from xin -> xout
    """
    log = logutils.get_logger(__name__)

    start = datetime.now()
    # Since optimize.brute can't handle "fixed" parameters, we have to unfix
    # them and control things by setting the bounds to a zero-width interval
    for p in model.param_names:
        param = getattr(model, p)
        if param.fixed:
            param.bounds = (param.value, param.value)
            param.fixed = False

    ranges = np.array([np.diff(getattr(model, p).bounds)[0]
                       for p in model.param_names])
    m = model.copy()
    while np.max(ranges) > tolerance:
        # Brute-force grid search using an image landscape
        fit_it = BruteLandscapeFitter()
        m = fit_it(m, xin, xout, sigma=sigma)
        if verbose:
            log.stdinfo(_show_model(m, "Model in {:.2f} seconds".
                        format((datetime.now() - start).total_seconds())))
        ranges = reduction * ranges
        for p, range in zip(m.param_names, ranges):
            param = getattr(m, p)
            param.bounds = (param.value-0.5*range, param.value+0.5*range)

    return m

def _show_model(model, intro=""):
    """Provide formatted output of a (possibly compound) transformation"""
    model_str = "{}\n".format(intro) if intro else ""
    try:
        iterator = iter(model)
    except TypeError:
        iterator = [model]
    # We don't want to show the centering model (or its inverse), and we want
    # to scale the model parameters to their internally-stored values
    for m in iterator:
        if m.name != 'Centering':
            pscale = m._param_scale if hasattr(m, '_param_scale') else 1.0
            for name, value in zip(m.param_names, m.parameters):
                model_str += "{}: {}\n".format(name, value/pscale)
    return model_str


def align_catalogs(xin, yin, xref, yref, model_guess=None,
                   translation=None, translation_range=None,
                   rotation=None, rotation_range=None,
                   magnification=None, magnification_range=None,
                   tolerance=0.1, center_of_field=None, simplex=True):
    """
    Generic interface for a 2D catalog match. Either an initial model guess
    is provided, or a model will be created using a combination of
    translation, rotation, and magnification, as requested. Only those
    transformations for which a *range* is specified will be used. In order
    to keep the translation close to zero, the rotation and magnification
    are performed around the centre of the field, which can either be provided
    -- as (x,y) in 1-based pixels -- or will be determined from the mid-range
    of the x and y input coordinates.

    Parameters
    ----------
    xin, yin: float arrays
        input coordinates
    xref, yref: float arrays
        reference coordinates to map and match to
    model_guess: Model
        initial model guess (overrides the next parameters)
    translation: 2-tuple of floats
        initial translation guess
    translation_range: None, value, 2-tuple or 2x2-tuple
        None => fixed
        value => search range from initial guess (same for x and y)
        2-tuple => search limits (same for x and y)
        2x2-tuple => search limits for x and y
    rotation: float
        initial rotation guess (degrees)
    rotation_range: None, float, or 2-tuple
        extent of search space for rotation
    magnification: float
        initial magnification factor
    magnification_range: None, float, or 2-tuple
        extent of search space for magnification
    tolerance: float
        accuracy required for final result
    center_of_field: 2-tuple
        rotation and magnification have no effect at this location
         (if None, uses middle of xin,yin ranges)
    simplex: boolean
        use a single brute-force iteration and then a simplex?

    Returns
    -------
    Model: a model that maps (xin,yin) to (xref,yref)
    """
    def _get_value_and_range(value, range):
        """Converts inputs to a central value and a range tuple"""
        try:
            r1, r2 = range
        except TypeError:
            r1, r2 = range, None
        except ValueError:
            r1, r2 = None, None
        if value is not None:
            if r1 is not None and r2 is not None:
                if r1 <= value <= r2:
                    return value, (r1, r2)
                else:
                    extent = 0.5*abs(r2-r1)
                    return value, (value-extent, value+extent)
            elif r1 is not None:
                return value, (value-r1, value+r1)
            else:
                return value, None
        elif r1 is not None:
            if r2 is None:
                return 0.0, (-r1, r1)
            else:
                return 0.5*(r1+r2), (r1, r2)
        else:
            return None, None

    log = logutils.get_logger(__name__)
    if model_guess is None:
        # Some useful numbers for later
        x1, x2 = np.min(xin), np.max(xin)
        y1, y2 = np.min(yin), np.max(yin)
        pixel_range = 0.5*max(x2-x1, y2-y1)

        # Set up translation part of the model
        if hasattr(translation, '__len__'):
            xoff, yoff = translation
        else:
            xoff, yoff = translation, translation
        trange = np.array(translation_range)
        if len(trange.shape) == 2:
            xvalue, xrange = _get_value_and_range(xoff, trange[0])
            yvalue, yrange = _get_value_and_range(yoff, trange[1])
        else:
            xvalue, xrange = _get_value_and_range(xoff, translation_range)
            yvalue, yrange = _get_value_and_range(yoff, translation_range)
        if xvalue is None or yvalue is None:
            trans_model = None
        else:
            trans_model = Shift2D(xvalue, yvalue)
            if xrange is None:
                trans_model.x_offset.fixed = True
            else:
                trans_model.x_offset.bounds = xrange
            if yrange is None:
                trans_model.y_offset.fixed = True
            else:
                trans_model.y_offset.bounds = yrange

        # Set up rotation part of the model
        rvalue, rrange = _get_value_and_range(rotation, rotation_range)
        if rvalue is None:
            rot_model = None
        else:
            # Getting the rotation wrong by da (degrees) will cause a shift of
            # da/57.3*pixel_range at the edge of the data, so we want
            # da=tolerance*57.3/pixel_range
            rot_scaling = pixel_range / 57.3
            rot_model = Rotate2D(rvalue*rot_scaling, param_scale=rot_scaling)
            if rrange is None:
                rot_model.angle.fixed = True
            else:
                rot_model.angle.bounds = tuple(x*rot_scaling for x in rrange)

        # Set up magnification part of the model
        mvalue, mrange = _get_value_and_range(magnification, magnification_range)
        if mvalue is None:
            mag_model = None
        else:
            # Getting the magnification wrong by dm will cause a shift of
            # dm*pixel_range at the edge of the data, so we want
            # dm=tolerance/pixel_range
            mag_scaling = pixel_range
            mag_model = Scale2D(mvalue*mag_scaling, param_scale=mag_scaling)
            if mrange is None:
                mag_model.factor.fixed = True
            else:
                mag_model.factor.bounds = tuple(x*mag_scaling for x in mrange)

        # Make the compound model
        if rot_model is None and mag_model is None:
            if trans_model is None:
                return models.Identity(2)  # Nothing to do
            else:
                init_model = trans_model  # Don't need center of field
        else:
            if center_of_field is None:
                center_of_field = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
                log.debug('No center of field given, using x={:.2f} '
                          'y={:.2f}'.format(*center_of_field))
            restore = Shift2D(*center_of_field).rename('Centering')
            restore.x_offset.fixed = True
            restore.y_offset.fixed = True

            init_model = restore.inverse
            if trans_model is not None:
                init_model |= trans_model
            if rot_model is not None:
                init_model |= rot_model
            if mag_model is not None:
                init_model |= mag_model
            init_model |= restore
    elif model_guess.fittable:
        init_model = model_guess
    else:
        log.warning('The transformation is not fittable!')
        return models.Identity(2)

    fit_function = fit_brute_then_simplex if simplex else fit_brute_repeatedly
    final_model = fit_function(init_model, (xin, yin), (xref, yref),
                               sigma=10.0, tolerance=tolerance)
    return final_model

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

def match_catalogs(xin, yin, xref, yref, use_in=None, use_ref=None,
                   model_guess=None, translation=None,
                   translation_range=None, rotation=None,
                   rotation_range=None, magnification=None,
                   magnification_range=None, tolerance=0.1,
                   center_of_field=None, simplex=True, match_radius=1.0):
    """
    Aligns catalogs with align_catalogs(), and then matches sources with
    match_sources()

    Parameters
    ----------
    xin, yin: float arrays
        input coordinates
    xref, yref: float arrays
        reference coordinates to map and match to
    use_in: list/None
        only use these input sources for matching (None => all)
    use_ref: list/None
        only use these reference sources for matching (None => all)
    model_guess: Model
        initial model guess (overrides the next parameters)
    translation: 2-tuple of floats
        initial translation guess
    translation_range: value, 2-tuple or 2x2-tuple
        value => search range from initial guess (same for x and y)
        2-tuple => search limits (same for x and y)
        2x2-tuple => search limits for x and y
    rotation: float
        initial rotation guess (degrees)
    rotation_range: float or 2-tuple
        extent of search space for rotation
    magnification: float
        initial magnification factor
    magnification_range: float or 2-tuple
        extent of search space for magnification
    tolerance: float
        accuracy required for final result
    center_of_field: 2-tuple
        rotation and magnification have no effect at this location
         (if None, uses middle of xin,yin ranges)
    simplex: boolean
        use a single brute-force iteration and then a simplex?

    Returns
    -------
    int array of length N:
        index of matched sources in the reference list (-1 means no match)
    Model:
        best-fitting alignment model
    """
    if use_in is None:
        use_in = list(range(len(xin)))
    if use_ref is None:
        use_ref = list(range(len(xref)))
    model = align_catalogs(xin[use_in], yin[use_in], xref[use_ref], yref[use_ref],
                           model_guess=model_guess, translation=translation,
                           translation_range=translation_range, rotation=rotation,
                           rotation_range=rotation_range, magnification=magnification,
                           magnification_range=magnification_range, tolerance=tolerance,
                           center_of_field=center_of_field, simplex=simplex)
    matched = match_sources(model(xin, yin), (xref, yref), radius=match_radius,
                               priority=use_in)
    return matched, model