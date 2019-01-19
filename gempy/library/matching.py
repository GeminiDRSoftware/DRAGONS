import numpy as np
from astropy.modeling.fitting import (_validate_model,
                                      _fitter_to_model_params,
                                      _model_to_fit_params, Fitter,
                                      _convert_input)
from astropy.wcs import WCS

from scipy import optimize, spatial
from datetime import datetime

from gempy.gemini import gemini_tools as gt
from ..utils import logutils

from .transform import Transform
from .astromodels import Pix2Sky

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
    #print updated_model.x_offset.value, updated_model.y_offset.value, sum
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
    dist, idx = tree.query(list(zip(xt, yt)), k=5, distance_upper_bound=maxsep)
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

        tree = spatial.cKDTree(list(zip(*ref_coords)))
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
        try:
            iter(coords[0])
        except TypeError:
            coords = (coords,)

        landscape = np.zeros(landshape)
        hw = int(maxsig * sigma)
        grid = np.meshgrid(*[np.arange(0, hw*2+1)]*landscape.ndim)
        rsq = np.sum((ax - hw)**2 for ax in grid)
        mountain = np.exp(-0.5 * rsq / (sigma * sigma))

        # Place a mountain onto the landscape for each coord in coords
        # Need to crop at edges if mountain extends beyond landscape
        for coord in zip(*coords):
            lslice = []
            mslice = []
            for pos, length in zip(coord[::-1], landshape):
                l1, l2 = int(pos-0.5)-hw, int(pos-0.5)+hw+1
                m1, m2 = 0, hw*2+1
                if l2 < 0 or l1 >= length:
                    break
                if l1 < 0:
                    m1 -= l1
                    l1 = 0
                if l2 > length:
                    m2 -= (l2 - length)
                    l2 = length
                lslice.append(slice(l1, l2))
                mslice.append(slice(m1, m2))
            else:
                landscape[tuple(lslice)] += mountain[tuple(mslice)]
        return landscape

    def __call__(self, model, in_coords, ref_coords, sigma=5.0, maxsig=4.0,
                 **kwargs):
        model_copy = _validate_model(model, ['bounds'])
        x, y = in_coords
        xref, yref = ref_coords
        xmax = max(np.max(x), np.max(xref))
        ymax = max(np.max(y), np.max(yref))
        landscape = self.mklandscape(ref_coords, sigma, maxsig,
                                    (int(ymax),int(xmax)))
        farg = (model_copy,) + _convert_input(x, y, landscape)
        p0, _ = _model_to_fit_params(model_copy)

        # TODO: Use the name of the parameter to infer the step size
        ranges = []
        for p in model_copy.param_names:
            bounds = model_copy.bounds[p]
            try:
                diff = np.diff(bounds)[0]
            except TypeError:
                pass
            else:
                if diff > 0:
                    ranges.append(slice(*(bounds+(min(0.5*sigma, 0.1*diff),))))
                    continue
            ranges.append((getattr(model_copy, p).value,) * 2)

        # Ns=1 limits the fitting along an axis where the range is not a slice
        # object: this is those were the bounds are equal (i.e. fixed param)
        fitted_params = self._opt_method(self.objective_function, ranges,
                                         farg, Ns=1, finish=None, **kwargs)
        _fitter_to_model_params(model_copy, fitted_params)
        return model_copy

def fit_model(model, xin, xout, sigma=5.0, tolerance=1e-8, brute=True,
              release=False, verbose=True):
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
    brute: bool
        perform brute-force fit first?
    release: boolean
        undo the parameter bounds for the simplex fit?
    verbose: boolean
        output model and time info?

    Returns
    -------
    Model: the best-fitting mapping from xin -> xout
    """
    log = logutils.get_logger(__name__)
    start = datetime.now()

    if brute:
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
            try:
                if np.diff(getattr(model, p).bounds)[0] == 0:
                    getattr(m, p).fixed = True
                    continue
            except TypeError:
                pass
            if release:
                getattr(m, p).bounds = (None, None)
    else:
        m = model.copy()

    # More precise minimization using pairwise calculations
    fit_it = KDTreeFitter()
    # We don't care about how much the function value changes (ftol), only
    # that the position is robust (xtol)
    final_model = fit_it(m, xin, xout, method='Nelder-Mead',
                     options={'xtol': tolerance})
    if verbose:
        log.stdinfo(_show_model(final_model, "Final model in {:.2f} seconds".
                                format((datetime.now() - start).total_seconds())))
    return final_model

def _show_model(model, intro=""):
    """Provide formatted output of a (possibly compound) transformation"""
    model_str = "{}\n".format(intro) if intro else ""
    try:
        iterator = iter(model)
    except TypeError:
        iterator = [model]
    # Only display parameters of those models that have names
    for m in iterator:
        if m.name is not None:
            for param in [getattr(m, name) for name in m.param_names]:
                if not (param.fixed or (param.bounds[0] == param.bounds[1]
                                        and param.bounds[0] is not None)):
                    model_str += "{}: {}\n".format(param.name, param.value)
    return model_str


def align_catalogs(incoords, refcoords, transform=None, tolerance=0.1):
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

    Returns
    -------
    Model: a model that maps (xin,yin) to (xref,yref)
    """
    if transform is None:
        if len(incoords) == 2:
            transform = Transform.create2d()
        else:
            raise ValueError("No transformation provided and data are not 2D")

    final_model = fit_model(transform.asModel(), incoords, refcoords,
                               sigma=10.0, tolerance=tolerance, brute=True)
    transform.replace(final_model)
    return transform

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
    tree = spatial.cKDTree(list(zip(*refcoords)))
    dist, idx = tree.query(list(zip(*incoords)), distance_upper_bound=radius)
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

def align_images_from_wcs(adinput, adref, transform=None, cull_sources=False,
                          min_sources=1, search_radius=10, rotate=False,
                          scale=False, full_wcs=False, return_matches=False):
    """
    This function takes two images (an input image, and a reference image) and
    works out the modifications needed to the WCS of the input images so that
    the world coordinates of its OBJCAT sources match the world coordinates of
    the OBJCAT sources in the reference image. This is done by modifying the
    WCS of the input image and mapping the reference image sources to pixels
    in the input image via the reference image WCS (fixed) and the input image
    WCS. As such, in the nomenclature of the fitting routines, the pixel
    positions of the input image's OBJCAT become the "reference" sources,
    while the converted positions of the reference image's OBJCAT are the
    "input" sources.

    Parameters
    ----------
    adinput: AstroData
        input AD whose pixel shift is requested
    adref: AstroData
        reference AD image
    transform: Transform/None
        existing transformation (if None, will do brute search)
    cull_sources: bool
        limit matched sources to "good" (i.e., stellar) objects
    min_sources: int
        minimum number of sources to use for cross-correlation
    search_radius: float
        size of search box (in arcseconds)
    rotate: bool
        add a rotation to the alignment transform?
    scale: bool
        add a magnification to the alignment transform?
    full_wcs: bool
        use recomputed WCS at each iteration, rather than modify the positions
        in pixel space?
    return_matches: bool
        return a list of matched objects as well as the Transform?

    Returns
    -------
    matches: 2 lists
        OBJCAT sources in input and reference that are matched
    WCS: new WCS for input image
    """
    log = logutils.get_logger(__name__)
    if len(adinput) * len(adref) != 1:
        log.warning('Can only match single-extension images')
        return None

    try:
        input_objcat = adinput[0].OBJCAT
        ref_objcat = adref[0].OBJCAT
    except AttributeError:
        log.warning('Both input images must have object catalogs')
        return None

    if len(input_objcat) < min_sources or len(ref_objcat) < min_sources:
        log.warning("Too few sources in one or both images. Cannot align.")
        return None

    # OK, we can proceed
    incoords = (input_objcat['X_IMAGE'], input_objcat['Y_IMAGE'])
    refcoords = (ref_objcat['X_IMAGE'], ref_objcat['Y_IMAGE'])
    if cull_sources:
        good_src1 = gt.clip_sources(adinput)[0]
        good_src2 = gt.clip_sources(adref)[0]
        if len(good_src1) < min_sources or len(good_src2) < min_sources:
            log.warning("Too few sources in culled list, using full set "
                        "of sources")
        else:
            incoords = (good_src1["x"], good_src1["y"])
            refcoords = (good_src2["x"], good_src2["y"])

    # Nomenclature here is that the ref_transform transforms the reference
    # OBJCAT coords immutably, and then the fit_transform is the fittable
    # thing we're trying to get to map those to the input OBJCAT coords
    ref_transform = Transform(Pix2Sky(WCS(adref[0].hdr)))
    #pixel_range = max(adinput[0].data.shape)
    if full_wcs:
        #fit_transform = Transform(Pix2Sky(WCS(adinput[0].hdr), factor=pixel_range, factor_scale=pixel_range, angle_scale=pixel_range/57.3).inverse)
        fit_transform = Transform(Pix2Sky(WCS(adinput[0].hdr)).rename('WCS').inverse)
        fit_transform.angle.fixed = not rotate
        fit_transform.factor.fixed = not scale
    else:
        ref_transform.append(Pix2Sky(WCS(adinput[0].hdr)).inverse)
        fit_transform = Transform.create2d(translation=(0, 0),
                                           rotation=0 if rotate else None,
                                           magnification=1 if scale else None,
                                           shape=adinput[0].shape)

    # Copy parameters across. We don't simply start with the provided
    # transform, because we may be moving from a geometric one to WCS
    if transform is not None:
        for param in fit_transform.param_names:
            if param in transform.param_names:
                setattr(fit_transform, param, getattr(transform, param))
    fit_transform.add_bounds('x_offset', search_radius)
    fit_transform.add_bounds('y_offset', search_radius)
    if rotate:
        fit_transform.add_bounds('angle', 5.)
    if scale:
        fit_transform.add_bounds('factor', 0.05)

    # Do the fit, and update the transform with the fitted parameters
    transformed_ref_coords = ref_transform(*refcoords)
    refine = (transform is not None)
    fitted_model = fit_model(fit_transform.asModel(), transformed_ref_coords,
                             incoords, sigma=10, tolerance=1e-6, brute=not refine)
    fit_transform.replace(fitted_model)

    if return_matches:
        matched = match_sources(fitted_model(*transformed_ref_coords), incoords,
                                radius=1.0)
        ind2 = np.where(matched >= 0)
        ind1 = matched[ind2]
        obj_list = [[], []] if len(ind1) < 1 else [np.array(list(zip(*incoords)))[ind1],
                                                   np.array(list(zip(*refcoords)))[ind2]]
        return obj_list, fit_transform

    return fit_transform
