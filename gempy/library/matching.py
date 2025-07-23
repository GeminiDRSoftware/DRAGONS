# Copyright(c) 2017-2025 Association of Universities for Research in Astronomy, Inc.

import numpy as np
from datetime import datetime
from functools import partial, reduce
import inspect

from scipy import optimize, spatial
from astropy.modeling import fitting, models, Model, FittableModel
from astropy.modeling.fitting import (_validate_constraints,
                                      _validate_model,
                                      Fitter)
try:
    # New public API in AstroPy 5.1:
    from astropy.modeling.fitting import (fitter_to_model_params,
                                          model_to_fit_params)
except ImportError:  # pragma: no cover
    # Earlier private API: The second of these returns 2 values instead of 3 in
    # the new version; we could append the correct model bounds here by copying
    # a small amount of code from AstroPy 5.1+, but the code below discards
    # return values after the first one anyway:
    from astropy.modeling.fitting import (
        _fitter_to_model_params as fitter_to_model_params,
        _model_to_fit_params as model_to_fit_params
    )

from astrodata import wcs as adwcs

try:
    from gempy.library import cython_utils
except ImportError:  # pragma: no cover
    raise ImportError("Run 'cythonize -i cython_utils.pyx' in gempy/library")
from ..utils import logutils
from .astromodels import Rotate2D, Scale2D


class BruteLandscapeFitter(Fitter):
    """
    Fitter class that employs brute-force optimization to map a set of input
    coordinates onto a set of reference coordinates by cross-correlation
    over a "landscape" of "mountains" representing the reference coords
    """

    supported_constraints = ['bounds', 'fixed']

    def __init__(self):
        super().__init__(optimize.brute, statistic=self._landstat)

    def _landstat(self, landscape, updated_model, in_coords):
        """
        Compute the statistic for transforming coordinates onto an existing
        "landscape" of "mountains" representing source positions. Since the
        landscape is an array and therefore pixellated, the precision is limited.

        Parameters
        ----------
        landscape: nD array
            synthetic image representing locations of sources in reference plane
        updated_model: Model
            transformation (input -> reference) being investigated
        in_coords: nD array
            input coordinates

        Returns
        -------
        float:
            statistic representing quality of fit to be minimized
        """

        #def _element_if_in_bounds(arr, index):
        #    try:
        #        return arr[index]
        #    except IndexError:
        #        return 0

        out_coords = (np.array(self.grid_model(*updated_model(*in_coords))) + 0.5).astype(np.int32)
        if len(in_coords) == 1:
            out_coords = out_coords[np.newaxis, :]
        #result = sum(_element_if_in_bounds(landscape, coord[::-1]) for coord in zip(*out_coords))
        result = cython_utils.landstat(landscape.ravel(), out_coords.ravel(),
                                       np.array(landscape.shape, dtype=np.int32),
                                       len(landscape.shape), out_coords[0].size)
        return -result  # to minimize

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
        # Turn 1D arrays into tuples to allow iteration over axes
        try:
            iter(coords[0])
        except TypeError:
            coords = (coords,)

        landscape = np.zeros(landshape)
        hw = int(maxsig * sigma)
        grid = np.meshgrid(*[np.arange(0, hw * 2 + 1)] * landscape.ndim)
        rsq = sum((ax - hw) ** 2 for ax in grid)
        mountain = np.exp(-0.5 * rsq / (sigma * sigma))

        # Place a mountain onto the landscape for each coord in coords
        # Need to crop at edges if mountain extends beyond landscape
        for coord in zip(*self.grid_model(*coords)):
            lslice = []
            mslice = []
            for pos, length in zip(coord[::-1], landshape):
                l1, l2 = int(pos - 0.5) - hw, int(pos - 0.5) + hw + 1
                m1, m2 = 0, hw * 2 + 1
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
                 landscape=None, scale=None, **kwargs):
        model_copy = _validate_model(model, self.supported_constraints)

        # Turn 1D arrays into tuples to allow iteration over axes
        try:
            iter(in_coords[0])
        except TypeError:
            in_coords = (in_coords,)
        try:
            iter(ref_coords[0])
        except TypeError:
            ref_coords = (ref_coords,)
            output1d = True
        else:
            output1d = False

        # Remember, coords are x-first (reversed python order)
        self.grid_model = models.Identity(len(in_coords))
        if landscape is None:
            mins = [min(refco) for refco in ref_coords]
            maxs = [max(refco) for refco in ref_coords]
            if scale:
                self.grid_model = reduce(Model.__and__, [models.Shift(-_min) |
                                                         models.Scale(scale) for _min in mins])
            else:
                scale = 1
                self.grid_model = reduce(Model.__and__, [models.Shift(-_min) for _min in mins])
            landshape = tuple(max(int((_max - _min) * scale), 1)
                              for _min, _max in zip(mins, maxs))[::-1]

        # We need to fiddle around a bit here to ensure a 1D output gets
        # returned in a way that can be unpacked (like higher-D outputs)
        if output1d:
            m = self.grid_model.copy()
            self.grid_model = lambda *args: m(args)
        if landscape is None:
            landscape = self.mklandscape(ref_coords, sigma*scale, maxsig, landshape)

        farg = (model_copy, np.asanyarray(in_coords, dtype=float), landscape)
        p0, *_ = model_to_fit_params(model_copy)

        ranges = []
        for p in model_copy.param_names:
            bounds = model_copy.bounds[p]
            try:
                diff = np.diff(bounds)[0]
            except TypeError:
                pass
            else:
                # We don't check that the value of a fixed param is within bounds
                if diff > 0 and not model_copy.fixed[p]:
                    if 'offset' in p:
                        stepsize = min(sigma, 0.1 * diff)
                    elif 'angle' in p:
                        stepsize = max(0.5, 0.1 * diff)
                    elif 'factor' in p:
                        stepsize = max(0.01, 0.1 * diff)
                    ranges.append(slice(*(bounds + (stepsize,))))
                    continue
            ranges.append((getattr(model_copy, p).value,) * 2)

        # Ns=1 limits the fitting along an axis where the range is not a slice
        # object: this is those were the bounds are equal (i.e. fixed param)
        fitted_params = self._opt_method(self.objective_function, ranges,
                                         farg, Ns=1, finish=None, **kwargs)
        fitter_to_model_params(model_copy, fitted_params)
        return model_copy


class KDTreeFitter(Fitter):
    """
    Fitter class to determine the best transformation from a set of N input
    coordinates (or any dimensionality) to a set of M reference coordinates
    with the same dimensionality. The two coordinate lists need not have
    the same length.

    When a call to the KDTreeFitter instance is made, a KDTree is constructed
    from the reference coordinates, allowing rapid computation of the distance
    between any location in the reference coordinate system and each of the
    reference coordinates. Then, for each interation of the optimization, the
    input coordinate list is transformed and a score given to each combination
    of input coordinates and reference coordinates. This score is defined by
    the "proximity function" and should be a monotonically-decreasing function
    of separation (although no check is made for this). Although there are NxM
    combinations of coordinates, the calculations for each transformed input
    coordinate are limited to those reference coordinates within a certain
    distance ("maxsig" multiplied by "sigma", where "sigma" is a scale-length
    in the proximity function), up to a maximum of "k" reference coordinates.
    Each coordinate in both the input and reference lists can be assigned a
    multiplicative weight to enhance the score when it is matched. These
    scores are summed for each combination and that score is maximized.

    An example of when multiple reference coordinates should be matched to a
    single input coordinate is the case of a low-resolution arc-lamp spectrum
    where peaks in the data could be the result of multiple blended arc lines
    in the reference list.

    An example of when a reference coordinate can match multiple input
    coordinates is when a high-resolution image is matched to a lower-resolution
    catalog (such as 2MASS) and a single catalog object may be detected as
    multiple sources.

    Although there is a defined maximum to the number of reference coordinates
    that can be matched to an input coordinate, multiple input coordinates can
    be matched to a single reference coordinate without limit (a coding
    limitation).

    The fitter is instantiated with parameters defining the matching
    requirements, and then called with the coordinate lists and their weights,
    and the initial guess of the transformation (an astropy Model instance).
    """

    supported_constraints = ['bounds', 'fixed']

    def __init__(self, method='Nelder-Mead', proximity_function=None,
                 sigma=5.0, maxsig=5.0, k=5):
        """
        Parameters
        ----------
        proximity_function : callable/None
            function to call to determine score for proximity of reference
            and transformed input coordinates. Must take two arguments: the
            distance and a "sigma" factor, indicating the matching scale
            (in the reference frame). If None, use the default
            `KDTreeFitter.gaussian`.

        sigma : float
            matching scale (in the reference frame)

        maxsig : float
            maximum number of scale lengths for a match to be counted

        k : int
            maximum number of matches to be considered
        """
        if proximity_function is None:
            proximity_function = KDTreeFitter.lorentzian

        self.statistic = None
        self.niter = None
        self.sigma = sigma
        self.maxsep = self.sigma * maxsig
        self.k = k
        self.proximity_function = partial(proximity_function, sigma=self.sigma)
        try:
            opt_method = getattr(optimize, method)
            self._method = None
        except AttributeError:
            # Fitter won't accept a partial object(!?)
            opt_method = optimize.minimize
            self._method = method

        super().__init__(opt_method, statistic=self._kdstat)

    def __call__(self, model, in_coords, ref_coords, in_weights=None,
                 ref_weights=None, matches=None, **kwargs):
        """
        Perform a minimization using the KDTreeFitter

        Parameters
        ----------
        model: FittableModel
            initial guess at model defining transformation
        in_coords: array-like (n x N)
            array of input coordinates
        ref_coords: array-like (n x M)
            array of reference coordinates
        in_weights: array-like (N,)
            weights for input coordinates
        ref_weights: array-like (M,)
            weights for reference coordinates
        kwargs: dict
            additional arguments to control fit

        Returns
        -------
        Model: best-fitting model
        also assigns attributes:
        x: array-like
            best-fitting parameters
        fun: float
            final value of fitting function
        nit: int
            number of iterations performed

        """
        _validate_constraints(self.supported_constraints, model)
        model_copy = model.copy()

        # Turn 1D arrays into tuples to allow iteration over axes
        try:
            iter(in_coords[0])
        except TypeError:
            in_coords = (in_coords,)
        try:
            iter(ref_coords[0])
        except TypeError:
            ref_coords = (ref_coords,)

        # Starting simplex step size is set to be 5% of parameter values
        # Need to ensure this is larger than the convergence tolerance
        # so move the initial values away from zero if necessary
        try:
            xatol = kwargs['options']['xatol']
        except KeyError:
            pass
        else:
            for p in model_copy.param_names:
                pval = getattr(model_copy, p).value
                ### EDITED THIS LINE SO TAKE A LOOK IF 2D MATCHING GOES WRONG!!
                if abs(pval).all() < 20 * xatol and not model_copy.fixed[p]:  # and 'offset' in p
                    getattr(model_copy, p).value = 20 * xatol if pval == 0 \
                        else (np.sign(pval) * 20 * xatol)

        # cKDTree.query() returns a value of n for no neighbour so make coding
        # easier by allowing this to match a zero-weighted reference
        self.match_weights = np.outer(np.ones(len(in_coords[0])) if in_weights is None else in_weights,
                                      list([1.0] * len(ref_coords[0]) if ref_weights is None else list(ref_weights)) + [0])
        if matches is not None:
            for i, m in enumerate(matches):
                if m >= 0:
                    value = self.match_weights[i, m]
                    self.match_weights[i] = 0
                    self.match_weights[:, m] = 0
                    self.match_weights[i, m] = value
        self.in_range = tuple(range(self.match_weights.shape[0]))

        ref_coords = np.array(list(zip(*ref_coords)))
        tree = spatial.cKDTree(ref_coords)
        # avoid _convert_input since tree can't be coerced to a float
        farg = (model_copy, in_coords, tree)
        p0, *_ = model_to_fit_params(model_copy)

        def bounds_for_unfixed_parameters(m):
            return tuple(m.bounds[p] for p in m.param_names if not m.fixed[p])

        opt_method_params = inspect.signature(self._opt_method).parameters
        arg_names = list(k for k, v in opt_method_params.items()
                         if v.default == inspect.Parameter.empty)
        kwarg_names = list(k for k, v in opt_method_params.items()
                           if v.default != inspect.Parameter.empty)
        args = [self.objective_function]
        if arg_names[1] == 'x0':
            args.append(p0)
        elif arg_names[1] == 'bounds':
            args.append(bounds_for_unfixed_parameters(model_copy))
        else:
            raise ValueError("Don't understand argument {}".format(arg_names[1]))

        # Just in case as a result of scipy change
        if 'bounds' in kwarg_names:
            kwargs['bounds'] = bounds_for_unfixed_parameters(model_copy)
        if 'args' in arg_names or 'args' in kwarg_names:
            kwargs['args'] = farg

        if self._method is not None:
            kwargs['method'] = self._method

        if 'minimizer_kwargs' in kwarg_names:
            kwargs['minimizer_kwargs'] = {'args': farg,
                                          'method': 'Nelder-Mead'}

        result = self._opt_method(*args, **kwargs)

        fitted_params = result['x']
        fitter_to_model_params(model_copy, fitted_params)
        self.statistic = result['fun']
        self.niter = result['nit']  # Number of iterations
        self.message = result['message']  # Message about why it terminated
        self.status = result['success']  # Numeric return status (0 for 'good')
        return model_copy

    @staticmethod
    def gaussian(distance, sigma):
        return np.exp(-0.5 * distance * distance / (sigma * sigma))

    @staticmethod
    def lorentzian(distance, sigma):
        return 1. / (distance * distance + sigma * sigma)

    def _kdstat(self, tree, updated_model, in_coords):
        """
        Compute the statistic for transforming coordinates onto a set of
        reference coordinates. This uses mathematical calculations and is not
        pixellated like the landscape-array methods.

        Parameters
        ----------
        tree: :class:`~scipy.spatial.KDTree`
            a KDTree made from the reference coordinates

        updated_model: :class:`~astropy.modeling.FittableModel`
            transformation (input -> reference) being investigated

        in_coords: list or :class:`~numpy.ndarray` dimension (n, m)
            List or array with the input coordinates.

        ref_coords: list or :class:`~numpy.ndarray` dimension (n', m)
            List or array with the reference coordinates.

        in_weights : list or :class:`~numpy.ndarray` dimension (n,)
            List or array with the input weights.

        ref_weights : list or :class:`~numpy.ndarray` dimension (n',)
            List or array with the reference weights.

        matches : None or list/array dimension (n,)
            Index in ref_coords to match to each input coordinate

        Returns
        -------
        float : Statistic representing quality of fit to be minimized
        """
        out_coords = np.asarray(updated_model(*in_coords))
        #if len(in_coords) == 1:
        #    out_coords = (out_coords,)
        #out_coords = np.array(list(zip(*out_coords)))
        #dist, idx = tree.query(out_coords, k=self.k,
        #                       distance_upper_bound=self.maxsep)
        if len(in_coords) > 1:
            dist, idx = tree.query(out_coords.T, k=self.k,
                                   distance_upper_bound=self.maxsep)
        else:
            dist, idx = tree.query(np.expand_dims(out_coords, 1), k=self.k,
                                   distance_upper_bound=self.maxsep)

        # This if statement doesn't seem to speed things up very much
        pf = self.proximity_function(dist)
        if self.k > 1:
           result = np.sum([(self.match_weights[self.in_range, tuple(idx.T[i])] * pf.T[i]).sum()
                            for i in range(self.k)])
        else:
            result = (self.match_weights[self.in_range, tuple(idx)] * pf).sum()

        return -result  # to minimize


def fit_model(model, xin, xout, sigma=5.0, tolerance=1e-5, scale=None,
              brute=True, release=False, verbose=True):
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
    scale: float/None
        scaling to use for coordinates if performing a brute-force fit,
        to ensure that the peaks in the "landscape" are separated
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
        m_init = fit_it(model, xin, xout, scale=scale, sigma=sigma)
        if verbose:
            log.stdinfo(_show_model(m_init, "Coarse model in {:.2f} seconds".
                                    format((datetime.now() - start).total_seconds())))

        # Re-fix parameters in the intermediate model, if they were fixed
        # in the original model
        for p in m_init.param_names:
            try:
                if np.diff(getattr(model, p).bounds)[0] == 0:
                    getattr(m_init, p).fixed = True
                    continue
            except TypeError:
                pass
            if release:
                getattr(m_init, p).bounds = (None, None)
    else:
        m_init = model.copy()

    # More precise minimization using pairwise calculations
    fit_it = KDTreeFitter(sigma=sigma, method='Nelder-Mead')
    # We don't care about how much the function value changes (fatol), only
    # that the position is robust (xatol)
    m_final = fit_it(m_init, xin, xout, options={'xatol': tolerance})
    if verbose:
        log.stdinfo(_show_model(m_final, "Final model in {:.2f} seconds".
                                format((datetime.now() - start).total_seconds())))
    return m_final


def _show_model(model, intro=""):
    """Provide formatted output of a (possibly compound) transformation"""
    model_str = "{}\n".format(intro) if intro else ""
    try:
        iterator = iter(model)
    except TypeError:
        iterator = [model]
    # Only display parameters of those models that have names
    for m in iterator:
        if m.name != 'xx':
            for param in [getattr(m, name) for name in m.param_names]:
                if not (param.fixed or (param.bounds[0] == param.bounds[1]
                                        and param.bounds[0] is not None)):
                    model_str += "{}: {}\n".format(param.name, param.value)
    return model_str


##############################################################################
def match_sources(incoords, refcoords, radius=2.0):
    """
    Match two sets of sources that are on the same reference frame. In general
    the closest match will be used, but there can be a priority list that will
    take precedence.

    This does a "greedy" match, starting with the closest pair,
    instead of sequentially through the refcoords

    Parameters
    ----------
    incoords: nxN array
        input source coords (transformed to reference frame)
    refcoords: nxM array
        reference source coords
    radius:
        maximum separation for a match

    Returns
    -------
    int array of length N:
        index of matched sources in the reference list (-1 means no match)
    """
    if np.asarray(incoords).size == 0:
        return np.array([], dtype=int)

    try:
        iter(incoords[0])
    except TypeError:
        incoords = (incoords,)
        refcoords = (refcoords,)
    matched = np.full((len(incoords[0]),), -1, dtype=int)
    tree = spatial.cKDTree(list(zip(*refcoords)))
    dist, idx = tree.query(list(zip(*incoords)), distance_upper_bound=radius, k=5)

    while True:
        min_arg = np.unravel_index(np.argmin(dist), dist.shape)
        min_dist = dist[min_arg]
        if min_dist == np.inf:
            break
        i = idx[min_arg]
        # min_arg has value (number_of_incoord, order_in_distance_list)
        matched[min_arg[0]] = i
        # Remove this incoord from the pool
        dist[min_arg[0], :] = np.inf
        # Remove this refcoord from the pool
        dist[idx == i] = np.inf
    return matched


def find_alignment_transform(incoords, refcoords, transform=None, shape=None,
                             search_radius=10, match_radius=2, rotate=False,
                             scale=False, brute=True, sigma=5, factor=None,
                             return_matches=False):
    """
    This function computes a transform that maps one set of coordinates to
    another. By default, only a shift is used, by a rotation and magnification
    can also be applied if requested. An initial transform may be supplied
    and, if so, its affine approximation will be used as a starting point.

    Parameters
    ----------
    incoords: tuple
        x-coords and y-coords of objects in input image
    refcoords: tuple
        x-coords and y-coords of objects in reference image
    transform: Transform/None
        existing transformation (if None, will do brute search)
    shape: 2-tuple/None
        shape (standard python order, y-first)
    search_radius: float
        size of search box (in pixels)
    match_radius: float
        matching radius for objects (in pixels)
    rotate: bool
        add a rotation to the alignment transform?
    scale: bool
        add a magnification to the alignment transform?
    brute: bool
        perform brute (landscape) search first?
    sigma: float
        scale-length for source matching
    factor: float/None
        scaling factor to convert coordinates to pixels in the BruteLandscapeFitter()
    return_matches: bool
        return a list of matched objects as well as the Transform?

    Returns
    -------
    Model: alignment transform
    matches: 2 lists (optional)
        OBJCAT sources in input and reference that are matched
    """
    log = logutils.get_logger(__name__)
    if shape is None:
        shape = tuple(max(c) - min(c) for c in incoords)

    largest_dimension = max(*shape)
    # Values larger than these result in errors of >1 pixel
    mag_threshold = 1. / largest_dimension
    rot_threshold = np.degrees(mag_threshold)

    # Set up the initial model
    if transform is None:
        transform = models.Identity(2)

    # We always refactor the transform (if provided) in a prescribed way so
    # as to ensure it's fittable and not overly weird
    affine = adwcs.calculate_affine_matrices(transform, shape)
    m_init = models.Shift(affine.offset[1]) & models.Shift(affine.offset[0])

    # This is approximate since the affine matrix might have differential
    # scaling and a shear
    magnification = np.sqrt(abs(np.linalg.det(affine.matrix)))
    rotation = np.degrees(np.arctan2(affine.matrix[0, 1] - affine.matrix[1, 0],
                                     affine.matrix[0, 0] + affine.matrix[1, 1]))
    m_init.offset_0.bounds = (m_init.offset_0 - search_radius,
                              m_init.offset_0 + search_radius)
    m_init.offset_1.bounds = (m_init.offset_1 - search_radius,
                              m_init.offset_1 + search_radius)

    m_rotate = Rotate2D(rotation)
    if rotate:
        m_rotate.angle.bounds = (rotation-5, rotation+5)
        m_init = m_rotate | m_init
    elif abs(rotation) > rot_threshold:
        m_rotate.angle.fixed = True
        m_init = m_rotate | m_init
        log.warning(f"A rotation of {rotation:.3f} degrees is applied but "
                    "held fixed")

    m_magnify = Scale2D(magnification)
    if scale:
        m_magnify.factor.bounds = (magnification-0.05, magnification+0.05)
        m_init = m_magnify | m_init
    elif abs(magnification - 1) > mag_threshold:
        m_magnify.factor.fixed = True
        m_init = m_magnify | m_init
        log.warning(f"A magnification of {magnification:.4f} is applied but "
                    "held fixed")

    # Tolerance here aims to achieve <0.1 pixel differences in the tests
    try:
        m_final = fit_model(m_init, incoords, refcoords, sigma=sigma, scale=factor,
                            brute=brute, tolerance=sigma*1e-5)
    except ValueError as e:
        if any(np.logical_or(max(refco) < min(inco), min(refco) > max(inco))
               for inco, refco in zip(incoords, refcoords)):
            log.warning("No overlap between input and reference coords")
            m_final = models.Identity(len(incoords))
        else:
            raise e

    if return_matches:
        matched = match_sources(m_final(*incoords), refcoords, radius=match_radius)
        ind2 = np.where(matched >= 0)
        ind1 = matched[ind2]
        obj_list = [[], []] if len(ind1) < 1 else [np.array(list(zip(*incoords)))[ind2],
                                                   np.array(list(zip(*refcoords)))[ind1]]
        return m_final, obj_list
    return m_final
