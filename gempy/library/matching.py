# Copyright(c) 2017-2020 Association of Universities for Research in Astronomy, Inc.

import numpy as np
from datetime import datetime
from functools import partial, reduce
import inspect

from scipy import optimize, spatial
from astropy.modeling import fitting, models, Model, FittableModel
from astropy.modeling.fitting import (_validate_model,
                                      _fitter_to_model_params,
                                      _model_to_fit_params, Fitter,
                                      _convert_input)

from astrodata import wcs as adwcs

try:
    from gempy.library import cython_utils
except ImportError:  # pragma: no cover
    raise ImportError("Run 'cythonize -i cython_utils.pyx' in gempy/library")
from ..utils import logutils
from .astromodels import Rotate2D, Scale2D

from matplotlib import pyplot as plt

##############################################################################
class MatchBox:
    """
    A class to hold two sets of coordinates that have a one-to-one
    correspondence, and the transformations that go between them.
    """

    def __init__(self, input_coords, output_coords, forward_model=None,
                 backward_model=None, fitter=fitting.LinearLSQFitter,
                 **kwargs):
        super().__init__(**kwargs)
        self._input_coords = list(input_coords)
        self._output_coords = list(output_coords)
        try:
            self._ndim = len(self._input_coords[0])
        except TypeError:
            self._ndim = 1
        self.validate_coords()
        self._forward_model = None if forward_model is None else forward_model.copy()
        self._backward_model = None if backward_model is None else backward_model.copy()
        self._fitter = fitter

    @classmethod
    def create_from_kdfit(cls, input_coords, output_coords, model,
                          match_radius, sigma_clip=None, priority=[]):
        """
        Creates a MatchBox object from a KDTree-fitted model. This does
        the matching between input and output coordinates and, if
        requested, iteratively sigma-clips.

        Parameters
        ----------
        input_coords: array-like
            untransformed input coordinates
        output_coords: array-like
            output coordinates
        model: Model
            transformation
        match_radius: float
            maximum distance for matching coordinates
        sigma_clip: float/None
            if not None, iteratively sigma-clip using this number of
            standard deviations

        Returns
        -------
        MatchBox
        """
        num_matches = None
        init_match_radius = match_radius
        while True:
            matched = match_sources(model(input_coords), output_coords,
                                    radius=match_radius)
            incoords, outcoords = zip(*[(input_coords[i], output_coords[m])
                                        for i, m in enumerate(matched) if m > -1])
            m = cls(incoords, outcoords, forward_model=model)
            try:
                m.fit_forward()
            except ValueError:
                return cls([], [], forward_model=None)
            if sigma_clip is None or num_matches == len(incoords):
                break
            num_matches = len(incoords)
            match_radius = min(init_match_radius, sigma_clip * m.rms_output)
            model = m.forward
        return m

    def validate_coords(self, input_coords=None, output_coords=None):
        """Confirm that the two sets of coordinates are compatible"""
        if input_coords is None:
            input_coords = self._input_coords
        if output_coords is None:
            output_coords = self._output_coords
        if len(input_coords) != len(output_coords):
            raise ValueError("Coordinate lists have different lengths")
        try:
            for coord in input_coords + output_coords:
                try:
                    assert len(coord) == self._ndim
                except TypeError:
                    assert self._ndim == 1
        except AssertionError:
            raise ValueError("Incompatible elements in one or both coordinate lists")

    def __len__(self):
        return len(self._input_coords)

    @property
    def input_coords(self):
        return self._input_coords

    @property
    def output_coords(self):
        return self._output_coords

    @property
    def forward(self):
        """Compute the forward transformation"""
        return self._forward_model

    @forward.setter
    def forward(self, model):
        if isinstance(model, FittableModel):
            self._forward_model = model
        else:
            raise ValueError("Model is not Fittable")

    @property
    def backward(self):
        """Compute the backward transformation"""
        return self._backward_model

    @backward.setter
    def backward(self, model):
        if isinstance(model, FittableModel):
            self._backward_model = model
        else:
            raise ValueError("Model is not Fittable")

    def _fit(self, model, input_coords, output_coords):
        """Fits a model to input and output coordinates after repackaging
        them in the correct format"""
        fit = self._fitter()
        prepared_input = np.array(input_coords).T
        prepared_output = np.array(output_coords).T
        if len(prepared_input) == 1:
            prepared_input = (prepared_input,)
            prepared_output = (prepared_output,)
        try:
            return fit(model, prepared_input, prepared_output)
        except fitting.UnsupportedConstraintError:
            for p in model.param_names:
                getattr(model, p).bounds = (None, None)
            return fit(model, prepared_input, prepared_output)

    def fit_forward(self, model=None, coords=None, set_inverse=True):
        """
        Fit the forward (input->output) model.

        Parameters
        ----------
        model: FittableModel
            initial model guess (if None, use the _forward_model)
        coords: array-like
            if not None, fit the backward-transformed version of these coords
            to these coords (_backward_model must not be None)
        set_inverse: bool
            set the inverse (backward) model too, if possible?
        """
        if model is None:
            model = self._forward_model
        if model is None:
            raise ValueError("No forward model specified")
        if coords is None:
            coords = self._input_coords
            out_coords = self._output_coords
        else:
            if self._backward_model is None:
                raise ValueError("A backward model must exist to map specific coords")
            out_coords = self.backward(coords)
        fitted_model = self._fit(model, coords, out_coords)
        self._forward_model = fitted_model
        if set_inverse:
            try:
                self._backward_model = fitted_model.inverse
            except NotImplementedError:
                pass

    def fit_backward(self, model=None, coords=None, set_inverse=True):
        """
        Fit the backward (output->input) model. If this has an inverse, set the
        forward_model to its inverse.

        Parameters
        ----------
        model: FittableModel
            initial model guess (if None, use the _backward_model)
        coords: array-like
            if not None, fit the forward-transformed version of these coords
            to these coords (_forward_model must not be None)
        set_inverse: bool
            set the inverse (forward) model too, if possible?
        """
        if model is None:
            model = self._backward_model
        if model is None:
            raise ValueError("No backward model specified")
        if coords is None:
            coords = self._output_coords
            out_coords = self._input_coords
        else:
            if self._forward_model is None:
                raise ValueError("A forward model must exist to map specific coords")
            out_coords = self.forward(coords)
        fitted_model = self._fit(model, coords, out_coords)
        self._backward_model = fitted_model
        if set_inverse:
            try:
                self._forward_model = fitted_model.inverse
            except NotImplementedError:
                pass

    def add_coords(self, input_coords, output_coords):
        """
        Add coordinates to the input and output coordinate lists

        Parameters
        ----------
        input_coords: array-like/value
            New input coordinates
        output_coords: array-like/value
            New output coordinates
        """
        try:  # Replace value with single-element list
            len(input_coords)
        except TypeError:
            input_coords = [input_coords]
            output_coords = [output_coords]
        self.validate_coords(input_coords, output_coords)
        self._input_coords.extend(input_coords)
        self._output_coords.extend(output_coords)

    def __delitem__(self, index):
        del self._input_coords[index]
        del self._output_coords[index]

    def sort(self, by_output=False, reverse=False):
        """
        Sort the coordinate lists, either by input or output.

        Parameters
        ----------
        by_output: bool
            If set, sort by the output coords rather than input coords
        reverse: bool
            If set, put the largest elements at the start of the list
        """
        ordered = zip(*sorted(zip(self._input_coords, self._output_coords),
                              reverse=reverse, key=lambda x: x[1 if by_output else 0]))
        self._input_coords, self._output_coords = ordered

    @property
    def residuals(self):
        """
        Return the residuals of the fit
        """
        try:
            len(self._input_coords[0])
        except TypeError:
            return self._output_coords - self.forward(self._input_coords)
        else:
            return list(c1[i] - c2[i] for c1, c2 in zip(self._output_coords,
                                                        self.forward(self._input_coords))
                        for i in range(self._ndim))

    @property
    def rms_input(self):
        """
        Return the rms of the fit in input units
        """
        return self._rms(self._input_coords, self.backward(self._output_coords))

    @property
    def rms_output(self):
        """
        Return the rms of the fit in output units
        """
        return self._rms(self._output_coords, self.forward(self._input_coords))

    def _rms(self, coords1, coords2):
        try:
            len(coords1[0])
        except TypeError:
            return np.std(coords1 - coords2)
        else:
            return list(np.std([c1[i] - c2[i] for c1, c2 in zip(coords1, coords2)])
                        for i in range(self._ndim))


##############################################################################

class Chebyshev1DMatchBox(MatchBox):
    """
    A MatchBox that specifically has Chebyshev1D transformations, and provides
    additional plotting methods for analysis.
    """

    def __init__(self, input_coords, output_coords, forward_model=None,
                 backward_model=None, fitter=fitting.LinearLSQFitter,
                 **kwargs):
        if not isinstance(forward_model, models.Chebyshev1D):
            raise ValueError("forward_model is not a Chebyshev1D instance")
        if (backward_model is not None and
                not isinstance(backward_model, models.Chebyshev1D)):
            raise ValueError("backward_model is not a Chebyshev1D instance")
        super().__init__(input_coords, output_coords,
                         forward_model=forward_model,
                         backward_model=backward_model,
                         fitter=fitter, **kwargs)

    def display_fit(self, remove_orders=1, axes=None, show=False):
        """
        Plot the fit

        Parameters
        ----------
        remove_orders: int
            Only show the fit's orders above this value (so the default value
            of 1 removes the linear component)
        axes: None/Axes object
            axes for plotting (None => create new figure)
        show: bool
            call plt.show() method at end?
        """
        if axes is None:
            fig, axes = plt.subplots()

        model = self.forward.copy()
        if not (remove_orders is None or remove_orders < 0):
            for i in range(0, remove_orders + 1):
                setattr(model, 'c{}'.format(i), 0)

        limits = self._forward_model.domain or (min(self._input_coords), max(self._input_coords))
        x = np.linspace(limits[0], limits[1], 1000)
        axes.plot(self.forward(x), model(x))
        axes.plot(self.forward(self._input_coords), model(self._input_coords) + self.residuals, 'ko')

        if show:
            plt.show()


##############################################################################

class BruteLandscapeFitter(Fitter):
    """
    Fitter class that employs brute-force optimization to map a set of input
    coordinates onto a set of reference coordinates by cross-correlation
    over a "landscape" of "mountains" representing the reference coords
    """

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
            out_coords = out_coords[np.new_axis, :]
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
        model_copy = _validate_model(model, ['bounds', 'fixed'])

        # Turn 1D arrays into tuples to allow iteration over axes
        try:
            iter(in_coords[0])
        except TypeError:
            in_coords = (in_coords,)
        try:
            iter(ref_coords[0])
        except TypeError:
            ref_coords = (ref_coords,)

        # Remember, coords are x-first (reversed python order)
        self.grid_model = models.Identity(len(in_coords))
        if landscape is None:
            mins = [min(refco) for refco in ref_coords]
            maxs = [max(refco) for refco in ref_coords]
            if scale:
                self.grid_model = reduce(Model.__and__, [models.Shift(-_min) |
                                                         models.Scale(scale) for _min in mins])
                landshape = tuple(int((_max - _min) * scale)
                                  for _min, _max in zip(mins, maxs))[::-1]
            else:
                scale = 1
                landshape = tuple(int(_max) for _max in maxs)[::-1]
            landscape = self.mklandscape(ref_coords, sigma*scale, maxsig, landshape)

        farg = (model_copy,) + _convert_input(in_coords, landscape)
        p0, _ = _model_to_fit_params(model_copy)

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
                        stepsize = min(0.5 * sigma, 0.1 * diff)
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
        _fitter_to_model_params(model_copy, fitted_params)
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
        model_copy = _validate_model(model, ['bounds', 'fixed'])

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
                if abs(pval) < 20 * xatol and not model_copy.fixed[p]:  # and 'offset' in p
                    getattr(model_copy, p).value = 20 * xatol if pval == 0 \
                        else (np.sign(pval) * 20 * xatol)

        if in_weights is None:
            in_weights = np.ones((len(in_coords[0]),))
        if ref_weights is None:
            ref_weights = np.ones((len(ref_coords[0]),))
        # cKDTree.query() returns a value of n for no neighbour so make coding
        # easier by allowing this to match a zero-weighted reference
        ref_weights = np.append(ref_weights, (0,))

        ref_coords = np.array(list(zip(*ref_coords)))
        tree = spatial.cKDTree(ref_coords)
        # avoid _convert_input since tree can't be coerced to a float
        farg = (model_copy, in_coords, ref_coords, in_weights, ref_weights, matches, tree)
        p0, _ = _model_to_fit_params(model_copy)

        arg_names = inspect.getfullargspec(self._opt_method).args
        args = [self.objective_function]
        if arg_names[1] == 'x0':
            args.append(p0)
        elif arg_names[1] == 'bounds':
            args.append(tuple(model_copy.bounds[p] for p in model_copy.param_names))
        else:
            raise ValueError("Don't understand argument {}".format(arg_names[1]))

        if 'args' in arg_names:
            kwargs['args'] = farg

        if 'method' in arg_names:
            kwargs['method'] = self._method

        if 'minimizer_kwargs' in arg_names:
            kwargs['minimizer_kwargs'] = {'args': farg,
                                          'method': 'Nelder-Mead'}

        result = self._opt_method(*args, **kwargs)

        fitted_params = result['x']
        _fitter_to_model_params(model_copy, fitted_params)
        self.statistic = result['fun']
        self.niter = result['nit']
        return model_copy

    @staticmethod
    def gaussian(distance, sigma):
        return np.exp(-0.5 * distance * distance / (sigma * sigma))

    @staticmethod
    def lorentzian(distance, sigma):
        return 1. / (distance * distance + sigma * sigma)

    def _kdstat(self, tree, updated_model, in_coords, ref_coords,
                in_weights, ref_weights, matches=None):
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
        out_coords = updated_model(*in_coords)
        if len(in_coords) == 1:
            out_coords = (out_coords,)
        out_coords = np.array(list(zip(*out_coords)))
        dist, idx = tree.query(out_coords, k=self.k,
                               distance_upper_bound=self.maxsep)

        if matches is not None:
            nr = len(ref_coords)
            for i, m in enumerate(matches):
                if m >= 0:
                    d = np.linalg.norm(out_coords[i] - ref_coords[m])
                    if self.k > 1:
                        dist[i] = np.array([d] + [np.inf] * (self.k-1))
                        idx[i] = np.array([m] + [nr] * (self.k-1))
                    else:
                        dist[i] = d
                        idx[i] = m

        if self.k > 1:
            result = sum(in_wt * ref_weights[i] * self.proximity_function(d)
                         for in_wt, dd, ii in zip(in_weights, dist, idx)
                         for d, i in zip(dd, ii))
        else:
            result = sum(in_wt * ref_weights[i] * self.proximity_function(d)
                         for in_wt, d, i in zip(in_weights, dist, idx))

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
        if m.name is not 'xx':
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
                             scale=False, brute=True, sigma=10, factor=None,
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
        log.warning("A rotation of {:.3f} degrees is expected but the "
                    "rotation is fixed".format(rotation))

    m_magnify = Scale2D(magnification)
    if scale:
        m_magnify.factor.bounds = (magnification-0.05, magnification+0.05)
        m_init = m_magnify | m_init
    elif abs(magnification - 1) > mag_threshold:
        m_magnify.factor.fixed = True
        m_init = m_magnify | m_init
        log.warning("A magnification of {:.4f} is expected but the "
                    "magnification is fixed".format(magnification))

    from astropy.table import Table
    tobj = Table([*incoords, *m_init(*incoords)], names=['XIN', 'YIN', 'XTRANS', 'YTRANS'])
    tref = Table(refcoords, names=['XREF', 'YREF'])
    tobj.write('obj.fits', format='fits', overwrite=True)
    tref.write('ref.fits', format='fits', overwrite=True)

    # Perform the fit. We expect sigma ~ 10 pixels, so the tolerance
    # here is ~ 0.001 pixels, which is fine
    m_final = fit_model(m_init, incoords, refcoords, sigma=sigma, scale=factor,
                        brute=brute, tolerance=sigma*1e-4)
    if return_matches:
        matched = match_sources(m_final(*incoords), refcoords, radius=match_radius)
        ind2 = np.where(matched >= 0)
        ind1 = matched[ind2]
        obj_list = [[], []] if len(ind1) < 1 else [np.array(list(zip(*incoords)))[ind2],
                                                   np.array(list(zip(*refcoords)))[ind1]]
        return m_final, obj_list
    return m_final
