import itertools
import warnings

import numpy as np

from scipy.interpolate import BSpline
from scipy.interpolate import RegularGridInterpolator

from astropy.modeling import models, Fittable1DModel, Parameter
from astropy.constants import c, h, k_B
from astropy import units as u

try:
    from gempy.library import cython_utils
except ImportError:  # pragma: no cover
    raise ImportError("Run 'cythonize -i cython_utils.pyx' in gempy/library")


class PCA(Fittable1DModel):
    """
    An astropy.modeling.Model class that can be used for Principal Component
    Analysis. Its parameters are the strengths of the provided components.

    When used in a fitter, the "input data" is irrelevant since the principal
    components are stored as attributes of the instance.
    """
    _param_names = ()
    linear = True

    def __init__(self, components, name=None, meta=None, copy=False):
        if isinstance(components, ArrayInterpolator):
            self._interpolator = components
            # Last two axes are PCA component and wavelength
            self.components = self._interpolator([x[0] for x in components.grid[:-2]])
        else:
            self.components = components.copy() if copy else components

        self.npca_params = self.components.shape[-2] - 1
        self._param_names = self._generate_coeff_names()
        for param_name in self._param_names:
            self._parameters_[param_name] = Parameter(param_name,
                                                      default=0)
        self._last_lsf_params = None
        super().__init__(name=name, meta=meta,
                         **dict(zip(self._param_names,
                                    np.zeros(len(self._param_names),))))

    @property
    def param_names(self):
        return self._param_names

    def _generate_coeff_names(self):
        names = [f'pc{i:02d}' for i, _ in enumerate(self.components[1:])]
        # We don't know the names of these because the LSF object isn't passed
        if not self.fixed:
            names.extend([f'lsf{i:02d}' for i in range(len(self._interpolator.values.shape) - 2)])
        return tuple(names)

    @property
    def fixed(self):
        return not hasattr(self, "_interpolator")

    @property
    def interpolation_points(self):
        try:
            return self._interpolator.grid
        except AttributeError:
            raise AttributeError("This PCA instance was not created with an interpolator")

    @property
    def interpolation_values(self):
        try:
            return self._interpolator.values
        except AttributeError:
            raise AttributeError("This PCA instance was not created with an interpolator")

    def evaluate(self, x, *params):
        if self.fixed:
            result = (np.dot(np.asarray(params).flatten(), self.components[1:]) +
                      self.components[0])
        else:
            params = np.asarray(params).flatten()
            self.set_interpolation_parameters(params[self.npca_params:])
            result = np.dot(params[:self.npca_params],
                            self.components[1:]) + self.components[0]
        return result

    def fit_deriv(self, x, *params):
        # This will get munged by the LinearLSQFitter, so return a copy
        return self.components[1:].copy()

    def sum_of_implicit_terms(self, x):
        return self.components[0]

    def set_interpolation_parameters(self, *params):
        # Don't reinterpolate if we don't have to! Need for speed!
        if params == self._last_lsf_params:
            return
        try:
            self.components = self._interpolator(*params)
            self._last_lsf_params = params
        except NameError:
            raise TypeError("This PCA instance was not created with an interpolator")


class ArrayInterpolator(RegularGridInterpolator):
    """A subclass of RegularGridInterpolator that returns an array parallel
    to the last N axes of the input data, rather than a single value. This
    allows us to create a set of PCA eigenspectra by interpolation, or
    interpolate spectra."""
    def __init__(self, points, values):
        self.num_returned_axes = len(values.shape) - len(points)
        points = tuple(points) + tuple(np.arange(values.shape[i])
                                       for i in range(-self.num_returned_axes, 0))
        super().__init__(points, values, method="linear", bounds_error=False)

    def __call__(self, *coords):
        new_coords = np.array([list(*coords) + list(x)
                               for x in itertools.product(
                *list(range(self.values.shape[i])
                      for i in range(-self.num_returned_axes, 0)))])
        # parent method returns a 1D array, so reshape it
        return super().__call__(new_coords).reshape(
            self.values.shape[-self.num_returned_axes:])


class SingleTelluricModel(Fittable1DModel):
    """
    An astropy.modeling.Model class that can be used to fit a spectrum
    with telluric absoprtion via Principal Component Analysis. Its parameters
    are the strengths of the provided components, followed by parameters
    defining a continuum. The continuum can either be a cubic spline or a
    Chebyshev polynomial.

    When used in a fitter, the "input data" is irrelevant since the principal
    components are stored as attributes of the instance. (This has to be the
    case since the telluric absorption profile isn't interpolated in any way.)

    Parameters are in the following order:
        PCA components
        LSF parameters (if variable)
        continuum model parameters
    """
    _param_names = ()

    def __init__(self, tspek, function="spline3", order=None, name=None,
                 meta=None):
        self.pca = tspek.pca
        self.domain = (tspek.waves.min(), tspek.waves.max())

        try:
            tspek.intrinsic_spectrum.unit.to(u.Unit('W m-2 nm-1'))
        except (AttributeError, u.UnitConversionError):
            raise u.UnitConversionError("Intrinsic spectrum either has no units "
                                        "or they cannot be converted to F_lambda")
        # We don't want to modify "tspek.intrinsic_spectrum"
        self.intrinsic_spectrum = tspek.intrinsic_spectrum

        self.waves = tspek.waves
        self.dwaves = tspek.dwaves

        if order is None:
            raise ValueError("'order' must be specified")
        else:
            self.order = order

        self.place_knots(tspek.mask)
        if function == "spline3":
            self.continuum_function = self.spline3
        elif function == "chebyshev":
            self.continuum_function = self.chebyshev
        else:
            raise ValueError(f"Function type '{function}' not known")

        self._param_names = self._generate_coeff_names(tspek)
        for param_name in self._param_names:
            self._parameters_[param_name] = Parameter(param_name,
                                                      default=0)

        # This is what the UI uses to report the rms. We can't calculate it
        # here because the actual data are not stored in this object, so we
        # leave it to the UI to perform the calculation and update the attribute
        self.rms = np.nan

        super().__init__(name=name, meta=meta,
                         **dict(zip(self._param_names,
                                    np.zeros(len(self._param_names),))))

        self.bounds.update({p: (-100, 100) for p in
                            self._param_names[len(self.pca.param_names):]})

    @property
    def param_names(self):
        return self._param_names

    def _generate_coeff_names(self, tspek):
        names = [f'pc{i:02d}' for i, _ in enumerate(tspek.pca.components[1:])]
        if not tspek.pca.fixed:
            names.extend(tspek.lsf.parameters)
        self.pca_params = slice(None, len(names))
        self.cont_params = slice(len(names), None)
        if self.continuum_function == self.spline3:
            names.extend([f'spl{i:02d}' for i in range(self.order + 3)])
        else:
            names.extend([f'cheb{i:02d}' for i in range(self.order + 1)])
        return tuple(names)

    @property
    def intrinsic_spectrum(self):
        return self._scaling * self._normalized_intrinsic_spectrum

    @property
    def intrinsic_spectrum_scaling(self):
        return self._scaling

    @intrinsic_spectrum.setter
    def intrinsic_spectrum(self, value):
        # Scaling is needed to stop underflow-type errors in the fitting
        self._scaling = np.median(value)  # inherits unit (must have a unit)
        self._normalized_intrinsic_spectrum = (value / self._scaling).value

    def spline3(self, params):
        c = list(params) + [0] * 4
        return BSpline(t=self.knots, c=c, k=3)

    def chebyshev(self, params):
        kwargs = {f'c{i}': p for i, p in enumerate(params)}
        return models.Chebyshev1D(degree=self.order, domain=self.domain, **kwargs)

    def place_knots(self, mask):
        # Space the knots equally among the *unmasked* pixels
        ngood = (~mask.astype(bool)).sum()
        if self.order < ngood + 3:
            knot_pixels = np.linspace(0, ngood - 1, self.order + 1)
            knot_waves = np.interp(knot_pixels, np.arange(ngood),
                                   sorted(self.waves[~mask.astype(bool)]))
        else:
            knot_pixels = np.linspace(0, mask.size, self.order + 1)
            knot_waves = np.interp(knot_pixels, np.arange(mask.size),
                                   sorted(self.waves))
        self.knots = np.r_[[knot_waves[0]] * 3, knot_waves, [knot_waves[-1]] * 3]

    def continuum(self, x):
        """
        Return the continuum part of the model only. Although this shares
        a lot of code with the evaluate() method, it's not worth trying to
        combine them because of the need to call the get_unmasked() cython
        function.
        """
        continuum_model = self.continuum_function(np.asarray(self.parameters[self.cont_params]).flatten())
        good = get_good_pixels(x, self.waves)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            continuum = 10 ** (0.4 * continuum_model(self.waves[good]))
        return continuum * (self._normalized_intrinsic_spectrum * self.dwaves)[good]

    def evaluate(self, x, *params):
        continuum_model = self.continuum_function(np.asarray(params[self.cont_params]).flatten())
        good = get_good_pixels(x, self.waves)

        # Fitter can try parameters which cause the continuum to overflow,
        # so suppress such RuntimeWarnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            continuum = 10 ** (0.4 * continuum_model(self.waves[good]))
        #if not self.pca.fixed:
        #    self.pca.set_interpolation_parameters(np.asarray(params[self.lsf_params]).flatten())
        #transmission = (np.dot(np.asarray(params[self.pca_params]).flatten(),
        #                       self.pca.components[1:]) + self.pca.components[0])
        transmission = self.pca.evaluate(x, np.asarray(params[self.pca_params]).flatten())
        result = continuum * (self._normalized_intrinsic_spectrum * self.dwaves * transmission)[good]
        #if np.isinf(result).any():
        #    print("INFINITE")
        #    print(np.asarray(params).flatten())
        #if np.isnan(result).any():
        #    print("NAN")
        #    print(self.dwaves)
        #    print(np.asarray(params).flatten())
        #    print(continuum.max())
        #    print(self.waves[0], good.sum(), continuum.max())
        #    print(params)
        # Calculations are done in float64, but there is no reason for any
        # value to exceed the float32 limit
        maxval = np.finfo(np.float32).max
        result = np.maximum(np.minimum(result, maxval), -maxval)
        return result

    def self_correction(self):
        """
        Return multiplicative scaling factors to correct the data to the
        intrinsic spectrum used to construct the model.
        """
        return self.intrinsic_spectrum.value / self(self.waves)

class MultipleTelluricModels(Fittable1DModel):
    """
    This is a challenge, an attempt to fit multiple telluric spectra.
    """
    _param_names = ()

    def __init__(self, spectra, function="spline3", order=None,
                 name=None, meta=None, copy=False):
        #pca_components = spectra[0].pca.components
        #self.components = pca_components.copy() if copy else pca_components
        self.npca_components = len(spectra[0].pca.components)
        self.nspectra = len(spectra)
        self.npixels = np.array([tspek.waves.size for tspek in spectra])
        self.result = np.empty((self.npixels.sum(),))
        self.waves = np.asarray([tspek.waves for tspek in spectra]).flatten()
        self.dwaves = np.asarray([tspek.dwaves for tspek in spectra]).flatten()

        if isinstance(function, list):
            self.functions = function
        else:
            self.functions = [function] * self.nspectra
        try:
            order[0]
        except TypeError:
            self.orders = np.full((self.nspectra,), order)
        else:
            self.orders = np.asarray(order)

        self.models = []
        for tspek, func, ord in zip(spectra, self.functions, self.orders):
            self.models.append(SingleTelluricModel(
                tspek=tspek, function=func, order=ord))
        self.nparams = []
        self._param_names = self._generate_coeff_names()
        for param_name in self._param_names:
            self._parameters_[param_name] = Parameter(param_name,
                                                      default=0)
        super().__init__(name=name, meta=meta,
                         **dict(zip(self._param_names,
                                    np.zeros(len(self._param_names),))))

        # Avoid over/underflow errors by constraining continuum to lie
        # between 1e-25 and 1e+25
        #self.bounds.update({p: (-100, 100) for p in
        #                    self._param_names[self.npca_components-1:]})

    @property
    def param_names(self):
        return self._param_names

    def _generate_coeff_names(self):
        names = list(self.models[0].param_names[self.models[0].pca_params])
        for n, (func, ord) in enumerate(zip(self.functions, self.orders)):
            if func == "spline3":
                names.extend([f'm{n}spl{i:02d}' for i in range(ord + 3)])
                self.nparams.append(ord + 3)
            else:
                names.extend([f'm{n}cheb{i}' for i in range(ord + 1)])
                self.nparams.append(ord + 1)
        return tuple(names)

    def concatenate(self, property='data'):
        return np.asarray([getattr(tspek, property)
                           for tspek in self.spectra]).flatten()

    def evaluate(self, x, *params):
        start_pix = 0
        transmission_params = list(params[self.models[0].pca_params])
        start_param = len(transmission_params)
        good = get_good_pixels(x, self.waves)
        for m, nparam, npix in zip(self.models, self.nparams, self.npixels):
            these_params = (transmission_params +
                            list(params[start_param:start_param+nparam]))
            waves = m.waves[good[start_pix:start_pix+npix]]
            result = m.evaluate(waves, *these_params)
            self.result[start_pix:start_pix+npix][good[start_pix:start_pix+npix]] = result
            start_param += nparam
            start_pix += npix
        return self.result[good]

    def spectral_results(self):
        """Return the output as individual spectra"""
        result = self.evaluate(self.waves, *self.parameters)
        outputs = []
        start_pix = 0
        for npix in self.npixels:
            outputs.append(result[start_pix:start_pix+npix])
            start_pix += npix
        return outputs

    def update_individual_models(self):
        """
        Update the SingleTelluricModel instances with the fitted parameters.
        You can only update the parameters en masse, not bit-by-bit
        """
        npca_parameters = self.models[0].pca_params.stop
        start_param = npca_parameters
        for m, nparams in zip(self.models, self.nparams):
            new_params = np.empty((npca_parameters+nparams,))
            new_params[:npca_parameters] = self.parameters[:npca_parameters]
            new_params[npca_parameters:] = self.parameters[start_param:start_param+nparams]
            m.parameters = new_params
            start_param += nparams

    def fit_results(self):
        """
        Update the individual models with the best-fitting parameters and
        return these models plus the principal component strengths.

        Returns
        -------
        array: PCA parameters
        list: individual SingleTelluricModel instances
        """
        self.update_individual_models()
        npca_parameters = self.npca_components - 1  # because of the mean
        individual_models = []
        start_param = len(self.models[0].pca.param_names)  # includes LSF params
        for m, nparams in zip(self.models, self.nparams):
            these_params = self.parameters[start_param:start_param+nparams]
            this_model = m.continuum_function(these_params)
            # Correct the coefficients for the internal scaling parameter
            try:
                offset = -2.5 * np.log10(m.intrinsic_spectrum_scaling.to(u.Unit('erg cm-2 s-1 nm-1')).value)
            except AttributeError:  # no "to"
                pass
            else:
                try:
                    this_model.c0 += offset
                except AttributeError:
                    this_model.c += offset
            individual_models.append(this_model)
            start_param += nparams

        return self.parameters[:npca_parameters], individual_models


def get_good_pixels(x, waves):
    """
    Return a boolean array indicating which elements in the wavelength array
    were sent for evaluation. It uses the fact that "x" will be a sequence of
    elements of "waves" (so equality checking is OK, even though they're
    floats) and the sequence order is the same.

    Parameters
    ----------
    x: array
        elements to be evaluated
    waves: array
        full array of wavelengths for this model

    Returns
    -------
    boolean array of elements in waves that are in x
    """
    good = np.zeros_like(waves, dtype=np.uint16)
    cython_utils.get_unmasked(waves.astype(np.float32),
                              x.astype(np.float32), x.size, good)
    return good.astype(bool)


@models.custom_model
def Planck(w, temperature=10000., scale=1.):
    """Planck function returned in f_lambda units (W/m^2/nm)
    w *must* be a float (array) in units of nm, *not* a Quantity"""
    return scale * (2 * h * c**2 / (w * u.nm) ** 5 /
                    (np.exp(h * c/(w * u.nm * k_B * temperature * u.K)) -
                     1)).to(u.W / (u.m ** 2 * u.nm)).value

