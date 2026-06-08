import numpy as np

from astropy.modeling import fitting, Parameter, polynomial
from astropy.modeling.utils import _validate_domain_window, poly_map_domain


class Chebyshev3D(polynomial.PolynomialBase):
    """A 3D Chebyshev polynomial, with coefficients named c{i}_{j}_{k}
    This is almost verbatim taken from Chebyshev2D with changes to
    accommodate a third input variable/axis. It does not have complete
    functionality of Chebyshev2D (e.g. it does not support fitting multiple
    models at once) but it is sufficient for our purposes."""
    n_inputs = 3
    n_outputs = 1

    def __init__(self, x_degree, y_degree, z_degree,
                 x_domain=None, x_window=None,
                 y_domain=None, y_window=None,
                 z_domain=None, z_window=None, n_models=None,
                 model_set_axis=None, name=None, meta=None, **params):
        self.x_degree = x_degree
        self.y_degree = y_degree
        self.z_degree = z_degree
        self._order = self.get_num_coeff()

        self._default_domain_window = {
            "x_window": (-1, 1),
            "y_window": (-1, 1),
            "z_window": (-1, 1),
            "x_domain": None,
            "y_domain": None,
            "z_domain": None,
        }
        self.x_window = x_window or self._default_domain_window["x_window"]
        self.y_window = y_window or self._default_domain_window["y_window"]
        self.z_window = z_window or self._default_domain_window["z_window"]
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.z_domain = z_domain

        self._param_names = self._generate_coeff_names()
        if n_models:
            raise NotImplementedError("Only one model can be fitted.")
        else:
            minshape = ()

        for param_name in self._param_names:
            self._parameters_[param_name] = Parameter(
                param_name, default=np.zeros(minshape)
            )
        super().__init__(
            n_models=n_models,
            model_set_axis=model_set_axis,
            name=name,
            meta=meta,
            **params,
        )

    def __repr__(self):
        return self._format_repr(
            [self.x_degree, self.y_degree, self.z_degree],
            kwargs={
                "x_domain": self.x_domain,
                "y_domain": self.y_domain,
                "z_domain": self.z_domain,
                "x_window": self.x_window,
                "y_window": self.y_window,
                "z_window": self.z_window,
            },
            defaults=self._default_domain_window,
        )

    def __str__(self):
        return self._format_str(
            [
                ("X_Degree", self.x_degree),
                ("Y_Degree", self.y_degree),
                ("Z_Degree", self.z_degree),
                ("X_Domain", self.x_domain),
                ("Y_Domain", self.y_domain),
                ("Z_Domain", self.z_domain),
                ("X_Window", self.x_window),
                ("Y_Window", self.y_window),
                ("Z_Window", self.z_window),
            ],
            self._default_domain_window,
        )

    @property
    def z_domain(self):
        return self._z_domain

    @z_domain.setter
    def z_domain(self, val):
        self._z_domain = _validate_domain_window(val)

    @property
    def z_window(self):
        return self._z_window

    @z_window.setter
    def z_window(self, val):
        self._z_window = _validate_domain_window(val)

    def get_num_coeff(self):
        if self.x_degree<0 or self.y_degree<0 or self.z_degree<0:
            raise ValueError("Degree of polynomial must be positive or null")
        return (self.x_degree+1) * (self.y_degree+1) * (self.z_degree+1)

    def _generate_coeff_names(self):
        names = []
        for k in range(self.z_degree + 1):
            for j in range(self.y_degree + 1):
                for i in range(self.x_degree + 1):
                    names.append(f"c{i}_{j}_{k}")
        return tuple(names)

    def _fcache(self, x, y, z):
        """
        Calculate the individual Chebyshev functions once and store them in a
        dictionary to be reused.
        """
        x_terms = self.x_degree + 1
        y_terms = self.y_degree + 1
        z_terms = self.z_degree + 1
        kfunc = {}
        kfunc[0, 0] = np.ones(x.shape)
        kfunc[0, 1] = x.copy()
        kfunc[1, 0] = np.ones(y.shape)
        kfunc[1, 1] = y.copy()
        kfunc[2, 0] = np.ones(z.shape)
        kfunc[2, 1] = z.copy()
        for n in range(2, x_terms):
            kfunc[0, n] = 2 * x * kfunc[0, n - 1] - kfunc[0, n - 2]
        for n in range(2, y_terms):
            kfunc[1, n] = 2 * y * kfunc[1, n - 1] - kfunc[1, n - 2]
        for n in range(2, z_terms):
            kfunc[2, n] = 2 * z * kfunc[2, n - 1] - kfunc[2, n - 2]
        return kfunc

    def fit_deriv(self, x, y, z, *params):
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        x_deriv = self._chebderiv1d(x, self.x_degree+1).T
        y_deriv = self._chebderiv1d(y, self.y_degree+1).T
        z_deriv = self._chebderiv1d(z, self.z_degree+1).T
        ijk = []
        for i in range(self.z_degree+1):
            for j in range(self.y_degree+1):
                for k in range(self.x_degree+1):
                    ijk.append(x_deriv[k] * y_deriv[j] * z_deriv[i])
        return np.array(ijk).T

    def evaluate(self, x, y, z, *coeffs):
        if self.x_domain is not None:
            x = poly_map_domain(x, self.x_domain, self.x_window)
        if self.y_domain is not None:
            y = poly_map_domain(y, self.y_domain, self.y_window)
        if self.z_domain is not None:
            z = poly_map_domain(z, self.z_domain, self.z_window)
        kfunc = self._fcache(x, y, z)
        result = np.zeros_like(x)
        for k in range(self.z_degree + 1):
            for j in range(self.y_degree + 1):
                for i in range(self.x_degree + 1):
                    name = f"c{i}_{j}_{k}"
                    coeff = coeffs[self.param_names.index(name)]
                    result += coeff * kfunc[0, i] * kfunc[1, j] * kfunc[2, k]
        return result

    def prepare_inputs(self, x, y, z, **kwargs):
        inputs, broadcasted_shapes = super().prepare_inputs(x, y, z, **kwargs)
        x, y, z = inputs
        if x.shape != y.shape != z.shape:
            raise ValueError("Expected input arrays to have the same shape")
        return (x, y, z), broadcasted_shapes

    def _chebderiv1d(self, x, deg):
        x = np.array(x, dtype=float, copy=None, ndmin=1)
        d = np.empty((deg + 1, len(x)), dtype=x.dtype)
        d[0] = x * 0 + 1
        if deg > 0:
            x2 = 2 * x
            d[1] = x
            for i in range(2, deg + 1):
                d[i] = d[i - 1] * x2 - d[i - 2]
        return np.rollaxis(d, 0, d.ndim)


class Fitter3D(fitting.LinearLSQFitter):
    supported_constraints = ["fixed"]
    supports_masked_input = True

    def __init__(self, calc_uncertainties=False):
        assert not calc_uncertainties  # for now
        super().__init__(calc_uncertainties=calc_uncertainties)

    def _map_domain_window(self, model, x, y, z):
        """
        Maps domain into window for a polynomial model which has these
        attributes.
        """
        if model.x_domain is None:
            model.x_domain = [x.min(), x.max()]
        if model.y_domain is None:
            model.y_domain = [y.min(), y.max()]
        if model.z_domain is None:
            model.z_domain = [z.min(), z.max()]
        if model.x_window is None:
            model.x_window = [-1.0, 1.0]
        if model.y_window is None:
            model.y_window = [-1.0, 1.0]
        if model.z_window is None:
            model.z_window = [-1.0, 1.0]
        xnew = poly_map_domain(x, model.x_domain, model.x_window)
        ynew = poly_map_domain(y, model.y_domain, model.y_window)
        znew = poly_map_domain(z, model.z_domain, model.z_window)
        return xnew, ynew, znew

    def __call__(self, model, x, y, z, data, weights=None, rcond=None,
                 *, inplace=False):
        assert model.fittable and model.linear and model.n_inputs == 3
        assert len(model) == 1  # no model sets
        fitting._validate_constraints(self.supported_constraints, model)

        model_copy = model if inplace else model.copy()
        model_copy.sync_constraints = False
        _, fit_param_indices, _ = fitting.model_to_fit_params(model_copy)

        # This performs the relevant part of _convert_input()
        x = np.asanyarray(x, dtype=float)
        y = np.asanyarray(y, dtype=float)
        z = np.asanyarray(z, dtype=float)
        data = np.asanyarray(data, dtype=float)
        assert x.shape == y.shape == z.shape == data.shape

        n_fixed = sum(model_copy.fixed.values())
        if n_fixed:
            fixparam_indices = [
                idx
                for idx in range(len(model_copy.param_names))
                if idx not in fit_param_indices
            ]
            fixparams = np.asarray(
                [
                    getattr(model_copy, model_copy.param_names[idx]).value
                    for idx in fixparam_indices
                ]
            )

        x, y, z = self._map_domain_window(model_copy, x, y, z)
        if weights is not None:
            weights = np.asanyarray(weights, dtype=float)
            assert weights.shape == x.shape

        if n_fixed:
            lhs = np.asarray(self._deriv_with_constraints(
                model_copy, fit_param_indices, x=x, y=y, z=z))
            fixderivs = self._deriv_with_constraints(
                model_copy, fixparam_indices, x=x, y=y, z=z)
        else:
            lhs = np.asanyarray(model_copy.fit_deriv(
                x, y, z, *model_copy.parameters))
        rhs = data.ravel()
        if weights is not None:
            weights = weights.ravel()

        sum_of_implicit_terms = model_copy.sum_of_implicit_terms()
        if model_copy.col_fit_deriv:
            lhs = np.asarray(lhs).T
        if np.asanyarray(lhs).ndim > 2:
            raise ValueError(
                f"{type(model_copy).__name__} gives unsupported >2D "
                "derivative matrix for this x/y"
            )

        if n_fixed:
            if model_copy.col_fit_deriv:
                fixderivs = np.asarray(fixderivs).T  # as for lhs above
            rhs = rhs - fixderivs.dot(fixparams)

        if sum_of_implicit_terms is not None:
            rhs = rhs - sum_of_implicit_terms

        if weights is not None:
            lhs *= weights[:, np.newaxis]
            rhs = rhs * weights

        scl = (lhs * lhs).sum(0)
        lhs /= scl

        masked = np.any(np.ma.getmask(rhs))
        if weights is not None and not masked and np.any(np.isnan(lhs)):
            raise ValueError(
                "Found NaNs in the coefficient matrix, which "
                "should not happen and would crash the lapack "
                "routine. Maybe check that weights are not null."
            )

        a = None  # need for calculating covariance

        good = ~rhs.mask if masked else slice(None)
        a = lhs[good]
        lacoef, resids, rank, sval = np.linalg.lstsq(lhs[good], rhs[good],
                                                     rcond)

        self.fit_info["residuals"] = resids
        self.fit_info["rank"] = rank
        self.fit_info["singular_values"] = sval

        lacoef /= scl[:, np.newaxis] if scl.ndim < rhs.ndim else scl
        self.fit_info["params"] = lacoef

        fitting.fitter_to_model_params(model_copy, lacoef.ravel())

        model_copy.sync_constraints = True
        return model_copy
