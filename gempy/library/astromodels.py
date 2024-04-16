# Copyright(c) 2019-2020 Association of Universities for Research in Astronomy, Inc.
#
# astromodels.py
#
# This module contains classes and function to interface with the
# astropy.modeling module.
#
# New Model classes to aid with image transformations:
# Pix2Sky: allows modification of a WCS object as a FittableModel
# Rotate2D: a FittableModel version of Rotation2D
# Scale2D: single model to scale in 2D
# Shift2D: single model to shift in 2D
#
# Functions:
# model_to_table / table_to_model: allow certain types of models (including
#                                  splines) to be converted to/from Tables
# make_inverse_chebyshev1d:        make a Chebyshev1D model that provides
#                                  the inverse of the given model

import math
import re

import numpy as np
from astropy.modeling import FittableModel, Parameter, fitting, models
from astropy.modeling.core import Model, CompoundModel
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy import units as u
from astropy.io.fits import Header
from scipy.interpolate import splrep, BSpline

from gempy.utils import logutils

# -----------------------------------------------------------------------------
# NEW MODEL CLASSES


class Pix2Sky(FittableModel):
    """
    Wrapper to make an astropy.WCS object act like an astropy.modeling.Model
    object, including having an inverse.

    Parameters
    ----------
    wcs: astropy.wcs.WCS object
        the WCS object defining the transformation
    x_offset: float
        offset to apply to CRPIX1 value
    y_offset: float
        offset to apply to CRPIX2 value
    factor: float
        scaling factor (applied to CD matrix)
    angle: float
        rotation in degrees (applied to CD matrix)
    origin: int (0 or 1)
        value for WCS origin parameter
    """

    n_inputs = 2
    n_outputs = 2

    x_offset = Parameter()
    y_offset = Parameter()
    factor = Parameter()
    angle = Parameter()

    def __init__(self, wcs, x_offset=0.0, y_offset=0.0, factor=1.0,
                 angle=0.0, origin=1, **kwargs):
        self._wcs = wcs.deepcopy()
        self._direction = 1  # pix->sky direction
        self._origin = origin
        super().__init__(x_offset, y_offset, factor, angle, **kwargs)

    def evaluate(self, x, y, x_offset, y_offset, factor, angle):
        # x_offset and y_offset are actually arrays in the Model
        #temp_wcs = self.wcs(x_offset[0], y_offset[0], factor, angle)
        temp_wcs = self.wcs
        return temp_wcs.all_pix2world(x, y, self._origin) if self._direction > 0 \
            else temp_wcs.all_world2pix(x, y, self._origin)

    @property
    def inverse(self):
        inv = self.copy()
        inv._direction = -self._direction
        return inv

    @property
    def wcs(self):
        """Return the WCS modified by the translation/scaling/rotation"""
        wcs = self._wcs.deepcopy()
        x_offset = self.x_offset.value
        y_offset = self.y_offset.value
        angle = self.angle.value
        factor = self.factor.value
        wcs.wcs.crpix += np.array([x_offset, y_offset])
        if factor != 1:
            wcs.wcs.cd *= factor
        if angle != 0.0:
            m = models.Rotation2D(angle)
            wcs.wcs.cd = m(*wcs.wcs.cd)
        return wcs


class Shift2D(FittableModel):
    """2D translation"""

    n_inputs = 2
    n_outputs = 2

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
        return x + x_offset, y + y_offset


class Scale2D(FittableModel):
    """2D scaling"""

    n_inputs = 2
    n_outputs = 2

    factor = Parameter(default=1.0)

    def __init__(self, factor=1.0, **kwargs):
        super().__init__(factor, **kwargs)

    @property
    def inverse(self):
        inv = self.copy()
        inv.factor = 1.0 / self.factor
        return inv

    @staticmethod
    def evaluate(x, y, factor):
        return x * factor, y * factor


class Rotate2D(FittableModel):
    """Rotation; Rotation2D isn't fittable"""

    n_inputs = 2
    n_outputs = 2

    angle = Parameter(default=0.0, getter=np.rad2deg, setter=np.deg2rad)

    def __init__(self, angle=0.0, **kwargs):
        super().__init__(angle, **kwargs)

    @property
    def inverse(self):
        inv = self.copy()
        inv.angle = -self.angle
        return inv

    @staticmethod
    def evaluate(x, y, angle):
        if x.shape != y.shape:
            raise ValueError("Expected input arrays to have the same shape")
        orig_shape = x.shape or (1,)
        inarr = np.array([x.flatten(), y.flatten()])
        if np.isscalar(angle):
            s, c = math.sin(angle), math.cos(angle)
        else:
            s, c = math.sin(angle[0]), math.cos(angle[0])
        x, y = np.dot(np.array([[c, -s], [s, c]], dtype=np.float64), inarr)
        x.shape = y.shape = orig_shape
        return x, y


def make_serializable(transform):
    """
    Convert any of the models above into standard astropy models so
    the result is serializable by ASDF
    """
    replacements = []
    for i, m in enumerate(transform):
        replacement = None
        if m.__class__ == Rotate2D:
            replacement = models.Rotation2D(m.angle)
        elif m.__class__ == Shift2D:
            replacement = models.Shift(m.x_offset) & models.Shift(m.y_offset)
        elif m.__class__ == Scale2D:
            replacement = models.Scale(m.factor) & models.Scale(m.factor)
        if replacement:
            m.name = f"model{i}"
            replacements.append((m.name, replacement))

    for name, replacement in replacements:
        transform = transform.replace_submodel(name, replacement)
    return transform


class UnivariateSplineWithOutlierRemoval:
    """
    Instantiating this class creates a spline object that fits to the
    1D data, iteratively removing outliers using a specified function.
    A LSQUnivariateSpline() object will be used if the locations of
    the spline knots are specified, otherwise a UnivariateSpline() object
    will be used with the specified smoothing factor.

    Duplicate x values are allowed here in the case of a specified order,
    because the spline is an approximation and therefore does not need to
    pass through all the points. However, for the purposes of determining
    whether knots satisfy the Schoenberg-Whitney conditions, duplicates
    are treated as a single x-value.

    If an order is specified, it may be reduced proportionally to the
    number of unmasked pixels.

    Once the spline has been finalized, an identical BSpline object is
    created and returned.

    Parameters
    ----------
    x: array
        x-coordinates of datapoints to fit
    y: array/maskedarray
        y-coordinates of datapoints to fit (mask is used)
    order: int/None
        order of spline fit (if not using smoothing factor)
    s: float/None
        smoothing factor (see UnivariateSpline description)
    w: array
        weighting for each point
    bbox: (2,), array-like, optional
        x-coordinate region over which interpolation is valid
    k: int
        order of spline interpolation
    ext: int/str
        type of extrapolation outside bounding box
    check_finite: bool
        check whether input contains only finite numbers
    outlier_func: callable
        function to call for defining outliers
    niter: int, optional
        maximum number of clipping iterations to perform
    grow: int
        radius to reject pixels adjacent to masked pixels
    knot_spacing : str
        describes how the knots should be spaced if order is not None.
        "limits": equally in x between the min/max x values
        "points": so that there are the same number of points in each region
        "good": so that there are the same number of good points in each region
    downscale_order: bool
        reduce number of spline pieces in direct proportion to the fraction
        of initially masked pixels?
    outlier_kwargs: dict-like
        parameter dict to pass to outlier_func()

    Returns
    -------
    BSpline object
        a callable to return the value of the interpolated spline
    """
    def __new__(cls, x, y, order=None, s=None, w=None, bbox=[None]*2, k=3,
                ext=0, check_finite=True, outlier_func=sigma_clip,
                niter=0, grow=0, debug=False, knot_spacing="good",
                downscale_order=True, **outlier_kwargs):

        log = logutils.get_logger(__name__)

        if x.size == 0:  # bail out now
            log.warning("No data sent to spline fitter")
            spline = BSpline(np.r_[(0,) * 4, (1,) * 4],
                             np.r_[(0.,) * 4, (0.,) * 4], 3)
            spline.fit_info = {"rank": 0}
            spline.mask = False
            spline.data = np.zeros_like(y)
            return spline

        if niter is None:
            niter = 100  # really should converge by this point

        spline_kwargs = {"xb": bbox[0], "xe": bbox[1], "k": k, "s": s}
        if order is None:
            spline_kwargs["task"] = 0
        elif s is None:
            spline_kwargs["task"] = -1
        else:
            raise ValueError("Both order and s have been specified")

        # For compatibility with an older version which was using
        # NDStacker.sigclip, rename parameters for sigma_clip
        if 'lsigma' in outlier_kwargs:
            outlier_kwargs['sigma_lower'] = outlier_kwargs.pop('lsigma')
        if 'hsigma' in outlier_kwargs:
            outlier_kwargs['sigma_upper'] = outlier_kwargs.pop('hsigma')

        # Override sigma if we have upper/lower set.  Otherwise astropy treats
        # a 0.0 lower/upper sigma as a request to use sigma
        if outlier_kwargs.get('sigma_lower', None) is not None and \
            outlier_kwargs.get('sigma_upper', None) is not None:
            outlier_kwargs['sigma'] = 0.0

        epsf = np.finfo(float).eps

        orig_mask = np.zeros(y.shape, dtype=bool)
        if isinstance(y, np.ma.masked_array):
            if y.mask is not np.ma.nomask:
                orig_mask = y.mask.astype(bool)
            y = y.data

        # Setting a lot of weights to zero can cause problems with the spline
        # fitting, so instead we set weights to epsf. To ensure this is a
        # sufficiently small value to be zero-like, we scale the input weights
        # so that the smallest "real" weight is 1.0. Input points with a weight
        # of zero are masked
        if w is None:
            wts = np.ones_like(x)
        elif np.any(w < 0):
            raise ValueError("Weights should not be negative")
        else:
            orig_mask |= (w == 0)
            try:
                wmin = w[w > 0].min()
            except ValueError:  # all w==0
                wts = np.ones_like(x)
                wmin = 1.0
            else:
                wts = w / wmin

        if check_finite:
            if (not np.isfinite(x).all() or not np.isfinite(y).all() or
                    not np.isfinite(wts).all()):
                raise ValueError("Input arrays must not contain NaNs or Infs.")

        if debug:
            print('y=', y)
            print('orig_mask=', orig_mask.astype(int))

        if (order is None or s == 0) and (np.unique(x).size < x.size):
            raise ValueError("Must specify spline order or have s > 0 when "
                             "there are duplicate x values")
        xunique = x
        sort_indices = np.argsort(xunique)

        if (~orig_mask).sum() <= k + 1:
            log.warning("Too few unmasked points. Unmasking all data.")
            orig_mask[:] = False
        if order is not None:
            if downscale_order:
                order = int(order * (~orig_mask).sum() / orig_mask.size + 0.5)
            if order > (~orig_mask).sum() - k:
                order = max((~orig_mask).sum() - k, 0)
                log.warning("Underconstrained fit. Reducing number of spline "
                            f"pieces to {order}")

            if order > 0:
                if knot_spacing == "limits":
                    knots = np.linspace(xunique.min(), xunique.max(), order + 1)
                elif knot_spacing == "points":
                    knots = np.interp(np.linspace(0, xunique.size - 1, order + 1),
                                      range(xunique.size), xunique[sort_indices])
                elif knot_spacing == "good":
                    knots = np.interp(
                        np.linspace(0, xunique[~orig_mask].size - 1, order + 1),
                        range(xunique[~orig_mask].size), sorted(xunique[~orig_mask]))
                else:
                    raise ValueError(f"Unrecognized option: knot_spacing='{knot_spacing}'")
                spline_kwargs["t"] = knots[1:-1]
                if debug:
                    print("KNOTS", knots)

        iteration = 0
        full_mask = orig_mask  # Will include pixels masked because of "grow"
        while iteration < niter+1:
            # There's a problem if too many inter-knot regions have no data
            # with non-zero weights so we fix this by setting the weights to
            # epsf instead in such cases. Remember that the knots are in the
            # x-value space, not the x-index space!
            if order is not None:
                if order > 0:
                    fully_masked_regions = np.sum([
                        full_mask[np.logical_and(xunique>=x1, xunique<=x2)].all()
                        for x1, x2 in zip(knots[:-1], knots[1:])])
                    wts[full_mask] = epsf if fully_masked_regions > min(k, order) else epsf
                else:
                    wts = None if w is None else w.copy()

            last_mask = full_mask
            avg_y = np.average(y, weights=wts)
            rank = 0
            if order is None or order > 0:
                # Explicitly send s since we've modified the weights
                if spline_kwargs["task"] == 0:
                    spline_kwargs["s"] = len(wts[sort_indices]) / (wmin * wmin)
                tck = splrep(xunique[sort_indices], y[sort_indices],
                             w=wts[sort_indices], **spline_kwargs)
                spline = BSpline(*tck)
                rank = tck[0].size - (2 * tck[2] + 1)  # actual order used
                # Ensure we get a real-valued fit
                if np.isnan(tck[1]).any():
                    spline = lambda xx: avg_y
            else:
                spline = lambda xx: avg_y
            spline_y = spline(x)

            # on last pass, do not update sigma clipping
            if iteration >= niter:
                break

            masked_residuals = outlier_func(np.ma.array(y - spline_y,
                                                        mask=full_mask | np.isnan(spline_y)),
                                            **outlier_kwargs)
            mask = masked_residuals.mask

            if debug:
                print('mask=', mask.astype(int))

            if grow > 0:
                new_mask = mask ^ full_mask
                if new_mask.any():
                    for i in range(1, grow + 1):
                        mask[i:] |= new_mask[:-i]
                        mask[:-i] |= new_mask[i:]
                    if debug:
                        print('mask after growth=', mask.astype(int))

            if order is not None and order > (~mask).sum() - k:
                log.warning("Too many points rejected - "
                            f"exiting after {iteration} iterations")
                break

            full_mask = mask

            # Check if the mask is unchanged
            if not np.logical_or.reduce(last_mask ^ full_mask):
                if debug:
                    print(f"Iter {iteration}: Breaking")
                break

            if debug:
                print(f"Iter {iteration}: Starting new iteration")
            iteration += 1

        # Create a standard BSpline object
        if not isinstance(spline, BSpline):
            # Create a spline object that's just a constant
            if len(x) > 1:
                spline = BSpline(np.r_[(x[0],) * 4, (x[-1],) * 4],
                                 np.r_[(spline(0),) * 4, (0.,) * 4], 3)
            else:
                spline = BSpline(np.r_[(x[0],) * 4, (x[0]+1,) * 4],
                                 np.r_[(spline(0),) * 4, (0.,) * 4], 3)
        # Attach the mask and model (may be useful)
        spline.mask = full_mask
        spline.data = spline_y
        spline.fit_info = {"rank": rank}
        return spline


# -----------------------------------------------------------------------------
# MODEL <-> TABLE FUNCTIONS
#
def model_to_table(model, xunit=None, yunit=None, zunit=None):
    """
    Convert a model instance to a Table, suitable for attaching to an AstroData
    object or interrogating.

    Parameters
    ----------
    model : a callable function , either a `~astropy.modeling.core.Model`
        or a `~scipy.interpolate.BSpline` instance
    xunit, yunit, zunit : Unit or None
        unit of each axis (y axis may be dependent or independent variable)

    Returns
    -------
    Table : a Table describing the model, with some information in the
        meta["header"] dict
    """
    # Override existing model units with new ones if requested
    meta = getattr(model, "meta", {})
    if xunit:
        meta["xunit"] = xunit
    if yunit:
        meta["yunit"] = yunit
    if zunit:
        meta["zunit"] = zunit

    model_class = model.__class__.__name__
    if isinstance(model, Model):
        header = Header({"MODEL": model_class})
        ndim = model.n_inputs
        if ndim == 1:
            if getattr(model, "domain", None) is not None:
                header.update({"DOMAIN_START": model.domain[0],
                               "DOMAIN_END": model.domain[1]})
        elif ndim == 2:
            if getattr(model, "x_domain", None) is not None:
                header.update({"XDOMAIN_START": model.x_domain[0],
                               "XDOMAIN_END": model.x_domain[1]})
            if getattr(model, "y_domain", None) is not None:
                header.update({"YDOMAIN_START": model.y_domain[0],
                               "YDOMAIN_END": model.y_domain[1]})
        else:
            raise ValueError(f"Cannot handle model class '{model_class}' "
                             f"with dimensionality {ndim}")
        table = Table(model.parameters, names=model.param_names)
        for unit in ("xunit", "yunit", "zunit"):
            if unit in meta:
                header[unit.upper()] = (str(meta[unit]),
                                        f"Units of {unit[0]} axis")
    elif isinstance(model, BSpline):
        knots, coeffs, order = model.tck
        header = Header({"MODEL": f"spline{order}"})
        table = Table([knots, coeffs],
                      names=("knots", "coefficients"),
                      units=(meta.get("xunit"), meta.get("yunit")))
    else:
        raise TypeError(f"Cannot convert object of class '{model_class}'")

    table.meta["header"] = header
    return table


def table_to_model(table):
    """
    Convert a Table instance, as created by model_to_table(), back into a
    callable function. Some backward compatibility has been introduced, so
    the domain can be specified in the Table, rather than the meta, and a
    Chebyshev1D model will be assumed if not found in the meta.

    Parameters
    ----------
    table : `~astropy.table.Table` or `~astropy.table.Row`
        Table describing the model

    Returns
    -------
    callable : either a `~astropy.modeling.core.Model` or a
               `~scipy.interpolate.BSpline` instance
    """
    meta = table.meta["header"]
    try:
        model_class = meta['MODEL']
    except KeyError:
        if 'knots' in table.colnames:
            order = meta.get("ORDER", 3)
            model_class = f"SPLINE{order}"
        else:
            model_class = "Chebyshev1D"
    try:
        cls = getattr(models, model_class)
    except:  # it's a spline
        k = int(model_class[-1])
        knots, coeffs = table["knots"], table["coefficients"]
        model = BSpline(knots.data, coeffs.data, k)
        setattr(model, "meta", {"xunit": knots.unit,
                                "yunit": coeffs.unit})
    else:
        if isinstance(table, Table):
            if len(table) != 1:
                raise ValueError("Can only convert single-row Tables to a model")
            else:
                table = table[0]  # now a Row
        ndim = int(model_class[-2])
        table_dict = dict(zip(table.colnames, table))
        if ndim == 1:
            r = re.compile("c([0-9]+)")
            param_names = list(filter(r.match, table.colnames))
            # Handle cases (e.g., APERTURE tables) where the number of
            # columns must be the same for all rows but the degree of
            # polynomial might be different
            degree = max([int(r.match(p).groups()[0]) for p in param_names
                          if table[p] is not np.ma.masked])
            domain = [table_dict.get("domain_start", meta.get("DOMAIN_START", 0)),
                      table_dict.get("domain_end", meta.get("DOMAIN_END", 1))]
            model = cls(degree=degree, domain=domain)
        elif ndim == 2:
            r = re.compile("c([0-9]+)_([0-9]+)")
            param_names = list(filter(r.match, table.colnames))
            xdegree = max([int(r.match(p).groups()[0]) for p in param_names])
            ydegree = max([int(r.match(p).groups()[1]) for p in param_names])
            xdomain = [table_dict.get("xdomain_start", meta.get("XDOMAIN_START", 0)),
                       table_dict.get("xdomain_end", meta.get("XDOMAIN_END", 1))]
            ydomain = [table_dict.get("ydomain_start", meta.get("YDOMAIN_START", 0)),
                       table_dict.get("ydomain_end", meta.get("YDOMAIN_END", 1))]
            model = cls(x_degree=xdegree, y_degree=ydegree,
                        x_domain=xdomain, y_domain=ydomain)
        else:
            raise ValueError(f"Invalid dimensionality of model '{model_class}'")

        for k, v in table_dict.items():
            if k in param_names:
                setattr(model, k, v)
            elif not ("domain" in k or k in ("ndim", "degree")):
                # other columns go in the meta
                model.meta[k] = v
        for unit in ("xunit", "yunit", "zunit"):
            value = meta.get(unit.upper())
            if value:
                model.meta[unit] = u.Unit(value)

    return model


def make_inverse_chebyshev1d(model, sampling=1, rms=None, max_deviation=None):
    """
    This creates a Chebyshev1D model that attempts to be the inverse of
    a specified model that maps from an input space (e.g., pixels) to an
    output space (e.g., wavelength).

    Parameters
    ----------
    model: Chebyshev1D
        The model to be inverted
    sampling: int
        Frequency at which to sample the input coordinate space
    rms: float/None
        required maximum rms in input space
    max_deviation: float/None
        required maximum absolute deviation in input space
    """
    order = model.degree
    max_order = order if (rms is None and max_deviation is None) else order + 2
    incoords = np.arange(*model.domain, sampling)
    outcoords = model(incoords)
    while order <= max_order:
        m_init = models.Chebyshev1D(degree=order, domain=model(model.domain))
        fit_it = fitting.LinearLSQFitter()
        m_inverse = fit_it(m_init, outcoords, incoords)
        trans_coords = m_inverse(outcoords)
        rms_inverse = np.std(trans_coords - incoords)
        max_dev = np.max(abs(trans_coords - incoords))
        if ((rms is None or rms_inverse <= rms) and
                (max_deviation is None or max_dev <= max_deviation)):
            break
        order += 1
    return m_inverse


def get_named_submodel(model, name):
    """
    Extracts a named submodel from a CompoundModel. astropy allows
    CompoundModel instances to be indexed by name, but only single Models,
    whereas a submodel can be a CompoundModel.

    Parameters
    ----------
    model : CompoundModel
        the model containing the required submodel
    name : str
        name of the submodel requested

    Returns
    -------
    Model : the Model/CompoundModel instance with the requested name
    """
    if not isinstance(model, CompoundModel):
        raise TypeError("This is not a CompoundModel")
    if model._leaflist is None:
        model._make_leaflist()
    found = []
    for nleaf, leaf in enumerate(model._leaflist):
        if getattr(leaf, 'name', None) == name:
            found.append(nleaf)
    for m, start, stop in model._tdict.values():
        if getattr(m, 'name', None) == name:
            found.append(slice(start, stop+1))
    if len(found) == 0:
        raise IndexError("No component with name '{}' found".format(name))
    if len(found) > 1:
        raise IndexError("Multiple components found using '{}' as name\n"
                         "at indices {}".format(name, found))
    return model[found[0]]
