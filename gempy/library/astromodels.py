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
# chebyshev_to_dict / dict_to_chebyshev: Turn a Chebyshev model into a dict to
#                                        assist with reading/writing as a Table
# make_inverse_chebyshev1d:              make a Chebyshev1D model that provides
#                                        the inverse of the given model

import numpy as np
import math
from collections import OrderedDict

import astropy
from astropy.modeling import models, fitting, FittableModel, Parameter
from astropy.stats import sigma_clip
from astropy.utils import minversion
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline, BSpline

from .nddops import NDStacker

ASTROPY_LT_40 = not minversion(astropy, '4.0')

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

    if ASTROPY_LT_40:
        inputs = ('x', 'y')
        outputs = ('x', 'y')
    else:
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
        super(Pix2Sky, self).__init__(x_offset, y_offset, factor, angle,
                                      **kwargs)

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

    if ASTROPY_LT_40:
        inputs = ('x', 'y')
        outputs = ('x', 'y')
    else:
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

    if ASTROPY_LT_40:
        inputs = ('x', 'y')
        outputs = ('x', 'y')
    else:
        n_inputs = 2
        n_outputs = 2

    factor = Parameter(default=1.0)

    def __init__(self, factor=1.0, **kwargs):
        super(Scale2D, self).__init__(factor, **kwargs)

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

    if ASTROPY_LT_40:
        inputs = ('x', 'y')
        outputs = ('x', 'y')
    else:
        n_inputs = 2
        n_outputs = 2

    angle = Parameter(default=0.0, getter=np.rad2deg, setter=np.deg2rad)

    def __init__(self, angle=0.0, **kwargs):
        super(Rotate2D, self).__init__(angle, **kwargs)

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
        s, c = math.sin(angle), math.cos(angle)
        x, y = np.dot(np.array([[c, -s], [s, c]], dtype=np.float64), inarr)
        x.shape = y.shape = orig_shape
        return x, y


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
    niter: int
        maximum number of clipping iterations to perform
    grow: int
        radius to reject pixels adjacent to masked pixels
    outlier_kwargs: dict-like
        parameter dict to pass to outlier_func()

    Returns
    -------
    BSpline object
        a callable to return the value of the interpolated spline
    """
    def __new__(cls, x, y, order=None, s=None, w=None, bbox=[None]*2, k=3,
                ext=0, check_finite=False, outlier_func=sigma_clip,
                niter=3, grow=0, debug=False, **outlier_kwargs):

        # Decide what sort of spline object we're making
        spline_kwargs = {'bbox': bbox, 'k': k, 'ext': ext,
                         'check_finite': check_finite}
        if order is None:
            cls_ = UnivariateSpline
            spline_args = ()
            spline_kwargs['s'] = s
        elif s is None:
            cls_ = LSQUnivariateSpline
        else:
            raise ValueError("Both t and s have been specified")

        # Both spline classes require sorted x, so do that here. We also
        # require unique x values, so we're going to deal with duplicates by
        # making duplicated values slightly larger. But we have to do this
        # iteratively in case of a basket-case scenario like (1, 1, 1, 1+eps, 2)
        # which would become (1, 1+eps, 1+2*eps, 1+eps, 2), which still has
        # duplicates and isn't sorted!
        # I can't think of any better way to cope with this, other than write
        # least-squares spline-fitting code that handles duplicates from scratch
        epsf = np.finfo(float).eps

        orig_mask = np.zeros(y.shape, dtype=bool)
        if isinstance(y, np.ma.masked_array):
            if y.mask is not np.ma.nomask:
                orig_mask = y.mask.astype(bool)
            y = y.data

        if w is not None:
            orig_mask |= (w == 0)

        if debug:
            print(y)
            print(orig_mask)

        iter = 0
        full_mask = orig_mask  # Will include pixels masked because of "grow"
        while iter < niter+1:
            last_mask = full_mask
            x_to_fit = x.astype(float)

            if order is not None:
                # Determine actual order to apply based on fraction of unmasked
                # pixels, and unmask everything if there are too few good pixels
                this_order = int(order * (1 - np.sum(full_mask) / len(full_mask)) + 0.5)
                if this_order == 0:
                    full_mask = np.zeros(x.shape, dtype=bool)
                    if w is not None and not all(w == 0):
                        full_mask |= (w == 0)
                    this_order = int(order * (1 - np.sum(full_mask) / len(full_mask)) + 0.5)
                    if debug:
                        print("FULL MASK", full_mask)

            xgood = x_to_fit[~full_mask]
            while True:
                xunique, indices = np.unique(xgood, return_index=True)
                if len(indices) == len(xgood):
                    # All unique x values so continue
                    break
                if order is None:
                    raise ValueError("Must specify spline order when there are "
                                     "duplicate x values")
                for i in range(len(xgood)):
                    if not (last_mask[i] or i in indices):
                        xgood[i] *= (1.0 + epsf)

            # Space knots equally based on density of unique x values
            if order is not None:
                knots = [xunique[int(xx+0.5)]
                         for xx in np.linspace(0, len(xunique)-1, this_order+1)[1:-1]]
                spline_args = (knots,)
                if debug:
                    print ("KNOTS", knots)

            sort_indices = np.argsort(xgood)
            # Create appropriate spline object using current mask
            try:
                spline = cls_(xgood[sort_indices], y[~full_mask][sort_indices],
                              *spline_args, w=None if w is None else w[~full_mask][sort_indices],
                              **spline_kwargs)
            except ValueError as e:
                if this_order == 0:
                    avg_y = np.average(y[~full_mask],
                                       weights=None if w is None else w[~full_mask])
                    spline = lambda xx: avg_y
                else:
                    raise e
            spline_y = spline(x)
            #masked_residuals = outlier_func(spline_y - masked_y, **outlier_kwargs)
            #mask = masked_residuals.mask

            # When sigma-clipping, only remove the originally-masked points.
            # Note that this differs from the astropy.modeling code because
            # the sigma-clipping and spline-fitting are done independently here.
            d, mask, v = NDStacker.sigclip(spline_y-y, mask=orig_mask, variance=None,
                                           **outlier_kwargs)
            if grow > 0:
                maskarray = np.zeros((grow * 2 + 1, len(y)), dtype=bool)
                for i in range(-grow, grow + 1):
                    mx1 = max(i, 0)
                    mx2 = min(len(y), len(y) + i)
                    maskarray[grow + i, mx1:mx2] = mask[:mx2 - mx1]
                grow_mask = np.logical_or.reduce(maskarray, axis=0)
                full_mask = np.logical_or(mask, grow_mask)
            else:
                full_mask = mask.astype(bool)

            # Check if the mask is unchanged
            if not np.logical_or.reduce(last_mask ^ full_mask):
                break
            iter += 1

        # Create a standard BSpline object
        try:
            bspline = BSpline(*spline._eval_args)
        except AttributeError:
            # Create a spline object that's just a constant
            bspline = BSpline(np.r_[(x[0],)*4, (x[-1],)*4],
                              np.r_[(spline(0),)*4, (0.,)*4], 3)
        # Attach the mask and model (may be useful)
        bspline.mask = full_mask
        bspline.data = spline_y
        return bspline


# -----------------------------------------------------------------------------
# MODEL -> DICT FUNCTIONS
#
def chebyshev_to_dict(model):
    """
    This function turns an instance of a ChebyshevND model into a dict of
    parameter and property names and their values. This allows it to be
    written as a Table and attached to an AstroData object. A Table is not
    constructed here because it may have additional rows added to it, and
    that's inefficient.

    Parameters
    ----------
    model: a ChebyshevND model instance

    Returns
    -------
    OrderedDict: property names and their values
    """
    if isinstance(model, models.Chebyshev1D):
        ndim = 1
        properties = ('degree', 'domain')
    elif isinstance(model, models.Chebyshev2D):
        ndim = 2
        properties = ('x_degree', 'y_degree', 'x_domain', 'y_domain')
    else:
        return {}

    model_dict = OrderedDict({'ndim': ndim})
    for property in properties:
        if 'domain' in property:
            domain = getattr(model, property)
            if domain is not None:
                model_dict['{}_start'.format(property)] = domain[0]
                model_dict['{}_end'.format(property)] = domain[1]
        else:
            model_dict[property] = getattr(model, property)
    for name in model.param_names:
        model_dict[name] = getattr(model, name).value

    return model_dict


def dict_to_chebyshev(model_dict):
    """
    This is the inverse of chebyshev_to_dict(), taking a dict of property/
    parameter names and their values and making a ChebyshevND model instance.

    Parameters
    ----------
    model_dict: dict
        Dictionary with pair/value that defines the Chebyshev model.

    Returns
    -------
    models.ChebyshevND or None
        Returns the models if it is parsed successfully. If not, it will return
        None.

    Examples
    --------

    .. code-block:: python

        my_model = dict(
            zip(
                ad[0].WAVECAL['name'], ad[0].WAVECAL['coefficient']
            )
        )

    """
    try:
        ndim = int(model_dict.pop('ndim'))
        if ndim == 1:
            model = models.Chebyshev1D(degree=int(model_dict.pop('degree')))
        elif ndim == 2:
            model = models.Chebyshev2D(x_degree=int(model_dict.pop('x_degree')),
                                       y_degree=int(model_dict.pop('y_degree')))
        else:
            return None
    except KeyError:
        return None

    for k, v in model_dict.items():
        try:
            if k.endswith('domain_start'):
                setattr(model, k.replace('_start', ''), [v, model_dict[k.replace('start', 'end')]])
            elif k and not k.endswith('domain_end'):  # ignore k==""
                setattr(model, k, v)
        except (KeyError, AttributeError):
            return None

    return model


def make_inverse_chebyshev1d(model, sampling=1, rms=None):
    """
    This creates a Chebyshev1D model that attempts to be the inverse of
    the model provided.

    Parameters
    ----------
    model: Chebyshev1D
        The model to be inverted
    rms: float/None
        required maximum rms in input space (i.e., pixels)
    """
    order = model.degree
    max_order = order if rms is None else order + 2
    incoords = np.arange(*model.domain, sampling)
    outcoords = model(incoords)
    while order <= max_order:
        m_init = models.Chebyshev1D(degree=order, domain=model(model.domain))
        fit_it = fitting.LinearLSQFitter()
        m_inverse = fit_it(m_init, outcoords, incoords)
        rms_inverse = np.std(m_inverse(outcoords) - incoords)
        if rms is None or rms_inverse <= rms:
            break
        order += 1
    return m_inverse
