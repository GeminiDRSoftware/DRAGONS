# nddops.py -- operations on NDData objects
#
# This module contains the NDStacker class, used to combine multiple NDData
# objects with arbitrary rejector/combiner functions. A bad pixel mask can
# either be accepted as a boolean array or with the individual bits having
# different meanings that can be ordered into a hierarchy of "badness". For
# each output pixel, the least bad input pixels will be used, and the output
# pixel will be flagged accordingly.
#
# The methods of this class are static so they can be called on any dataset
# to reduce its dimensionality by one.

from __future__ import print_function

import numpy as np
from functools import wraps
from astrodata import NDAstroData
from geminidr.gemini.lookups import DQ_definitions as DQ
try:
    from . import cyclip
except ImportError:
    raise ImportError("Run 'cythonize -i cyclip.pyx' in gempy/library")

# Some definitions. Non-linear and saturated pixels are not considered to
# be "bad" when rejecting pixels from the input data. If one takes multiple
# images of an object and one of those images is saturated, it would clearly
# be wrong statistically to reject the saturated one.
BAD = 65535 ^ (DQ.non_linear | DQ.saturated)

# A hierarchy of "badness". Pixels in the inputs are considered to be as
# bad as the worst bit set, so a "bad_pixel" will only be used if there are
# no "overlap" or "cosmic_ray" pixels that can be used. Worst pixels are
# listed first.
DQhierarchy = (DQ.no_data, DQ.unilluminated, DQ.bad_pixel, DQ.overlap,
               DQ.cosmic_ray, DQ.non_linear | DQ.saturated)
ZERO = DQ.datatype(0)
ONE = DQ.datatype(DQ.bad_pixel)

import inspect

def take_along_axis(arr, ind, axis):
    """
    Returns a view of an array (arr), re-ordered along an axis according to
    the indices (ind). Shamelessly stolen from StackOverflow.
    """
    if arr is None:
        return None
    if axis < 0:
       if axis >= -arr.ndim:
           axis += arr.ndim
       else:
           raise IndexError('axis out of range')
    ind_shape = (1,) * ind.ndim
    ins_ndim = ind.ndim - (arr.ndim - 1)   #inserted dimensions

    dest_dims = list(range(axis)) + [None] + list(range(axis+ins_ndim, ind.ndim))

    # could also call np.ix_ here with some dummy arguments, then throw those results away
    inds = []
    for dim, n in zip(dest_dims, arr.shape):
        if dim is None:
            inds.append(ind)
        else:
            ind_shape_dim = ind_shape[:dim] + (-1,) + ind_shape[dim+1:]
            inds.append(np.arange(n).reshape(ind_shape_dim))
    return arr[tuple(inds)]

# This code was shamelessly stolen from StackOverflow (user Ants Aasma)
# It allows a decorator to be used on a function *or* a method, hiding
# the class instance from the decorator, which can simply assume it's been
# given a function. This is important for us because we plan to have our
# averaging functions as __call__ methods of classes.
class _MethodDecoratorAdaptor(object):
    def __init__(self, decorator, func):
        self.decorator = decorator
        self.func = func
    def __call__(self, *args, **kwargs):
        return self.decorator(self.func)(*args, **kwargs)
    def __get__(self, instance, owner):
        return self.decorator(self.func.__get__(instance, owner))

def auto_adapt_to_methods(decorator):
    """Allows you to use the same decorator on methods and functions,
    hiding the self argument from the decorator."""
    def adapt(func):
        return _MethodDecoratorAdaptor(decorator, func)
    return adapt

@auto_adapt_to_methods
def unpack_nddata(fn):
    """
    This decorator wraps a function that takes a sequence of NDAstroData
    objects and stacks them into data, mask, and variance arrays of one
    higher dimension, which get passed to the wrapped function.
    It also applies a set of scale factors and/or offsets, if supplied,
    to the raw data before stacking and passing them on.
    The returned arrays are then stuffed back into an NDAstroData object.
    """
    @wraps(fn)
    def wrapper(sequence, scale=None, zero=None, *args, **kwargs):
        nddata_list = list(sequence)
        if scale is None:
            scale = [1.0] * len(nddata_list)
        if zero is None:
            zero = [0.0] * len(nddata_list)
        # Coerce all data to 32-bit floats. FITS data on disk is big-endian
        # and preserving that datatype will cause problems with Cython
        # stacking if the compiler is little-endian.
        dtype = np.float32
        data = np.empty((len(nddata_list),)+nddata_list[0].data.shape, dtype=dtype)
        for i, (ndd, s, z) in enumerate(zip(nddata_list, scale, zero)):
            data[i] = ndd.data * s + z
        if any(ndd.mask is None for ndd in nddata_list):
            mask = None
        else:
            mask = np.empty_like(data, dtype=DQ.datatype)
            for i, ndd in enumerate(nddata_list):
                mask[i] = ndd.mask
        if any(ndd.variance is None for ndd in nddata_list):
            variance = None
        else:
            variance = np.empty_like(data)
            for i, (ndd, s, z) in enumerate(zip(nddata_list, scale, zero)):
                variance[i] = ndd.variance * s*s
        out_data, out_mask, out_var = fn(data=data, mask=mask,
                                    variance=variance, *args, **kwargs)

        # Can't instantiate NDAstroData with variance
        ret_value = NDAstroData(out_data, mask=out_mask)
        if out_var is not None:
            ret_value.variance = out_var
        return ret_value
    return wrapper

# Decorators to identify methods for combining and rejecting. Note that
# NDStacker.<method>.required_args will return a list of required arguments
def combiner(fn):
    fn.is_combiner = True
    args = inspect.getargspec(fn).args[3:]
    fn.required_args = list(args)
    return fn

def rejector(fn):
    fn.is_rejector = True
    args = inspect.getargspec(fn).args[3:]
    fn.required_args = list(args)
    return fn


class NDStacker(object):
    # Base class from which all stacking functions should subclass.
    # Put helper functions here so they can be inherited.
    def __init__(self, combine='mean', reject='none', log=None, **kwargs):
        self._log = log
        try:
            combiner = getattr(self, combine)
            assert getattr(combiner, 'is_combiner')
        except AttributeError:
            self._logmsg("No such combiner as {}. Using mean instead.".format(combine),
                         level='warning')
            combiner = self.mean
        # No combine functions require arguments (yet) but futureproofing
        req_args = getattr(combiner, 'required_args', [])
        self._combiner = combiner
        self._dict = {k: v for k, v in kwargs.items() if k in req_args}

        try:
            rejector = getattr(self, reject)
            assert getattr(rejector, 'is_rejector')
        except AttributeError:
            self._logmsg('No such rejector as {}. Using none instead.'.format(reject),
                         level='warning')
            rejector = self.none
        req_args = getattr(rejector, 'required_args', [])
        self._rejector = rejector
        self._dict.update({k: v for k, v in kwargs.items() if k in req_args})

        # Pixel to trace for debugging purposes
        self._debug_pixel = kwargs.get('debug_pixel')

    def _logmsg(self, msg, level='stdinfo'):
        """
        Logging function. Prints to screen if no logger is available.
        """
        if self._log is None:
            print(msg)
        else:
            getattr(self._log, level)(msg)

    @staticmethod
    def _process_mask(mask):
        """
        Interpret and manipulate the mask arrays of the input images to
        select which pixels should be combined, and what the output mask pixel
        should be.

        It works through the DQhierarchy, allowing pixels with increasing
        levels of badness. After each iteration, output pixels which have at
        least one input pixel have their input mask pixels reset to either 0
        (use to calculate output) or 32768 (don't use). This is a slightly
        hacky way to do things but we can use the 32768 bit to indicate that
        we don't need to process this particular pixel any more, even if we
        have to continue to iterate in order to get inputs for other output
        pixels.

        Parameters
        ----------
        mask: N x (input_shape) array
            the input masks from all the images, combined to have one higher
            dimension

        Returns
        -------
        mask: N x (input_shape) array
            modified version of the mask where only pixels with the value zero
            should be used to calculate the output pixel value
        out_mask: (input_shape) array
            output mask
        """
        if mask is None:
            return None, None

        # It it's a boolean mask we don't need to do much
        if mask.dtype == bool:
            out_mask = np.bitwise_and.reduce(mask, axis=0)
            mask ^= out_mask  # Set mask=0 if all pixels have mask=1
            return mask, out_mask

        consider_all = ZERO
        out_mask = np.full(mask.shape[1:], 0, dtype=DQ.datatype)
        for consider_bits in reversed(DQhierarchy):
            consider_all |= consider_bits
            tmp_mask = (mask & consider_all != mask)
            out_mask |= (np.bitwise_or.reduce(np.where(tmp_mask, ZERO, mask), axis=0))
            ngood = NDStacker._num_good(tmp_mask)

            # Where we've been able to construct an output pixel (ngood>0)
            # we need to stop any further processing. Set the mask for "good"
            # pixels to 0, and for bad pixels to 32768.
            mask = np.where(np.logical_and(ngood>0, tmp_mask), DQ.datatype(32768), mask)
            mask = np.where(np.logical_and(ngood>0, ~tmp_mask), ZERO, mask)
            # 32768 in output mask means we have an output pixel
            out_mask[ngood>0] |= 32768

            # If we've found "good' pixels for all output pixels, leave
            if np.all(out_mask & 32768):
                break

        return mask, out_mask & 32767

    @staticmethod
    def calculate_variance(data, mask, out_data):
        # gemcombine-style estimate of variance about the returned value
        ngood = data.shape[0] if mask is None else NDStacker._num_good(mask)
        return NDStacker._divide0(np.ma.masked_array(np.square(data - out_data),
                                            mask=mask).sum(axis=0).data.astype(data.dtype), ngood*(ngood-1))

    @staticmethod
    def _num_good(mask):
        # Return the number of unflagged pixels at each output pixel
        return np.sum(mask==False, axis=0)

    @staticmethod
    def _divide0(numerator, denominator):
        # Division with divide-by-zero catching
        return np.divide(numerator, denominator,
                         out=np.zeros_like(numerator), where=(denominator!=0))

    @unpack_nddata
    def __call__(self, data, mask=None, variance=None):
        """
        Perform the rejection and combining. The unpack_nddata decorator
        allows a series of NDData object to be sent, and split into data, mask,
        and variance.
        """
        rej_args = {arg: self._dict[arg] for arg in self._rejector.required_args
                    if arg in self._dict}

        # Convert the debugging pixel to (x,y) coords and bounds check
        if self._debug_pixel is not None:
            pixel_coords = []
            for length in reversed(data.shape[1:-1]):
                pixel_coords.append(self._debug_pixel % length)
                self._debug_pixel //= length
            self._debug_pixel = tuple(reversed(pixel_coords + [self._debug_pixel]))
            if self._debug_pixel[0] > data.shape[1]:
                self._logmsg("Debug pixel out of range")
                self._debug_pixel = None
            else:
                self._logmsg("Debug pixel coords {}".format(self._debug_pixel))
                self._pixel_debugger(data, mask, variance, stage='at input')
                self._logmsg("Rejection: {} {}".format(self._rejector.__name__, rej_args))

        data, mask, variance = self._rejector(data, mask, variance, **rej_args)
        #try:
        #    data, mask, variance = self._rejector(data, mask, variance, **rej_args)
        #except Exception as e:
        #    self._logmsg(str(e), level='warning')
        #    self._logmsg("Continuing without pixel rejection")
        #    self._rejector = self.none
        comb_args = {arg: self._dict[arg] for arg in self._combiner.required_args
                     if arg in self._dict}

        if self._debug_pixel is not None:
            self._pixel_debugger(data, mask, variance, stage='after rejection')
            self._logmsg("Combining: {} {}".format(self._combiner.__name__, comb_args))
        out_data, out_mask, out_var = self._combiner(data, mask, variance, **comb_args)
        #self._pixel_debugger(data, mask, variance, stage='combined')
        if self._debug_pixel is not None:
            info = [out_data[self._debug_pixel]]
            info.append(None if out_mask is None else out_mask[self._debug_pixel])
            info.append(None if out_var is None else out_var[self._debug_pixel])
            self._logmsg("out {:15.4f} {:5d} {:15.4f}".format(*info))
        return out_data, out_mask, out_var


    def _pixel_debugger(self, data, mask, variance, stage=''):
        self._logmsg("img     data        mask    variance       "+stage)
        for i in range(data.shape[0]):
            coords = (i,) + self._debug_pixel
            info = [data[coords]]
            info.append(None if mask is None else mask[coords])
            info.append(None if variance is None else variance[coords])
            self._logmsg("{:3d} {:15.4f} {:5d} {:15.4f}".format(i, *info))
        self._logmsg("-" * 41)

    #------------------------ COMBINER METHODS ----------------------------
    # These methods must all return data, mask, and varianace arrays of one
    # lower dimension than the input, with the valid (mask==0) input pixels
    # along the axis combined to produce a single output pixel.

    @staticmethod
    @combiner
    def mean(data, mask=None, variance=None):
        # Regular arithmetic mean
        mask, out_mask = NDStacker._process_mask(mask)
        out_data = np.ma.masked_array(data, mask=mask).mean(axis=0).data.astype(data.dtype)
        ngood = data.shape[0] if mask is None else NDStacker._num_good(mask)
        if variance is None:  # IRAF gemcombine calculation
            out_var = NDStacker.calculate_variance(data, mask, out_data)
        else:
            out_var = np.ma.masked_array(variance, mask=mask).mean(axis=0).data.astype(data.dtype) / ngood
        return out_data, out_mask, out_var

    average = mean  # Formally, these are all averages

    @staticmethod
    @combiner
    def wtmean(data, mask=None, variance=None):
        # Inverse-variance weighted mean
        if variance is None:
            return NDStacker.mean(data, mask, variance)
        mask, out_mask = NDStacker._process_mask(mask)
        out_data = (np.ma.masked_array(data/variance, mask=mask).sum(axis=0).data /
                    np.ma.masked_array(1.0/variance, mask=mask).sum(axis=0).data).astype(data.dtype)
        out_var = 1.0 / np.ma.masked_array(1.0/variance, mask=mask).sum(axis=0).data.astype(data.dtype)
        return out_data, out_mask, out_var

    @staticmethod
    @combiner
    def median(data, mask=None, variance=None):
        # Median
        if mask is None:
            num_img = data.shape[0]
            if num_img % 2:
                med_index = num_img // 2
                index = np.argpartition(data, med_index, axis=0)[med_index]
                out_data = take_along_axis(data, index, axis=0)
                out_var = (None if variance is None else
                           take_along_axis(variance, index, axis=0))
                #out_data = np.expand_dims(take_along_axis(data, index, axis=0), axis=0).mean(axis=0)
            else:
                med_index = num_img // 2 - 1
                indices = np.argpartition(data, [med_index, med_index+1],
                                          axis=0)[med_index:med_index+2]
                out_data = take_along_axis(data, indices, axis=0).mean(axis=0).astype(data.dtype)
                # According to Laplace, the uncertainty on the median is
                # sqrt(2/pi) times greater than that on the mean
                out_var = (None if variance is None else
                           0.5 * np.pi * np.ma.masked_array(variance, mask=mask).mean(axis=0).data.astype(data.dtype) / num_img)
            out_mask = None
        else:
            mask, out_mask = NDStacker._process_mask(mask)
            arg = np.argsort(np.where(mask>0, np.inf, data), axis=0)
            num_img = NDStacker._num_good(mask>0)
            med_index = num_img // 2
            med_indices = np.array([np.where(num_img % 2, med_index, med_index-1),
                                    np.where(num_img % 2, med_index, med_index)])
            indices = take_along_axis(arg, med_indices, axis=0)
            out_data = take_along_axis(data, indices, axis=0).mean(axis=0).astype(data.dtype)
            #out_mask = np.bitwise_or(*take_along_axis(mask, indices, axis=0))
            out_var = (None if variance is None else
                       0.5 * np.pi * np.ma.masked_array(variance, mask=mask).mean(axis=0).data.astype(data.dtype) / num_img)
        if variance is None:  # IRAF gemcombine calculation, plus Laplace
            out_var = 0.5 * np.pi * NDStacker.calculate_variance(data, mask, out_data)
        return out_data, out_mask, out_var

    @staticmethod
    @combiner
    def lmedian(data, mask=None, variance=None):
        # Low median: i.e., if even number, take lower of 2 middle items
        num_img = data.shape[0]
        if mask is None:
            med_index = (num_img - 1) // 2
            index = np.argpartition(data, med_index, axis=0)[med_index]
            out_mask = None
        else:
            mask, out_mask = NDStacker._process_mask(mask)
            # Because I'm sorting, I'll put large dummy values in a numpy array
            # np.choose() can't handle more than 32 input images
            # Partitioning the bottom half is slower than a full sort
            arg = np.argsort(np.where(mask>0, np.inf, data), axis=0)
            num_img = NDStacker._num_good(mask>0)
            med_index = (num_img - 1) // 2
            index = take_along_axis(arg, med_index, axis=0)
        out_data = take_along_axis(data, index, axis=0)
        if variance is None:  # IRAF gemcombine calculation, plus Laplace
            out_var = 0.5 * np.pi * NDStacker.calculate_variance(data, mask, out_data)
        else:
            out_var = 0.5 * np.pi * np.ma.masked_array(variance, mask=mask).mean(axis=0).data.astype(data.dtype) / num_img
        return out_data, out_mask, out_var

    #------------------------ REJECTOR METHODS ----------------------------
    # These methods must all return data, mask, and variance arrays of the
    # same size as the input, but with pixels reflagged if necessary to
    # indicate the results of the rejection. Pixels can be reordered along
    # the axis that is being compressed.

    @staticmethod
    @rejector
    def none(data, mask=None, variance=None):
        # No rejection: That's easy!
        return data, mask, variance

    @staticmethod
    @rejector
    def minmax(data, mask=None, variance=None, nlow=0, nhigh=0):
        # minmax rejection, following IRAF rules when pixels are rejected
        # We flag the pixels to be rejected as DQ.bad_pixel. For any pixels
        # to be flagged this way, there have to be good (or nonlin/saturated)
        # pixels around so they will get combined before the DQhierarchy
        # looks at DQ.bad_pixel
        num_img = data.shape[0]
        if nlow+nhigh >= num_img:
            raise ValueError("Only {} images but nlow={} and nhigh={}"
                             .format(num_img, nlow, nhigh))
        if mask is None:
            nlo = int(nlow+0.001)
            nhi = data.shape[0] - int(nhigh+0.001)
            # Sorts variance with data
            arg = np.argsort(data, axis=0)
            data = take_along_axis(data, arg, axis=0)
            variance = take_along_axis(variance, arg, axis=0)
            mask = np.zeros_like(data, dtype=bool)
            mask[:nlo] = True
            mask[nhi:] = True
        else:
            # Because I'm sorting, I'll put large dummy values in a numpy array
            # Have to keep all values if all values are masked!
            # Sorts variance and mask with data
            arg = np.argsort(np.where(mask & BAD, np.inf, data), axis=0)
            data = take_along_axis(data, arg, axis=0)
            variance = take_along_axis(variance, arg, axis=0)
            mask = take_along_axis(mask, arg, axis=0)
            # IRAF imcombine maths
            num_good = NDStacker._num_good(mask & BAD > 0)
            nlo = (num_good * float(nlow) / num_img + 0.001).astype(int)
            nhi = num_good - (num_good * float(nhigh) / num_img + 0.001).astype(int) - 1
            for i in range(num_img):
                mask[i][i<nlo] |= ONE
                mask[i][np.logical_and(i>nhi, i<num_good)] |= ONE
        return data, mask, variance

    @staticmethod
    @rejector
    def sigclip(data, mask=None, variance=None, mclip=True, lsigma=3.0,
                hsigma=3.0, max_iters=None):
        # Sigma-clipping based on scatter of data
        return NDStacker._cyclip(data, mask=mask, variance=variance,
                                  mclip=mclip, lsigma=lsigma, hsigma=hsigma,
                                  max_iters=max_iters, sigclip=True)

    @staticmethod
    @rejector
    def varclip(data, mask=None, variance=None, mclip=True, lsigma=3.0,
                hsigma=3.0, max_iters=None):
        # Sigma-type-clipping where VAR array is used to determine deviancy
        return NDStacker._cyclip(data, mask=mask, variance=variance,
                                  mclip=mclip, lsigma=lsigma, hsigma=hsigma,
                                  max_iters=max_iters, sigclip=False)

    @staticmethod
    def _cyclip(data, mask=None, variance=None, mclip=True, lsigma=3.0,
                 hsigma=3.0, max_iters=None, sigclip=False):
        # Prepares data for Cython iterative-clippingroutine
        if mask is None:
            mask = np.zeros_like(data, dtype=DQ.datatype)
        if variance is None:
            variance = np.empty((1,), dtype=np.float32)
            has_var = False
        else:
            has_var = True
        if max_iters is None:
            max_iters = 0
        # We send each array as a 1D view, with all the input pixels together,
        # So if we have 10 input images, the first 10 pixels in this 1D array
        # will be the (0,0) pixels from each image, and so on...
        shape = data.shape
        num_img = shape[0]
        data_size = np.multiply.reduce(data.shape[1:])
        data, mask, variance = cyclip.iterclip(data.ravel(), mask.ravel(), variance.ravel(),
                                               has_var=has_var, num_img=num_img, data_size=data_size,
                                               mclip=int(mclip), lsigma=lsigma, hsigma=hsigma,
                                               max_iters=max_iters, sigclip=int(sigclip))
        return data.reshape(shape), mask.reshape(shape), (None if not has_var else variance.reshape(shape))

    @staticmethod
    def _iterclip(data, mask=None, variance=None, mclip=True, lsigma=3.0,
                 hsigma=3.0, max_iters=None, sigclip=False):
        # SUPERSEDED BY CYTHON ROUTINE
        """Mildly generic iterative clipping algorithm, called by both sigclip
        and varclip. We don't use the astropy.stats.sigma_clip() because it
        doesn't check for convergence if max_iters is not None."""
        if max_iters is None:
            max_iters = 100
        high_limit = hsigma
        low_limit = -lsigma
        cenfunc = np.ma.median  # Always median for first pass
        clipped_data = np.ma.masked_array(data, mask=None if mask is None else
                                                     (mask & BAD))
        iter = 0
        ngood = clipped_data.count()
        while iter < max_iters and ngood > 0:
            avg = cenfunc(clipped_data, axis=0)
            if variance is None or sigclip:
                deviation = clipped_data - avg
                sig = np.ma.std(clipped_data, axis=0)
                high_limit = hsigma * sig
                low_limit = -lsigma * sig
            else:
                deviation = (clipped_data - avg) / np.sqrt(variance)
            clipped_data.mask |= deviation > high_limit
            clipped_data.mask |= deviation < low_limit
            new_ngood = clipped_data.count()
            if new_ngood == ngood:
                break
            if not mclip:
                cenfunc = np.ma.mean
            ngood = new_ngood
            iter += 1

        if mask is None:
            mask = clipped_data.mask
        else:
            mask |= clipped_data.mask
        return data, mask, variance
