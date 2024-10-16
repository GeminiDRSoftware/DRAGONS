# Copyright(c) 2017-2020 Association of Universities for Research in Astronomy, Inc.
#
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


import numpy as np
import inspect
from collections import namedtuple
from functools import wraps
from astrodata import NDAstroData
from .astrotools import divide0
from geminidr.gemini.lookups import DQ_definitions as DQ
from ..utils.decorators import unpack_nddata
try:
    from . import cython_utils
except ImportError:  # pragma: no cover
    raise ImportError("Run 'cythonize -i cython_utils.pyx' in gempy/library")

# A lightweight NDData-like object
NDD = namedtuple("NDD", "data mask variance")

# Some definitions. Non-linear and saturated pixels are not considered to
# be "bad" when rejecting pixels from the input data. If one takes multiple
# images of an object and one of those images is saturated, it would clearly
# be wrong statistically to reject the saturated one.
BAD = 65535 ^ (DQ.non_linear | DQ.saturated)  # NUMPY_2: OK

# A hierarchy of "badness". Pixels in the inputs are considered to be as
# bad as the worst bit set, so a "bad_pixel" will only be used if there are
# no "overlap" or "cosmic_ray" pixels that can be used. Worst pixels are
# listed first.
DQhierarchy = (DQ.no_data, DQ.unilluminated, DQ.bad_pixel, DQ.overlap,
               DQ.cosmic_ray, DQ.non_linear | DQ.saturated)
ZERO = DQ.datatype(0)
ONE = DQ.datatype(DQ.bad_pixel)


def stack_nddata(fn):
    """
    This decorator wraps a method that takes a sequence of NDAstroData
    objects and stacks them into data, mask, and variance arrays of one
    higher dimension, which get passed to the wrapped function.
    It also applies a set of scale factors and/or offsets, if supplied,
    to the raw data before stacking and passing them on.
    The returned arrays are then stuffed back into an NDAstroData object.
    """
    @wraps(fn)
    def wrapper(instance, sequence, scale=None, zero=None, *args, **kwargs):
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
            data[i] = ndd.data * s + z  # NUMPY_2: OK

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

        out_data, out_mask, out_var, rejmap = fn(
            instance, data=data, mask=mask, variance=variance, *args, **kwargs)

        # Can't instantiate NDAstroData with variance
        ret_value = NDAstroData(out_data, mask=out_mask, variance=out_var)
        if rejmap is not None:
            ret_value.meta['other'] = {'REJMAP': NDAstroData(rejmap)}
        return ret_value
    return wrapper


# Decorators to identify methods for combining and rejecting. Note that
# NDStacker.<method>.required_args will return a list of required arguments
def combiner(fn):
    fn.is_combiner = True
    args = list(inspect.signature(fn).parameters)[3:]
    fn.required_args = list(args)
    return fn


def rejector(fn):
    fn.is_rejector = True
    args = list(inspect.signature(fn).parameters)[3:]
    fn.required_args = list(args)
    return fn


def _masked_mean(data, mask=None):
    if mask is None:
        # Creating a masked array with mask=None is extremely slow, use
        # False instead which is the same and much faster.
        mask = False
    data = np.ma.masked_array(data, mask=mask)
    return data.mean(axis=0).data.astype(data.dtype)


def _masked_sum(data, mask=None):
    if mask is None:
        # Creating a masked array with mask=None is extremely slow, use
        # False instead which is the same and much faster.
        mask = False
    data = np.ma.masked_array(data, mask=mask)
    return data.sum(axis=0).data.astype(data.dtype)


def _median_uncertainty(variance, mask, num_img):
    # According to Laplace, the uncertainty on the median is
    # sqrt(2/pi) times greater than that on the mean
    return 0.5 * np.pi * _masked_mean(variance, mask=mask) / num_img


class NDStacker:
    # Base class from which all stacking functions should subclass.
    # Put helper functions here so they can be inherited.
    def __init__(self, combine='mean', reject='none', log=None, **kwargs):
        self._log = log
        try:
            combiner = getattr(self, combine)
            assert getattr(combiner, 'is_combiner')
        except AttributeError:
            self._logmsg("No such combiner as {}. Using mean instead."
                         .format(combine), level='warning')
            combiner = self.mean

        # No combine functions require arguments (yet) but futureproofing
        req_args = getattr(combiner, 'required_args', [])
        self._combiner = combiner
        self._comb_args = {k: v for k, v in kwargs.items() if k in req_args}

        try:
            rejector = getattr(self, reject)
            assert getattr(rejector, 'is_rejector')
        except AttributeError:
            self._logmsg('No such rejector as {}. Using none instead.'
                         .format(reject), level='warning')
            rejector = self.none

        req_args = getattr(rejector, 'required_args', [])
        self._rejector = rejector
        self._rej_args = {k: v for k, v in kwargs.items() if k in req_args}

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
    def _process_mask(mask, unflag_best_pixels=True):
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
            # pixels to 0, and for bad pixels to 65535.
            mask = np.where(np.logical_and(ngood > 0, tmp_mask), DQ.max, mask)
            if unflag_best_pixels:
                mask = np.where(np.logical_and(ngood > 0, ~tmp_mask), ZERO, mask)
            # 32768 in output mask means we have an output pixel
            out_mask[ngood > 0] |= 32768

            # If we've found "good' pixels for all output pixels, leave
            if np.all(out_mask & 32768):
                break

        return mask, out_mask & 32767

    @staticmethod
    def calculate_variance(data, mask, out_data):
        # gemcombine-style estimate of variance about the returned value
        ngood = data.shape[0] if mask is None else NDStacker._num_good(mask)
        return divide0(np.ma.masked_array(np.square(data - out_data),
                       mask=mask).sum(axis=0).data.astype(data.dtype), ngood*(ngood-1))

    @staticmethod
    def _num_good(mask):
        # Return the number of unflagged pixels at each output pixel
        return np.sum(mask == False, axis=0)

    @stack_nddata
    def __call__(self, data, mask=None, variance=None,
                 save_rejection_map=False):
        """
        Perform the rejection and combining. The stack_nddata decorator
        allows a series of NDData object to be sent, and split into data, mask,
        and variance.
        """

        # Convert the debugging pixel to (x,y) coords and bounds check
        if self._debug_pixel is not None:
            try:
                self._debug_pixel = np.unravel_index(self._debug_pixel,
                                                     data.shape[1:])
            except ValueError:
                self._logmsg("Debug pixel out of range")
                self._debug_pixel = None

        if self._debug_pixel is not None:
            self._logmsg("Debug pixel coords {}".format(self._debug_pixel))
            self._pixel_debugger(data, mask, variance, stage='at input')
            info = data[(slice(None),) + self._debug_pixel]
            self._logmsg("stats: mean={:.4f}, median={:.4f}, std={:.4f}"
                         .format(np.mean(info), np.median(info), np.std(info)))
            self._logmsg("-" * 41)
            self._logmsg("Rejection: {} {}".format(self._rejector.__name__,
                                                   self._rej_args))

        # We need to process the mask initially to only keep the "best" pixels
        # around for consideration
        if mask is not None:
            mask, _ = NDStacker._process_mask(mask, unflag_best_pixels=False)
            if self._debug_pixel is not None:
                self._pixel_debugger(data, mask, variance,
                                     stage='after first mask processing')

        data, rejmask, variance = self._rejector(data, mask, variance,
                                                 **self._rej_args)

        if self._debug_pixel is not None:
            self._pixel_debugger(data, rejmask, variance,
                                 stage='immediately after rejection')

        # when mask is None rejector return a bool mask.
        # convert dtype and set mask values to 32768
        rejmap = None
        if rejmask is not None:
            if rejmask.dtype.kind == 'b':
                rejmask = rejmask.astype(DQ.datatype) * 32768

            # Unset the 32768 bit *only* if it's set in all input pixels
            rejmask &= ~(np.bitwise_and.reduce(rejmask, axis=0) & 32768)

            if save_rejection_map:
                # int16 to avoid scaling issue when writing and re-reading
                # with astrodata
                rejmap = np.sum(rejmask > 32767, axis=0, dtype=np.int16)

        if self._debug_pixel is not None:
            self._pixel_debugger(data, rejmask, variance,
                                 stage='after rejection')
            self._logmsg("Combining: {} {}".format(self._combiner.__name__,
                                                   self._comb_args))

        out_data, out_mask, out_var = self._combiner(data, rejmask, variance,
                                                     **self._comb_args)

        if self._debug_pixel is not None:
            self._pixel_debugger_print_line('out', self._debug_pixel, out_data,
                                            out_mask, out_var)
        return out_data, out_mask, out_var, rejmap

    @classmethod
    def combine(cls, data, mask=None, variance=None, rejector="none", combiner="mean", **kwargs):
        """
        Perform the same job as calling an instance of the NDStacker class,
        but without the data unpacking. A convenience method.
        """
        rej_func = getattr(cls, rejector)
        comb_func = getattr(cls, combiner)
        rej_args = {arg: kwargs[arg] for arg in rej_func.required_args
                    if arg in kwargs}
        data, mask, variance = rej_func(data, mask, variance, **rej_args)
        comb_args = {arg: kwargs[arg] for arg in comb_func.required_args
                     if arg in kwargs}
        out_data, out_mask, out_var = comb_func(data, mask, variance, **comb_args)
        return out_data, out_mask, out_var

    def _pixel_debugger_print_line(self, idx, coord, data, mask, variance):
        idx = f'{idx:3d}' if isinstance(idx, int) else idx
        info = [data[coord]]
        info.append('    -' if mask is None else f'{mask[coord]:5d}')
        info.append(' ' * 14 + '-' if variance is None
                    else f'{variance[coord]:15.4f}')
        self._logmsg("{} {:15.4f} {} {}".format(idx, *info))

    def _pixel_debugger(self, data, mask, variance, stage=''):
        self._logmsg("img     data        mask    variance       " + stage)
        for i in range(data.shape[0]):
            coord = (i,) + self._debug_pixel
            self._pixel_debugger_print_line(i, coord, data, mask, variance)
        self._logmsg("-" * 41)

    # ------------------------ COMBINER METHODS ----------------------------
    # These methods must all return data, mask, and varianace arrays of one
    # lower dimension than the input, with the valid (mask==0) input pixels
    # along the axis combined to produce a single output pixel.

    @staticmethod
    @combiner
    @unpack_nddata
    def mean(data, mask=None, variance=None):
        # Regular arithmetic mean
        mask, out_mask = NDStacker._process_mask(mask)
        out_data = _masked_mean(data, mask=mask)
        ngood = data.shape[0] if mask is None else NDStacker._num_good(mask)
        if variance is None:  # IRAF gemcombine calculation
            out_var = NDStacker.calculate_variance(data, mask, out_data)
        else:
            out_var = _masked_mean(variance, mask=mask) / ngood
        return out_data, out_mask, out_var

    average = mean  # Formally, these are all averages

    @staticmethod
    @combiner
    @unpack_nddata
    def wtmean(data, mask=None, variance=None):
        # Inverse-variance weighted mean
        if variance is None:
            return NDStacker.mean(data, mask, variance)
        mask, out_mask = NDStacker._process_mask(mask)
        with np.errstate(all="ignore"):
            out_data = (_masked_sum(data / variance, mask=mask) /
                        _masked_sum(1 / variance, mask=mask))
            out_var = 1 / _masked_sum(1 / variance, mask=mask)
        return out_data, out_mask, out_var

    @staticmethod
    @combiner
    @unpack_nddata
    def median(data, mask=None, variance=None):
        # Median
        out_var = None
        out_mask = None

        if mask is None:
            num_img = data.shape[0]
            if num_img % 2:
                med_index = num_img // 2
                index = np.argpartition(data, med_index, axis=0)[med_index]
                index = np.expand_dims(index, axis=0)
                out_data = np.take_along_axis(data, index, axis=0)[0]
                if variance is not None:
                    out_var = np.take_along_axis(variance, index, axis=0)[0]
            else:
                med_index = num_img // 2 - 1
                indices = np.argpartition(data, [med_index, med_index+1],
                                          axis=0)[med_index:med_index+2]
                out_data = np.take_along_axis(data, indices, axis=0)\
                    .mean(axis=0).astype(data.dtype)
                if variance is not None:
                    out_var = _median_uncertainty(variance, mask, num_img)
        else:
            mask, out_mask = NDStacker._process_mask(mask)
            arg = np.argsort(np.where(mask > 0, np.inf, data), axis=0)
            num_img = NDStacker._num_good(mask > 0)
            med_index = num_img // 2
            med_indices = np.array([np.where(num_img % 2, med_index, med_index-1),
                                    np.where(num_img % 2, med_index, med_index)])
            indices = np.take_along_axis(arg, med_indices, axis=0)
            out_data = np.take_along_axis(data, indices, axis=0)\
                .mean(axis=0).astype(data.dtype)
            # out_mask = np.bitwise_or(*np.take_along_axis(mask, indices, axis=0))
            if variance is not None:
                out_var = _median_uncertainty(variance, mask, num_img)
        if variance is None:  # IRAF gemcombine calculation, plus Laplace
            out_var = 0.5 * np.pi * NDStacker.calculate_variance(data, mask, out_data)
        return out_data, out_mask, out_var

    @staticmethod
    @combiner
    @unpack_nddata
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
            arg = np.argsort(np.where(mask > 0, np.inf, data), axis=0)
            num_img = NDStacker._num_good(mask > 0)
            med_index = np.expand_dims((num_img - 1) // 2, axis=0)
            index = np.take_along_axis(arg, med_index, axis=0)[0]

        index = np.expand_dims(index, axis=0)
        out_data = np.take_along_axis(data, index, axis=0)[0]

        if variance is None:  # IRAF gemcombine calculation, plus Laplace
            out_var = 0.5 * np.pi * NDStacker.calculate_variance(data, mask, out_data)
        else:
            out_var = _median_uncertainty(variance, mask, num_img)

        return out_data, out_mask, out_var

    # ------------------------ REJECTOR METHODS ----------------------------
    # These methods must all return data, mask, and variance arrays of the
    # same size as the input, but with pixels reflagged if necessary to
    # indicate the results of the rejection. Pixels can be reordered along
    # the axis that is being compressed.

    @staticmethod
    @rejector
    @unpack_nddata
    def none(data, mask=None, variance=None):
        # No rejection: That's easy!
        return data, mask, variance

    @staticmethod
    @rejector
    @unpack_nddata
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
            # Sorts data and apply this to the mask
            arg = np.argsort(data, axis=0)
            mask = np.zeros_like(data, dtype=bool)
            np.put_along_axis(mask, arg[:nlo], True, axis=0)
            np.put_along_axis(mask, arg[nhi:], True, axis=0)
        else:
            # Because I'm sorting, I'll put large dummy values in a numpy array
            # Have to keep all values if all values are masked!
            # Sorts variance and mask with data
            arg = np.argsort(np.where(mask == DQ.max, np.inf, data), axis=0)

            # IRAF imcombine maths
            num_good = NDStacker._num_good(mask == DQ.max)
            nlo = (num_good * nlow / num_img + 0.001).astype(int)
            nhi = num_good - (num_good * nhigh / num_img + 0.001).astype(int) - 1

            arg2 = np.argsort(arg, axis=0)
            mask[arg2 < nlo] = DQ.max
            mask[(arg2 > nhi) & (arg2 < num_good)] = DQ.max

        return data, mask, variance

    @staticmethod
    @rejector
    @unpack_nddata
    def sigclip(data, mask=None, variance=None, mclip=True, lsigma=3.0,
                hsigma=3.0, max_iters=None):
        # Sigma-clipping based on scatter of data
        return NDStacker._cyclip(data, mask=mask, variance=variance,
                                 mclip=mclip, lsigma=lsigma, hsigma=hsigma,
                                 max_iters=max_iters, sigclip=True)

    @staticmethod
    @rejector
    @unpack_nddata
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
        if max_iters is None:  # iterate to convergence
            max_iters = 100
        # We send each array as a 1D view, with all the input pixels together,
        # So if we have 10 input images, the first 10 pixels in this 1D array
        # will be the (0,0) pixels from each image, and so on...
        shape = data.shape
        num_img = shape[0]
        # Force int in case arrays are 1D
        data_size = int(np.multiply.reduce(data.shape[1:]))
        data, mask, variance = cython_utils.iterclip(
            data.ravel().astype(np.float32), mask.ravel().astype(DQ.datatype),
            variance.ravel().astype(np.float32),
            has_var=has_var, num_img=num_img, data_size=data_size,
            mclip=int(mclip), lsigma=lsigma, hsigma=hsigma,
            max_iters=max_iters, sigclip=int(sigclip)
        )
        return (data.reshape(shape), mask.reshape(shape),
                None if not has_var else variance.reshape(shape))

    # @staticmethod
    # def _iterclip(data, mask=None, variance=None, mclip=True, lsigma=3.0,
    #               hsigma=3.0, max_iters=None, sigclip=False):
    #     # SUPERSEDED BY CYTHON ROUTINE
    #     """Mildly generic iterative clipping algorithm, called by both sigclip
    #     and varclip. We don't use the astropy.stats.sigma_clip() because it
    #     doesn't check for convergence if max_iters is not None."""
    #     if max_iters is None:
    #         max_iters = 100
    #     high_limit = hsigma
    #     low_limit = -lsigma
    #     cenfunc = np.ma.median  # Always median for first pass
    #     clipped_data = np.ma.masked_array(data, mask=None if mask is None else
    #                                       (mask & BAD))
    #     iter = 0
    #     ngood = clipped_data.count()
    #     while iter < max_iters and ngood > 0:
    #         avg = cenfunc(clipped_data, axis=0)
    #         if variance is None or sigclip:
    #             deviation = clipped_data - avg
    #             sig = np.ma.std(clipped_data, axis=0)
    #             high_limit = hsigma * sig
    #             low_limit = -lsigma * sig
    #         else:
    #             deviation = (clipped_data - avg) / np.sqrt(variance)
    #         clipped_data.mask |= deviation > high_limit
    #         clipped_data.mask |= deviation < low_limit
    #         new_ngood = clipped_data.count()
    #         if new_ngood == ngood:
    #             break
    #         if not mclip:
    #             cenfunc = np.ma.mean
    #         ngood = new_ngood
    #         iter += 1

    #     if mask is None:
    #         mask = clipped_data.mask
    #     else:
    #         mask |= clipped_data.mask
    #     return data, mask, variance


def sum1d(ndd, x1, x2, proportional_variance=True):
    """
    This function sums the pixels between x1 and x2 of a 1-dimensional
    NDData-like object. Fractional pixel locations are defined in the usual
    manner such that each pixel covers the region (i-0.5, i+0.5) where i
    is the integer index.

    Parameters
    ----------
    ndd: NDAstroData
        A 1D NDAstroData object
    x1: float
        start pixel location of region to sum
    x2: float
        end pixel location of region to sum
    proportional_variance: bool
        should fractional pixels contribute to the total variance in linear,
        rather than quadratic, proportion? When summing rows of a spectral
        image, this removes the strong dependency of the output variance on
        the subpixel alignement of the aperture edges.

    Returns
    -------
    NDD:
        sum of pixels (and partial pixels) between x1 and x2, with mask
        and variance
    """
    x1 = max(x1, -0.5)
    x2 = min(x2, ndd.shape[0]-0.5)
    ix1 = int(np.floor(x1 + 0.5))
    ix2 = int(np.ceil(x2 - 0.5))
    fx1 = ix1 - x1 + 0.5
    fx2 = x2 - ix2 + 0.5
    mask = var = None

    try:
        data = fx1*ndd.data[ix1] + ndd.data[ix1+1:ix2].sum() + fx2*ndd.data[ix2]  # NUMPY_2: OK
    except IndexError:  # catches the *entire* aperture being off the image
        return NDD(0, DQ.no_data, 0)
    if ndd.mask is not None:
        mask = np.bitwise_or.reduce(ndd.mask[ix1:ix2+1])
    if ndd.variance is not None:
        # NUMPY_2: OK
        var = ((fx1 if proportional_variance else fx1*fx1)*ndd.variance[ix1] +
               ndd.variance[ix1:ix2].sum() +
               (fx2 if proportional_variance else fx2*fx2)*ndd.variance[ix2])

    return NDD(data, mask, var)
