from collections import OrderedDict

import numpy as np
import scipy.ndimage as ni
from scipy.interpolate import LSQUnivariateSpline

from . import destripe_helper as dh


def _get_slices(ny, dy):
    n_dy = ny//dy
    dy_slices = [slice(iy*dy, (iy+1)*dy) for iy in range(n_dy)]
    return dy_slices


def get_ref(d):
    """
    Ext
    """
    dm = ni.median_filter(d, [1, 64])
    dd = d - dm

    ddm = ni.median_filter(dd, [64, 1])
    ddd = dd - ddm[:]

    return ddd


def get_individual_bg64(d, median_col=None):
    ny_slices = dh._get_ny_slice(2048, 64)
    rr = []
    for sl in ny_slices[:]:
        # sl = ny_slices[-3]
        r = np.median(d[sl][2:-2,:], axis=0)
        if median_col is not None:
            r = ni.median_filter(r, median_col)
        # plot(r)
        rr.append(r)

    return np.vstack(rr)


def sub_individual_bg64(d, median_col=None):
    g = get_individual_bg64(d, median_col=median_col)
    vv = np.vstack(np.swapaxes(np.broadcast_to(g,
                                               (64,) + g.shape), 0, 1))

    return d - vv


def get_guard_column(d):
    k = dh.get_median_guard_column(d)
    k0 = dh.subtract_row64_median(k)
    ds, p = dh.get_pattern64_from_each_column(k0)

    return p


def sub_guard_column(d, mask=None):
    p = get_guard_column(d)
    vv = d - p[:, np.newaxis]
    # xx = np.median(vv[-512:], axis=0)
    return vv  # - xx


# x = np.arange(2048)
# yy = []
# for r in p64:
#     p = np.polyfit(x[4:-4], r[4:-4], 1)
#     y0 = np.polyval([p[0], 0], x)
#     yy.append(y0)

def get_row64_median(d, mask=None, q=None):
    if q is None:
        f = np.nanmedian
    else:
        def f(d2):
            return np.nanpercentile(d2, q)

    d3 = np.ma.array(d, mask=mask).filled(np.nan)

    ny_slices = _get_slices(2048, 64)
    return np.concatenate([np.broadcast_to(f(d3[sl]), (len(d3[sl]), ))
                           for sl in ny_slices])


class PatternAmpWiseBiasClass(object):
    def __init__(self, q=None):
        self._q = q

        if q is None:
            f = np.nanmedian
        else:
            def f(d2):
                return np.nanpercentile(d2, q)

        self._f = f
        self.slice_size = 64
        self.ny_slices = _get_slices(2048, self.slice_size)

    def get(self, d, mask=None):
        d3 = np.ma.array(d, mask=mask).filled(np.nan)

        s = np.array([self._f(d3[sl]) for sl in self.ny_slices])

        return s - np.median(s)

    def broadcast(self, d, s):
        # return np.broadcast_to(s, (self.slice_size, ))
        kk = [np.broadcast_to(s1, (self.slice_size,)) for s1 in s]
        k0 = np.concatenate(kk)
        return k0[:, np.newaxis]

    def sub(self, d, mask=None):
        s = self.get(d, mask)
        p = self.broadcast(d, s)

        return d - p

# def factory_get_amp_bias(q=None):

#     def _broadcast(s):
#         return np.broadcast_to(s, (slice_size, ))

#     def _get_amp_bias(d2, mask=None):
#         d3 = np.ma.array(d2, mask=mask).filled(np.nan)

#         s = np.array([f(d3[sl]) for sl in ny_slices])

#         return s - np.median(s)

#     return _get_amp_bias, _broadcast

    # return np.concatenate([np.broadcast_to(f(d3[sl]), (len(d3[sl]), ))
    #                        for sl in ny_slices])

    # def _get_amp_bias(d2, mask=None):
    #     if q is None:
    #         k = np.median(d2, axis=1)
    #     else:
    #         k = np.percentile(d2, q, axis=1)
    #     k0 = dh.get_row64_median(k)

    #     return k0

    # return _get_amp_bias


# def factory_sub_amp_bias(q=None):
#     _get_amp_bias, _broadcast = factory_get_amp_bias(q=None)

#     def _sub_amp_bias(d2, mask=None):
#         s = _get_amp_bias(d2, mask)
#         k0 = np.concatenate([_broadcast(s1) for s1 in s])
#         return d2 - k0[:, np.newaxis]

#     return _sub_amp_bias


# def factory_get_amp_bias_old(q=None):
#     def _get_amp_bias(d2, mask=None):
#         if q is None:
#             k = np.median(d2, axis=1)
#         else:
#             k = np.percentile(d2, q, axis=1)
#         k0 = dh.get_row64_median(k)

#         return k0

#     return _get_amp_bias


# def factory_sub_amp_bias_old(q=None):
#     _get_amp_bias = factory_get_amp_bias(q)

#     def _sub_amp_bias(d2, mask=None):
#         k0 = _get_amp_bias(d2, mask)
#         return d2 - k0[:, np.newaxis]

#     return _sub_amp_bias


# def get_col_median_slow_old(d5, mask=None):
#     k = get_col_median(d5, mask=mask)
#     # k = np.ma.median(np.ma.array(d5, mask=mask), axis=0)
#     k = ni.median_filter(np.ma.array(k, mask=~np.isfinite(k)).filled(0), 64)
#     # k = ni.median_filter(k, 64)
#     return k - np.nanmedian(k)

# def sub_col_median_slow_old(d5, mask=None):
#     k = get_col_median_slow(d5, mask=mask)

#     return d5 - k


class PatternBase(object):
    @classmethod
    def get(kls, d5, mask=None):
        pass

    @classmethod
    def broadcast(kls, d5, k):
        pass

    @classmethod
    def sub(kls, d5, mask=None):
        a = kls.get(d5, mask=mask)
        k = kls.broadcast(d5, a)

        return d5 - k

from .destriper import get_stack_subrows

def get_horizontal_stack(d, dy, mask=None, alt_sign=False):
    if mask is None:
        mask = ~np.isfinite(d)
    else:
        mask = ~np.isfinite(d) | mask

    dd, msk = get_stack_subrows(d, dy, mask=mask, alt_sign=alt_sign)
    h = np.hstack(dd)
    p64 = np.nanmedian(h, axis=1)

    return p64

class PatternP64GlobalMedian(PatternBase):
    @classmethod
    def get(cls, d, mask=None):
        p64 = get_horizontal_stack(d, 64, mask=mask, alt_sign=True)
        return p64

    @classmethod
    def broadcast(cls, d1, p64):
        k = dh.concat(p64, [1, -1], 16)
        return k[:, np.newaxis]


class PatternP64Zeroth(PatternBase):
    @classmethod
    def get(kls, d1, mask=None):
        p64a = dh.stack64(d1, mask)
        p64a0 = np.nanmedian(p64a, axis=1)

        return p64a0 - np.nanmedian(p64a0)

    @classmethod
    def broadcast(kls, d1, p64a0):
        k = dh.concat(p64a0, [1, -1], 16)
        
        return k[:, np.newaxis]


class PatternP64First(PatternBase):

    @classmethod
    def get(kls, d0, mask=None):
        p64 = dh.stack64(d0, mask=mask)
        # p64 = p64_ - np.median(p64_, axis=1)[:, np.newaxis]

        # extract the slope component
        a0 = np.nanmedian(p64[:, :1024], axis=1)
        a1 = np.nanmedian(p64[:, 1024:], axis=1)
        v = (a1 + a0)/2.
        x = np.arange(2048)
        u = (a1 - a0)[:, np.newaxis]/1024.*(x-1024.)

        assert np.all(np.isfinite(v))
        assert np.all(np.isfinite(u))

        return v, u

    @classmethod
    def broadcast(kls, d0, v_u):
        v, u = v_u
        p = dh.concat(v[:, np.newaxis] + u, [1, -1], 16)

        assert np.all(np.isfinite(p))

        return p


class PatternP64ColWise(PatternBase):

    @classmethod
    def get(kls, d1, mask=None):
        p64a = dh.stack64(d1, mask)

        return p64a

    @classmethod
    def broadcast(kls, d1, p64a):
        # p64a0 = get_p64_pattern_each(d1, mask)
        k = dh.concat(p64a, [1, -1], 16)

        return k


class PatternColWiseBias(PatternBase):
    @classmethod
    def get(kls, d5, mask=None):
        if mask is not None:
            d6 = np.ma.array(d5, mask=mask).filled(np.nan)
        else:
            d6 = d5

        k = np.nanmedian(d6, axis=0)
        # k = ni.median_filter(k, 64)
        return k - np.nanmedian(k)

    @classmethod
    def broadcast(kls, d5, k):
        return k


class PatternColWiseBiasC64(PatternBase):
    @classmethod
    def get(kls, d5, mask=None):
        k = PatternColWiseBias.get(d5, mask=mask)
        # k = np.ma.median(np.ma.array(d5, mask=mask), axis=0)
        k = np.ma.array(k, mask=~np.isfinite(k)).filled(0)
        k = ni.median_filter(k, 64)
        # k = np.nan_to_num(k)
        ix = np.arange(len(k))
        tsize = 64
        t = np.arange(tsize, len(k), tsize)
        spl = LSQUnivariateSpline(ix, k, t)

        knots, coeffs, k = spl._eval_args
        return knots, coeffs, k

    @classmethod
    def broadcast(kls, d5, tck):
        spl = LSQUnivariateSpline._from_tck(tck)
        s = spl(np.arange(len(d5)))
        return s

# def sub_col_median_slow(d5, mask=None):
#     tck = get_col_median_slow(d5, mask=mask)
#     s = broadcast_col_median_slow(d5, tck)
#     # spl = LSQUnivariateSpline._from_tck(tck)
#     # s = spl(np.arange(len(d5)))

#     return d5 - s


class PatternRowWiseBias(PatternBase):
    @classmethod
    def get(kls, d6, mask=None):
        if mask is not None:
            d6 = np.ma.array(d6, mask=mask).filled(np.nan)

        c = np.nanmedian(d6, axis=1)
        return c

    @classmethod
    def broadcast(kls, d6, c):
        return c[:, np.newaxis]


class PatternAmpP2(PatternBase):
    @classmethod
    def get(kls, d, mask=None):
        """
        returns a tuple of two 32 element array. First is per-amp bias values.
        The second is the [1,-1] amplitude for each amp.
        """
        d = np.ma.array(d, mask=mask).filled(np.nan)

        do = d.reshape(32, 32, 2, -1)
        av = np.nanmedian(do, axis=[1, 3])

        amp_bias_mean = np.mean(av, axis=1)
        amp_bias_amp = av[:, 0] - amp_bias_mean

        return amp_bias_mean, amp_bias_amp

    @classmethod
    def broadcast(kls, d, av_p):
        av, p = av_p
        k = p[:, np.newaxis] * np.array([1, -1])
        v = np.zeros((32, 32, 2, 1)) + k[:, np.newaxis, :, np.newaxis]
        avv = av.reshape(32, 1, 1, 1) + v
        return avv.reshape(2048, 1)


class PatternAmpP2v1(PatternBase):
    @classmethod
    def get(kls, d, mask=None):
        """
        returns a tuple of two 32 element array. First is per-amp bias values.
        The second is the [1,-1] amplitude for each amp.
        """
        d = np.ma.array(d, mask=mask).filled(np.nan)

        # TODO: This seems inefficient. See if there is a better way.
        # we first estimate (32x) bias
        do = d.reshape(32, 32, 2, -1)
        av = np.nanmedian(do, axis=[1, 2, 3])
        dd = do - av.reshape(32, 1, 1, 1)

        # and then [1, -1] bias.
        pp = np.array([1, -1])[np.newaxis, np.newaxis, :, np.newaxis]
        p = np.nanmedian(dd * pp, axis=[1, 2, 3])
        return av, p

    @classmethod
    def broadcast(kls, d, av_p):
        av, p = av_p
        k = p[:, np.newaxis] * np.array([1, -1])
        v = np.zeros((32, 32, 2, 1)) + k[:, np.newaxis, :, np.newaxis]
        avv = av.reshape(32, 1, 1, 1) + v
        return avv.reshape(2048, 1)

    # @classmethod
    # def sub(kls, d, mask=None):
    #     av_p = kls.get(d, mask)
    #     return d - kls.broadcast(d, av_p)


class PatternAmpWiseBiasC64(PatternBase):
    @classmethod
    def get(kls, d6, mask=None):
        g = get_individual_bg64(d6, median_col=64)
        return g

    @classmethod
    def broadcast(kls, d6, g):
        vv = np.vstack(np.swapaxes(np.broadcast_to(g,
                                                   (64,) + g.shape), 0, 1))
        return vv

# def get_amp_bias_variation(d7, mask=None):
#     g = get_individual_bg64(d7, median_col=64)

#     return g

# def sub_amp_bias_variation(d7, mask=None):
#     return sub_individual_bg64(d7, 64)


def _get_std(ds, f_drop):
    # s1, s2 = np.percentile(ds, 5), np.percentile(ds, 95)
    s1 = np.percentile(ds, f_drop*100)
    s2 = np.percentile(ds, 100 - f_drop*100)
    msk = (s1 < ds) & (ds < s2)
    # return np.ma.array(ds, mask=~msk).filled(np.nan)
    return ds[msk].std()


def get_amp_std(d, f_drop=0.01):
    ny_slices = _get_slices(2048, 64)
    # return [np.nanpercentile(d[sl], 10) for sl in ny_slices]
    return [_get_std(d[sl], f_drop) for sl in ny_slices]


def _apply(d, flist, mask=None, draw_hist_ax=None):
    for f in flist:
        d = f(d, mask=mask)
    return d


def apply(d, p_list, mask=None, draw_hist_ax=None):
    for p in p_list:
        d = p.sub(d, mask=mask)
    return d


pipes = OrderedDict(amp_wise_bias_r2=PatternAmpP2,
                    p64_global_median=PatternP64GlobalMedian,
                    p64_0th_order=PatternP64Zeroth,
                    col_wise_bias_c64=PatternColWiseBiasC64,
                    p64_1st_order=PatternP64First,
                    col_wise_bias=PatternColWiseBias,
                    p64_per_column=PatternP64ColWise,
                    row_wise_bias=PatternRowWiseBias,
                    amp_wise_bias_c64=PatternAmpWiseBiasC64)
