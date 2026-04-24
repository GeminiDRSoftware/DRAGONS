from __future__ import print_function

from operator import itemgetter
from itertools import groupby

import numpy as np
import scipy.ndimage as ni


# def get_flat_normalization(flat_on_off, bg_std, bpix_mask):

#     lower_limit = bg_std*10

#     flat_norm = bp.estimate_normalization_percentile(flat_on_off,
#                                                      lower_limit, bpix_mask,
#                                                      percentile=99.)
#     # alternative normalization value
#     # norm1 = bp.estimate_normalization(d, lower_limit, bpix_mask)

#     return flat_norm


# def estimate_bg_mean_std(flat, pad=4, smoothing_length=150):

#     flat = flat[pad:-pad, pad:-pad]

#     flat_flat = flat[np.isfinite(flat)].flat
#     flat_sorted = np.sort(flat_flat)

#     flat_gradient = ni.gaussian_filter1d(flat_sorted,
#                                          smoothing_length, order=1)

#     flat_sorted = flat_sorted[smoothing_length:]
#     flat_dist = 1. / flat_gradient[smoothing_length:]

#     over_half_mask = flat_dist > 0.5 * max(flat_dist)
#     max_width_slice = max((sl.stop-sl.start, sl) for sl,
#                           in ni.find_objects(over_half_mask))[1]

#     flat_selected = flat_sorted[max_width_slice]

#     l = len(flat_selected)
#     indm = int(0.5 * l)
#     # ind1, indm, ind2 = map(int, [0.05 * l, 0.5 * l, 0.95 * l])

#     flat_bg = flat_selected[indm]

#     fwhm = flat_selected[-1] - flat_selected[0]

#     return flat_bg, fwhm


# def get_flat_mask_auto(flat_bpix):
#     # now we try to build a reasonable mask
#     # start with a simple thresholded mask

#     bg_mean, bg_fwhm = estimate_bg_mean_std(flat_bpix, pad=4,
#                                             smoothing_length=150)
#     with np.errstate(invalid="ignore"):
#         flat_mask = (flat_bpix > bg_mean + bg_fwhm*3)

#     # remove isolated dots by doing binary erosion
#     m_opening = ni.binary_opening(flat_mask, iterations=2)
#     # try to extend the mask with dilation
#     m_dilation = ni.binary_dilation(m_opening, iterations=5)

#     return m_dilation


# def get_y_derivativemap(flat, flat_bpix, bg_std_norm,
#                         max_sep_order=150, pad=50,
#                         med_filter_size=(7, 7),
#                         flat_mask=None):

#     """
#     flat
#     flat_bpix : bpix'ed flat
#     """

#     # 1d-derivatives along y-axis : 1st attempt
#     # im_deriv = ni.gaussian_filter1d(flat, 1, order=1, axis=0)

#     # 1d-derivatives along y-axis : 2nd attempt. Median filter first.

#     # flat_deriv_bpix = ni.gaussian_filter1d(flat_bpix, 1,
#     #                                        order=1, axis=0)

#     # We also make a median-filtered one. This one will be used to make masks.
#     flat_medianed = ni.median_filter(flat,
#                                      size=med_filter_size)

#     flat_deriv = ni.gaussian_filter1d(flat_medianed, 1,
#                                       order=1, axis=0)

#     # min/max filter

#     flat_max = ni.maximum_filter1d(flat_deriv, size=max_sep_order, axis=0)
#     flat_min = ni.minimum_filter1d(flat_deriv, size=max_sep_order, axis=0)

#     # mask for aperture boundray
#     if pad is None:
#         sl = slice()
#     else:
#         sl = slice(pad, -pad)

#     flat_deriv_masked = np.zeros_like(flat_deriv)
#     flat_deriv_masked[sl, sl] = flat_deriv[sl, sl]

#     if flat_mask is not None:
#         flat_deriv_pos_msk = (flat_deriv_masked > flat_max * 0.5) & flat_mask
#         flat_deriv_neg_msk = (flat_deriv_masked < flat_min * 0.5) & flat_mask
#     else:
#         flat_deriv_pos_msk = (flat_deriv_masked > flat_max * 0.5)
#         flat_deriv_neg_msk = (flat_deriv_masked < flat_min * 0.5)

#     return dict(data=flat_deriv,  # _bpix,
#                 pos_mask=flat_deriv_pos_msk,
#                 neg_mask=flat_deriv_neg_msk,
#                 )


def mask_median_clip(y_ma, median_size=5, clip=1):
    """
    Subtract a median-ed singal from the original.
    Then, return a mask with out-of-sigma values clipped.
    """
    from scipy.stats.mstats import trima
    from scipy.signal import medfilt
    y_filtered = y_ma - medfilt(y_ma, median_size)
    y_trimmed = trima(y_filtered, (-clip, clip))
    return y_trimmed.mask


def find_nearest_object(mmp, im_labeled, slice_map, i, labels_center_column):
    """
    mmp : mask
    im_labeled : label
    i : object to be connected
    labels_center_column : known objects
    """
    thre = 40
    # threshold # of pixels (in y-direction) to detect adjacent object
    steps = [5, 10, 20, 40, 80]

    sl_y, sl_x = slice_map[i]

    # right side
    ss = im_labeled[:, sl_x.stop-3:sl_x.stop].max(axis=1)
    ss_msk = ni.maximum_filter1d(ss == i, thre)

    if sl_x.stop < 2048/2.:
        sl_x0 = sl_x.stop
        sl_x_pos = [sl_x.stop + s for s in steps]
    else:
        sl_x0 = sl_x.start
        sl_x_pos = [sl_x.start - s for s in steps]

    for pos in sl_x_pos:
        ss1 = im_labeled[:, pos]
        detected_ob = set(np.unique(ss1[ss_msk])) - set([0])
        for ob_id in detected_ob:
            if ob_id in labels_center_column:
                sl = slice_map[ob_id][1]
                if sl.start < sl_x0 < sl.stop:
                    continue
                else:
                    return ob_id


def identify_horizontal_line(d_deriv, mmp, pad=20, bg_std=None,
                             thre_dx=30):
    """
    d_deriv : derivative (along-y) image
    mmp : mask
    order : polyfit order
    pad : padding around the boundary will be ignored
    bg_std : if given, derivative smaller than bg_std will be suppressed.
             This will affect faint signal near the chip boundary

    Masks will be derived from mmp, and peak values of d_deriv will be
    fitted with polynomical of given order.

    We first limit the area between
       1024 - thre_dx > x > 1024 + thre_dx

    and identify objects from the mask whose x-slice is larger than thre_dx.
    This will identify at most one object per order. For objects not included
    in this list, we find nearest object and associate it to them.
    """
    # if 0:

    #     pad=50,
    #     bg_std=bg_std_norm
    #     d_deriv=-flat_deriv
    #     mmp=flat_deriv_neg_msk
    #     d_deriv=flat_deriv
    #     mmp=flat_deriv_pos_msk

    ny, nx = d_deriv.shape

    # We first identify objects
    im_labeled, label_max = ni.label(mmp)
    label_indx = np.arange(1, label_max+1, dtype="i")
    objects_found = ni.find_objects(im_labeled)

    # from itertools import compress

    slice_map = dict(zip(label_indx, objects_found))

    # if 0:
    #     # make labeld image with small objects delted.
    #     s = ni.measurements.sum(mmp, labels=im_labeled,
    #                             index=label_indx)
    #     tiny_traces = s < 10. # 0.1 * s.max()
    #     mmp2 = im_labeled.copy()
    #     for label_num in compress(label_indx, tiny_traces):
    #         sl = slice_map[label_num]
    #         mmp_sl = mmp2[sl]
    #         mmp_sl[mmp_sl == label_num] = 0

    # We only traces solutions that are detected in the centeral colmn

    # label numbers along the central column

    # from itertools import groupby
    # labels_center_column = [i for i, _ in groupby(im_labeled[:,nx/2]) if i>0]

    # thre_dx = 30
    center_cut = im_labeled[:, nx//2-thre_dx:nx//2+thre_dx]
    labels_ = list(set(np.unique(center_cut)) - set([0]))

    if True:  # remove flase detections
        sl_subset = [slice_map[l][1] for l in labels_]
        mm = [(sl1.stop - sl1.start) > thre_dx for sl1 in sl_subset]
        labels1 = [l1 for l1, m1 in zip(labels_, mm) if m1]

    # for i in labels_:
    #     if i not in labels1:
    #         center_cut[center_cut == i] = 0

    labels_center_column = sorted(labels1)

    # remove objects with small area
    s = ni.sum(mmp, labels=im_labeled,
               index=labels_center_column)

    labels_center_column = np.array(labels_center_column)[s > 0.1 * s.max()]

    # try to stitch undetected object to center ones.
    undetected_labels_ = [i for i in range(1, label_max+1)
                          if i not in labels_center_column]
    s2 = ni.sum(mmp, labels=im_labeled,
                index=undetected_labels_)

    undetected_labels = np.array(undetected_labels_)[s2 > 0.1 * s.max()]

    slice_map_update_required = False

    for i in undetected_labels:
        ob_id = find_nearest_object(mmp, im_labeled,
                                    slice_map, i, labels_center_column)
        if ob_id:
            im_labeled[im_labeled == i] = ob_id
            slice_map_update_required = True

    if slice_map_update_required:
        objects_found = ni.find_objects(im_labeled)
        slice_map = dict(zip(label_indx, objects_found))

    # im_labeled is now updated

    y_indices = np.arange(ny)
    x_indices = np.arange(nx)

    centroid_list = []
    for indx in labels_center_column:

        sl = slice_map[indx]

        y_indices1 = y_indices[sl[0]]
        x_indices1 = x_indices[sl[1]]

        # mask for line to trace
        feature_msk = im_labeled[sl] == indx

        # nan for outer region.
        feature = d_deriv[sl].copy()
        feature[~feature_msk] = np.nan

        # measure centroid
        yc = np.nansum(y_indices1[:, np.newaxis] * feature, axis=0)
        ys = np.nansum(feature, axis=0)
        yn = np.sum(np.isfinite(feature), axis=0)

        with np.errstate(invalid="ignore"):
            yy = yc/ys

            msk = mask_median_clip(yy) | ~np.isfinite(yy)

            # we also clip wose derivative is smaller than bg_std
            # This suprress the lowest order of K band
            if bg_std is not None:
                msk = msk | (ys/yn < bg_std)
                # msk = msk | (ys/yn < 0.0006 + 0.0003)

            # mask out columns with # of valid pixel is too many
            # number 10 need to be fixed - JJL
            msk = msk | (yn > 10)

        centroid_list.append((x_indices1,
                              np.ma.array(yy, mask=msk)))

    return centroid_list


def trace_aperture_chebyshev(xy_list, domain=None):
    """
    a list of (x_array, y_array).

    y_array must be a masked array
    """
    import numpy.polynomial.chebyshev as cheb

    if domain is None:
        domain = [0, 2047]

    # we first fit the all traces with 2d chebyshev polynomials
    x_list, o_list, y_list = [], [], []
    for o, (x, y) in enumerate(xy_list):
        if hasattr(y, "mask"):
            msk = ~y.mask & np.isfinite(y.data)
            y = y.data
        else:
            msk = np.isfinite(np.array(y, "d"))
        x1 = np.array(x)[msk]
        x_list.append(x1)
        o_list.append(np.zeros(len(x1))+o)
        y_list.append(np.array(y)[msk])

    n_o = len(xy_list)

    from astropy.modeling import fitting  # models, fitting
    from astropy.modeling.polynomial import Chebyshev2D
    x_degree, y_degree = 4, 5
    p_init = Chebyshev2D(x_degree, y_degree,
                         x_domain=domain, y_domain=[0, n_o-1])
    fit_p = fitting.LinearLSQFitter()

    xxx, ooo, yyy = (np.concatenate(x_list),
                     np.concatenate(o_list),
                     np.concatenate(y_list))
    p = fit_p(p_init, xxx, ooo, yyy)

    # if 0:
    #     ax1 = subplot(121)
    #     for o, xy in enumerate(xy_list):
    #         ax1.plot(x_list[o],
    #                  y_list[o] - p(x_list[o], o+np.zeros_like(x_list[o])))

    for ii in range(3):  # number of iteration
        mmm = np.abs(yyy - p(xxx, ooo)) < 1
        # This need to be fixed with actual estimation of sigma.

        p = fit_p(p_init, xxx[mmm], ooo[mmm], yyy[mmm])

    # if 0:
    #     ax2=subplot(122, sharey=ax1)
    #     for o, xy in enumerate(xy_list):
    #         ax2=plot(x_list[o], y_list[o] - p(x_list[o],
    #                                           o+np.zeros_like(x_list[o])))

    # Now we need to derive a 1d chebyshev for each order.  While
    # there should be an analitical way, here we refit the trace for
    # each order using the result of 2d fit.

    xx = np.arange(domain[0], domain[1])
    oo = np.zeros_like(xx)

    ooo = [o[0] for o in o_list]

    def _get_f(o0):
        y_m = p(xx, oo+o0)
        f = cheb.Chebyshev.fit(xx, y_m, x_degree, domain=domain)
        return f

    f_list = []
    f_list = [_get_f(o0) for o0 in ooo]

    # def _get_f_old(next_orders, y_thresh):
    #     oi = next_orders.pop(0)
    #     y_m = p(xx, oo+oi)
    #     f = cheb.Chebyshev.fit(xx, y_m, x_degree, domain=domain)
    #     if next_orders:  # if not the last order
    #         if np.all(y_thresh(y_m)):
    #             print("all negative at ", oi)
    #             next_orders = next_orders[:1]

    #     return oi, f, next_orders

    def _get_f(next_orders, y_thresh):
        oi = next_orders.pop(0)
        y_m = p(xx, oo+oi)
        f = cheb.Chebyshev.fit(xx, y_m, x_degree, domain=domain)
        if np.all(y_thresh(y_m)):
            # print("all negative at ", oi)
            next_orders = []

        return oi, f, next_orders

    # go down in order
    f_list_down = []
    o_list_down = []
    go_down_orders = [ooo[0] - _oi for _oi in range(1, 5)]
    while go_down_orders:
        oi, f, go_down_orders = _get_f(go_down_orders,
                                       y_thresh=lambda y_m: y_m < domain[0])
        f_list_down.append(f)
        o_list_down.append(oi)

    f_list_up = []
    o_list_up = []
    go_up_orders = [ooo[-1]+_oi for _oi in range(1, 5)]
    while go_up_orders:
        oi, f, go_up_orders = _get_f(go_up_orders,
                                     y_thresh=lambda y_m: y_m > domain[-1])
        f_list_up.append(f)
        o_list_up.append(oi)

    # if 0:
    #     _get_f(next_orders)

    #     oi = go_down_orders.pop(0)
    #     y_m = p(xx, oo+ooo[0]-oi)
    #     f = cheb.Chebyshev.fit(xx, y_m, x_degree, domain=domain)
    #     if go_down_orders: # if not the last order
    #         if np.all(y_m < domain[0]):
    #             print("all negative at ", ooo[0]-oi)
    #             go_down_orders = [oi+1]
    #         else:
    #             f_list_down.append(f)
    #     else:
    #         f_list_down.append(f)

    # print(o_list_down[::-1] + ooo + o_list_up)
    return f_list, f_list_down[::-1] + f_list + f_list_up


def get_matched_slices(yc_down_list, yc_up_list):

    mid_indx_down = len(yc_down_list) // 2

    mid_indx_up = np.searchsorted(yc_up_list, yc_down_list[mid_indx_down])

    n_lower = min(mid_indx_down, mid_indx_up)

    n_upper = min(len(yc_down_list) - mid_indx_down,
                  len(yc_up_list) - mid_indx_up)

    slice_down = slice(mid_indx_down - n_lower, mid_indx_down + n_upper)
    slice_up = slice(mid_indx_up - n_lower, mid_indx_up + n_upper)

    return slice_down, slice_up


def trace_centroids_chevyshev(centroid_bottom_list,
                              centroid_up_list,
                              domain, ref_x=None):
    # from .trace_aperture import trace_aperture_chebyshev

    if ref_x is None:
        ref_x = 0.5 * (domain[0] + domain[-1])

    _ = trace_aperture_chebyshev(centroid_bottom_list,
                                 domain=domain)
    sol_bottom_list, sol_bottom_list_full = _

    _ = trace_aperture_chebyshev(centroid_up_list,
                                 domain=domain)
    sol_up_list, sol_up_list_full = _

    yc_down_list = [s(ref_x) for s in sol_bottom_list_full]
    # lower-boundary list
    yc_up_list = [s(ref_x) for s in sol_up_list_full]
    # upper-boundary list

    # yc_down_list[1] should be the 1st down-boundary that is not
    # outside the detector

    indx_down_bottom = np.searchsorted(yc_down_list, yc_up_list[1])
    indx_up_top = np.searchsorted(yc_up_list, yc_down_list[-2],
                                  side="right")

    # indx_up_bottom = np.searchsorted(yc_up_list, yc_down_list[1])
    # indx_down_top = np.searchsorted(yc_down_list, yc_up_list[-2],
    #                                 side="right")

    # print zip(yc_down_list[1:-1], yc_up_list[indx:])
    # print "index", indx_down_bottom, indx_up_top
    # print "down", yc_down_list
    # print "up", yc_up_list
    # print zip(yc_down_list[indx_down_bottom-1:-1],
    #           yc_up_list[1:indx_up_top+1])

    sol_bottom_up_list_full = zip(sol_bottom_list_full[indx_down_bottom-1:-1],
                                  sol_up_list_full[1:indx_up_top+1])

    slice_down, slice_up = get_matched_slices(yc_down_list, yc_up_list)

    sol_bottom_up_list_full = zip(sol_bottom_list_full[slice_down],
                                  sol_up_list_full[slice_up])

    # check if the given order has pixels in the detector
    x = np.arange(2048)
    sol_bottom_up_list_full_filtered = []
    for s_bottom, s_up in sol_bottom_up_list_full:
        if max(s_up(x)) > 0. and min(s_bottom(x)) < 2048.:
            sol_bottom_up_list_full_filtered.append((s_bottom, s_up))

    # print sol_bottom_up_list_full

    sol_bottom_up_list = sol_bottom_list, sol_up_list
    centroid_bottom_up_list = centroid_bottom_list, centroid_up_list
    # centroid_bottom_up_list = []

    return (sol_bottom_up_list_full_filtered,
            sol_bottom_up_list, centroid_bottom_up_list)


# def get_smoothed_order_spec(s):
#     s = np.array(s)
#     k1, k2 = np.nonzero(np.isfinite(s))[0][[0, -1]]
#     s1 = s[k1:k2+1]

#     s0 = np.empty_like(s)
#     s0.fill(np.nan)
#     s0[k1:k2+1] = ni.median_filter(s1, 40)
#     return s0


# def get_order_boundary_indices(s1, s0=None):
#     # x = np.arange(len(s))

#     # select finite number only. This may happen when orders go out of
#     # chip boundary.
#     s1 = np.array(s1)
#     # k1, k2 = np.nonzero(np.isfinite(s1))[0][[0, -1]]

#     with np.errstate(invalid="ignore"):
#         nonzero_indices = np.nonzero(s1 > 0.05)[0]  # [[0, -1]]

#     # return meaningless indices if non-zero spectra is too short
#     with np.errstate(invalid="ignore"):
#         if len(nonzero_indices) < 5:
#             return 4, 4

#     k1, k2 = nonzero_indices[[0, -1]]
#     k1 = max(k1, 4)
#     k2 = min(k2, 2047-4)
#     s = s1[k1:k2+1]

#     if s0 is None:
#         s0 = get_smoothed_order_spec(s)
#     else:
#         s0 = s0[k1:k2+1]

#     mm = s > max(s) * 0.05
#     dd1, dd2 = np.nonzero(mm)[0][[0, -1]]

#     # mask out absorption feature
#     smooth_size = 20
#     # s_s0 = s-s0
#     # s_s0_std = s_s0[np.abs(s_s0) < 2.*s_s0.std()].std()

#     # mmm = s_s0 > -3.*s_s0_std

#     s1 = ni.gaussian_filter1d(s0[dd1:dd2], smooth_size, order=1)
#     # x1 = x[dd1:dd2]

#     # s1r = s1 # ni.median_filter(s1, 100)

#     s1_std = s1.std()
#     s1_std = s1[np.abs(s1) < 2.*s1_std].std()

#     s1[np.abs(s1) < 2.*s1_std] = np.nan

#     indx_center = int(len(s1)*.5)

#     left_half = s1[:indx_center]
#     if np.any(np.isfinite(left_half)):
#         i1 = np.nanargmax(left_half)
#         a_ = np.where(~np.isfinite(left_half[i1:]))[0]
#         if len(a_):
#             i1r = a_[0]
#         else:
#             i1r = 0
#         i1 = dd1+i1+i1r  # +smooth_size
#     else:
#         i1 = dd1

#     right_half = s1[indx_center:]
#     if np.any(np.isfinite(right_half)):
#         i2 = np.nanargmin(right_half)
#         a_ = np.where(~np.isfinite(right_half[:i2]))[0]

#         if len(a_):
#             i2r = a_[-1]
#         else:
#             i2r = i2
#         i2 = dd1+indx_center+i2r
#     else:
#         i2 = dd2

#     return k1+i1, k1+i2


# def get_finite_boundary_indices(s1):
#     # select finite number only. This may happen when orders go out of
#     # chip boundary.
#     s1 = np.array(s1)
#     # k1, k2 = np.nonzero(np.isfinite(s1))[0][[0, -1]]

#     # k1, k2 = np.nonzero(s1>0.)[0][[0, -1]]
#     with np.errstate(invalid="ignore"):
#         nonzero_indices = np.nonzero(s1 > 0.)[0]  # [[0, -1]]

#     # # return meaningless indices if non-zero spectra is too short
#     #  if len(nonzero_indices) < 5:
#     #      return 4, 4

#     k1, k2 = nonzero_indices[[0, -1]]
#     k1 = max(k1, 4)
#     k2 = min(k2, 2047-4)
#     return k1, k2


def get_y_derivativemap(flat,
                        max_sep_order=150, pad=50,
                        med_filter_size=(7, 7),
                        flat_mask=None):

    """
    flat
    """

    # 1d-derivatives along y-axis : 1st attempt
    # im_deriv = ni.gaussian_filter1d(flat, 1, order=1, axis=0)

    # 1d-derivatives along y-axis : 2nd attempt. Median filter first.

    # flat_deriv_bpix = ni.gaussian_filter1d(flat_bpix, 1,
    #                                        order=1, axis=0)

    # We also make a median-filtered one. This one will be used to make masks.
    flat_medianed = ni.median_filter(flat,
                                     size=med_filter_size)

    flat_deriv = ni.gaussian_filter1d(flat_medianed, 1,
                                      order=1, axis=0)

    # min/max filter

    flat_max = ni.maximum_filter1d(flat_deriv, size=max_sep_order, axis=0)
    flat_min = ni.minimum_filter1d(flat_deriv, size=max_sep_order, axis=0)

    # mask for aperture boundray
    if pad is None:
        sl = slice()
    else:
        sl = slice(pad, -pad)

    flat_deriv_masked = np.zeros_like(flat_deriv)
    flat_deriv_masked[sl, sl] = flat_deriv[sl, sl]

    if flat_mask is not None:
        flat_deriv_pos_msk = (flat_deriv_masked > flat_max * 0.5) & flat_mask
        flat_deriv_neg_msk = (flat_deriv_masked < flat_min * 0.5) & flat_mask
    else:
        flat_deriv_pos_msk = (flat_deriv_masked > flat_max * 0.5)
        flat_deriv_neg_msk = (flat_deriv_masked < flat_min * 0.5)

    return dict(data=flat_deriv,  # _bpix,
                pos_mask=flat_deriv_pos_msk,
                neg_mask=flat_deriv_neg_msk,
                )


def _check_boundary_orders(cent_list, nx=2048):

    c_list = []
    for xc, yc in cent_list:
        p = np.polyfit(xc[~yc.mask], yc.data[~yc.mask], 2)
        c_list.append(np.polyval(p, nx/2.))

    indexes = np.argsort(c_list)

    return [cent_list[i] for i in indexes]


def trace_flat_edges(flat):
    """
    For now, we assume that flat has no bad values.
    """
    flat_normed = flat / np.nanmax(flat)
    # if bpmask is None:
    #     flat_bpixed = np.zeros(flat_normed.shape, dtype=bool)
    bg_fwhm_normed = 0.02
    flat_mask = flat_normed > 0.05

    flat_deriv_ = get_y_derivativemap(flat_normed,
                                      max_sep_order=150, pad=10,
                                      flat_mask=flat_mask)

    flat_deriv = flat_deriv_["data"]
    flat_deriv_pos_msk = flat_deriv_["pos_mask"]
    flat_deriv_neg_msk = flat_deriv_["neg_mask"]


    ny, nx = flat_deriv.shape

    cent_bottom_list = identify_horizontal_line(flat_deriv,
                                                flat_deriv_pos_msk,
                                                pad=10,
                                                bg_std=bg_fwhm_normed)

    # make sure that centroid lists are in order by checking its center
    # position.
    cent_bottom_list = _check_boundary_orders(cent_bottom_list, nx=nx)

    cent_up_list = identify_horizontal_line(-flat_deriv,
                                            flat_deriv_neg_msk,
                                            pad=10,
                                            bg_std=bg_fwhm_normed)

    cent_up_list = _check_boundary_orders(cent_up_list, nx=nx)

    nx = 2048

    _ = trace_centroids_chevyshev(cent_bottom_list,
                                  cent_up_list,
                                  domain=[0, nx],
                                  ref_x=nx/2)

    bottom_up_solutions_full, bottom_up_solutions, bottom_up_centroids = _

    assert len(bottom_up_solutions_full) != 0

    from numpy.polynomial import Polynomial

    bottom_up_solutions_as_list = []

    def serialize_cheb(l1):
        return dict(kind="cheb", coef=l1.coef,
                    domain=l1.domain, window=l1.window)

    ll = []
    for i, (bottom, top) in enumerate(bottom_up_solutions_full):

        top_ = serialize_cheb(top)
        top_.update(order=i, edge="top")
        ll.append(top_)

        bottom_ = serialize_cheb(bottom)
        bottom_.update(order=i, edge="bottom")
        ll.append(bottom_)

    return ll


def table_to_poly(tbl):
    from numpy.polynomial.chebyshev import Chebyshev
    poly_dict = dict(cheb=Chebyshev)
    pp = [dict(zip(["order", "edge", "poly"],
                   [row["order"],
                    row["edge"],
                    poly_dict[row["kind"]](row["coef"],
                                           domain=row["domain"],
                                           window=row["window"])]))
          for row in tbl]

    # we groupby order and discard if not both bottom and top edges are
    # defined.
    kk = []
    for o, g in groupby(pp, itemgetter("order")):
        k = dict((g1["edge"], g1["poly"]) for g1 in g)
        if "bottom" not in k or "top" not in k:
            continue
        kk.append((o, k))

    return kk


# for deadpix from flat on

def get_flat_mask_auto(flat_bpix):
    # now we try to build a reasonable mask
    # start with a simple thresholded mask

    bg_mean, bg_fwhm = estimate_bg_mean_std(flat_bpix, pad=4,
                                            smoothing_length=150)
    with np.errstate(invalid="ignore"):
        flat_mask = (flat_bpix > bg_mean + bg_fwhm*3)

    # remove isolated dots by doing binary erosion
    m_opening = ni.binary_opening(flat_mask, iterations=2)
    # try to extend the mask with dilation
    m_dilation = ni.binary_dilation(m_opening, iterations=5)

    return m_dilation


def estimate_bg_mean_std(flat, pad=4, smoothing_length=150):

    flat = flat[pad:-pad, pad:-pad]

    flat_flat = flat[np.isfinite(flat)].flat
    flat_sorted = np.sort(flat_flat)

    flat_gradient = ni.gaussian_filter1d(flat_sorted,
                                         smoothing_length, order=1)

    flat_sorted = flat_sorted[smoothing_length:]
    flat_dist = 1. / flat_gradient[smoothing_length:]

    over_half_mask = flat_dist > 0.5 * max(flat_dist)
    max_width_slice = max((sl.stop-sl.start, sl) for sl,
                          in ni.find_objects(over_half_mask))[1]

    flat_selected = flat_sorted[max_width_slice]

    l = len(flat_selected)
    indm = int(0.5 * l)
    # ind1, indm, ind2 = map(int, [0.05 * l, 0.5 * l, 0.95 * l])

    flat_bg = flat_selected[indm]

    fwhm = flat_selected[-1] - flat_selected[0]

    return flat_bg, fwhm


# def table_to_poly(tbl):
#     from numpy.polynomial.chebyshev import Chebyshev
#     poly_dict = dict(cheb=Chebyshev)
#     pp = [(row["order"],
#            row["edge"],
#            poly_dict[row["kind"]](row["coef"],
#                                   domain=row["domain"],
#                                   window=row["window"]))
#           for row in tbl]
#     return pp


if __name__ == '__main__':
    import astropy.io.fits as pyfits
    hdu = pyfits.open("/home/jjlee/git_personal/IGRINSDR/SDCH_20220301_0011_lampstack.fits")
    ll = trace_flat_edges(hdu["FLAT_ORIGINAL"].data)
    from astropy.table import Table, hstack, vstack, MaskedColumn
    tbl = Table(ll)
