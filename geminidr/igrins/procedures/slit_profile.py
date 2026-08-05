import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import median_filter
from numpy.polynomial import Polynomial
from astropy.stats import biweight_location
from itertools import cycle
# from ..igrins_libs.resource_helper_igrins import ResourceHelper

def _get_norm_profile_ab(bins, hh0):
    peak1, peak2 = np.nanmax(hh0), -np.nanmin(hh0)
    profile_x = 0.5*(bins[1:]+bins[:-1])
    profile_y = hh0/(peak1+peak2)

    return profile_x, profile_y


def _get_norm_profile(bins, hh0):
    peak1 = np.nanmax(hh0)
    profile_x = 0.5*(bins[1:]+bins[:-1])
    profile_y = hh0/peak1

    return profile_x, profile_y


def _get_profile_func_ab(profile_x, profile_y):
    profile_ = UnivariateSpline(profile_x, profile_y, k=3, s=0,
                                bbox=[0, 1])

    roots = list(profile_.roots())
    #assert(len(roots) == 1)
    integ_list = []
    for ss, int_r1, int_r2 in zip(cycle([1, -1]),
                                  [0] + roots,
                                  roots + [1]):
        #print ss, int_r1, int_r2
        integ_list.append(profile_.integral(int_r1, int_r2))

    integ = np.abs(np.sum(integ_list))

    def profile(o, x, slitpos):
        r = profile_(slitpos) / integ
        return r

    return profile


def _get_profile_func(profile_x, profile_y):

    profile_ = UnivariateSpline(profile_x, profile_y, k=3, s=0,
                                bbox=[0, 1])
    integ = profile_.integral(0, 1)

    def profile(o, x, slitpos):
        return profile_(slitpos) / integ

    return profile


def extract_slit_profile(ap, ordermap_bpixed, slitpos_map,
                         data_minus_flattened,
                         x1=800, x2=1200, mode="median"):

    bins, slit_profile_list = \
          ap.extract_slit_profile(ordermap_bpixed,
                                  slitpos_map,
                                  data_minus_flattened,
                                  x1, x2, bins=None)


    if mode == "median":
        s0 = np.array(slit_profile_list)
        ss = np.nansum(np.abs(s0), axis=1)

        hh0 = np.nanmedian(s0/ss[:, np.newaxis], axis=0)
    elif mode == 'biweight_location':
        s0 = np.array(slit_profile_list)
        ss = np.nansum(np.abs(s0), axis=1)

        hh0 = biweight_location(s0/ss[:, np.newaxis], axis=0, ignore_nan=True)
    elif mode == 'mean':
        s0 = np.array(slit_profile_list)
        ss = np.nansum(np.abs(s0), axis=1)

        hh0 = np.nanmean(s0/ss[:, np.newaxis], axis=0)
    else:
        hh0 = np.nansum(slit_profile_list, axis=0)

    return bins, hh0, slit_profile_list


def make_slitprofile_map(ap, profile,
                         ordermap, slitpos_map,
                         frac_slit_list=None, slice_indicies=(0,2048)):

    # helper = ResourceHelper(obsset)

    # ordermap = helper.get("ordermap")
    # slitpos_map = helper.get("slitposmap")

    # profile = obsset.load("SLIT_PROFILE_JSON")

    profile_map = ap.make_profile_map(ordermap,
                                      slitpos_map,
                                      profile, slice_indicies=slice_indicies)

    # select portion of the slit to extract
    if frac_slit_list:
        slitpos_msk = np.zeros(slitpos_map.shape, dtype=bool)
        for frac_slits in frac_slit_list:
            slitpos_msk[(min(frac_slits) < slitpos_map)
                        & (slitpos_map <  max(frac_slits))] = np.nan
        profile_map[~slitpos_msk] = np.nan

    return profile_map

# def make_slitprofile_map(obsset)
#     hdul = obsset.get_hdul_to_write(([], profile_map))
#     obsset.store("slitprofile_fits", hdul, cache_only=True)


def estimate_slit_profile_1d(obsset,
                             x1=800, x2=2048-800,
                             do_ab=True, frac_slit_list=None, method='column'):
    """
    return a profile function

    def profile(order, x_pixel, y_slit_pos):
        return profile_value

    """

    helper = ResourceHelper(obsset)

    orderflat = helper.get("orderflat")

    data_minus = obsset.load_fits_sci_hdu("COMBINED_IMAGE1",
                                          postfix=obsset.basename_postfix).data
    data_minus_flattened = data_minus / orderflat

    # from .aperture_helper import get_aperture_from_obsset
    # orders = helper.get("orders")
    # ap = get_aperture_from_obsset(obsset, orders=orders)
    ap = helper.get_aperture(obsset)

    ordermap = helper.get("ordermap")
    ordermap_bpixed = helper.get("ordermap_bpixed")
    slitpos_map = helper.get("slitposmap")


    
    if method == 'full': #Old method that used a single profile for the full detector
        _ = extract_slit_profile(ap,
                                 ordermap_bpixed, slitpos_map,
                                 data_minus_flattened,
                                 x1=x1, x2=x2,
                                 # mode = 'mean',
                                 # mode = 'median',
                                 mode = 'biweight_location',
                                 )
        bins, hh0, slit_profile_list = _
        if do_ab:
            profile_x, profile_y = _get_norm_profile_ab(bins, hh0)
            # profile = get_profile_func_ab(profile_x, profile_y)
        else:
            profile_x, profile_y = _get_norm_profile(bins, hh0)
            # profile = get_profile_func(profile_x, profile_y)
        slit_profile_dict = dict(orders=ap.orders_to_extract,
                                 ab_mode=do_ab,
                                 slit_profile_list=slit_profile_list,
                                 profile_x=profile_x,
                                 profile_y=profile_y)
        obsset.store("SLIT_PROFILE_JSON", slit_profile_dict,
                     postfix=obsset.basename_postfix)
        profile = _get_profile_func_from_dict(slit_profile_dict)
        profile_map = make_slitprofile_map(ap, profile,
                                           ordermap, slitpos_map,
                                           frac_slit_list=frac_slit_list)
    elif method == 'column': #New method that uses a running median to find the profile per column
        profile_map = np.zeros([2048, 2048])

        for i in range(2048):
            x1 = i - 64 #Range +/- 
            x2 = i + 64
            if x1 < 0: x1 = 0
            if x2 > 2048: x2 = 2048

            bins, hh0, slit_profile_list = extract_slit_profile(ap,
                                     ordermap_bpixed, slitpos_map,
                                     data_minus_flattened,
                                     x1=x1, x2=x2,
                                     mode='median',
                                     #mode = 'biweight_location',
                                     )
            if do_ab:
                profile_x, profile_y = _get_norm_profile_ab(bins, hh0)
            else:
                profile_x, profile_y = _get_norm_profile(bins, hh0)
                # profile = get_profile_func(profile_x, profile_y)
            slit_profile_dict = dict(orders=ap.orders_to_extract,
                                     ab_mode=do_ab,
                                     slit_profile_list=slit_profile_list,
                                     profile_x=profile_x,
                                     profile_y=profile_y)
            profile_map[:,i] = ap.make_profile_column(ordermap, slitpos_map, _get_profile_func_from_dict(slit_profile_dict), slice_index=i)

    hdul = obsset.get_hdul_to_write(([], profile_map))
    obsset.store("slitprofile_fits", hdul, cache_only=True)


def _get_profile_func_from_dict(slit_profile_dict):
    do_ab = slit_profile_dict["ab_mode"]

    profile_x = slit_profile_dict["profile_x"]
    profile_y = slit_profile_dict["profile_y"]

    if do_ab:
        profile = _get_profile_func_ab(profile_x, profile_y)
    else:
        profile = _get_profile_func(profile_x, profile_y)

    return profile


def get_profile_func(obsset):
    slit_profile_dict = obsset.load("SLIT_PROFILE_JSON")
    return _get_profile_func_from_dict(slit_profile_dict)


def get_profile_func_extended(obsset, do_ab):
    if do_ab:
        delta = 0.01
        profile_ = UnivariateSpline([0, 0.5-delta, 0.5+delta, 1],
                                    [1., 1., -1., -1.],
                                    k=1, s=0,
                                    bbox=[0, 1])
    else:
        profile_ = UnivariateSpline([0, 1], [1., 1.],
                                    k=1, s=0,
                                    bbox=[0, 1])

    def profile(o, x, slitpos):
        return profile_(slitpos)

    return profile


def estimate_slit_profile_uniform(obsset,
                                  do_ab=True, frac_slit_list=None):

    helper = ResourceHelper(obsset)

    ap = helper.get("aperture")

    ordermap = helper.get("ordermap")
    slitpos_map = helper.get("slitposmap")

    profile = get_profile_func_extended(obsset, do_ab=do_ab)
    profile_map = make_slitprofile_map(ap, profile,
                                       ordermap, slitpos_map,
                                       frac_slit_list=frac_slit_list)

    hdul = obsset.get_hdul_to_write(([], profile_map))
    obsset.store("slitprofile_fits", hdul, cache_only=True)
