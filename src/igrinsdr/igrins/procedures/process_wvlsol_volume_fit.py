import numpy as np
import pandas as pd
# import json

from numpy.linalg import lstsq

from .nd_poly import NdPolyNamed


def _get_center(key_list):
    key_list = sorted(key_list)
    n = len(key_list)
    assert divmod(n, 2)[1] == 1
    center_key = key_list[divmod(n, 2)[0]]
    return center_key


def _append_offset(df):
    """
    input should be indexed with multiple values of 'slit_center'.
    Columns of 'pixel0' and 'offsets' will be appended and returned.
    """

    grouped = df.groupby("slit_center")

    slit_center0 = _get_center(grouped.groups.keys())
    rename_dict = {'pixels': 'pixels0'}
    center = grouped.get_group(slit_center0).rename(columns=rename_dict)

    pp = df.join(center["pixels0"])

    pp["offsets"] = pp["pixels"] - pp["pixels0"]
    pp_masked = pp[np.isfinite(pp["offsets"])]

    df_offset = pp_masked.reset_index()

    return df_offset


def _volume_poly_fit(points, scalar, orders, names):

    p = NdPolyNamed(orders, names)  # order 2 for all dimension.

    v = p.get_array(points)
    v = np.array(v)

    # errors are not properly handled for now.
    s = lstsq(v.T, scalar, rcond=None)

    return p, s


def _get_df(obsset):
    d = obsset.load("SKY_FITTED_PIXELS_JSON")
    df = pd.DataFrame(**d)

    index_names = ["kind", "order", "wavelength"]
    df = df.set_index(index_names)[["slit_center", "pixels"]]

    dd = _append_offset(df)
    return dd


def _filter_points(df, drop=0.10):
    ss0 = df.groupby("pixels0")["offsets"]
    ss0_std = ss0.transform(np.std)

    ss = ss0.std()
    vmin = np.percentile(ss, 100*drop)
    vmax = np.percentile(ss, 100*(1 - drop))

    msk = (ss0_std > vmin) & (ss0_std < vmax)

    return df[msk]


def volume_fit(obsset):

    dd = _get_df(obsset)
    dd = _filter_points(dd)

    names = ["pixel", "order", "slit"]
    orders = [3, 2, 1]

    # because the offset at slit center should be 0, we divide the
    # offset by slit_pos, and fit the data then multiply by slit_pos.

    cc0 = dd["slit_center"] - 0.5

    # 3d points : x-pixel, order, location on the slit
    points0 = dict(zip(names, [dd["pixels0"],
                               dd["order"],
                               cc0]))
    # scalar is offset of the measured line from the location at slic center.
    scalar0 = dd["offsets"]

    msk = abs(cc0) > 0.

    points = dict(zip(names, [dd["pixels0"][msk],
                              dd["order"][msk],
                              cc0[msk]]))

    scalar = dd["offsets"][msk] / cc0[msk]

    poly, params = _volume_poly_fit(points, scalar, orders, names)

    if 0:
        #values = dict(zip(names, [pixels, orders, slit_pos]))
        offsets_fitted = poly.multiply(points0, params[0])
        doffsets = scalar0 - offsets_fitted * cc0

        clf()
        scatter(dd["pixels0"], doffsets, c=cc0.values, cmap="gist_heat")

        # clf()
        # scatter(dd["pixels0"] + doffsets, dd["order"] + dd["slit_center"], color="g")
        # scatter(dd["pixels0"], dd["order"] + dd["slit_center"], color="r")


        # # test with fitted data
        # #input_points = np.zeros_like(offsets_fitted)
        # input_points = offsets_fitted
        # poly, params = volume_poly_fit(points,
        #                                input_points,
        #                                orders, names)

        # offsets_fitted = poly.multiply(points, params[0])
        # doffsets = input_points - offsets_fitted

        # clf()
        # scatter(dd["pixels0"], dd["order"] + dd["slit_center"] + doffsets, color="g")
        # scatter(dd["pixels0"], dd["order"] + dd["slit_center"], color="r")

    # save
    out_df = poly.to_pandas(coeffs=params[0])
    out_df = out_df.reset_index()

    d = out_df.to_dict(orient="split")
    obsset.store("VOLUMEFIT_COEFFS_JSON", d)


# from ..libs.recipe_helper import RecipeHelper

# def process_band_make_offset_map(utdate, recipe_name, band,
#                                  obsids, config_name):

#     from igrins.libs.recipe_helper import RecipeHelper
#     helper = RecipeHelper(config_name, utdate, recipe_name)
