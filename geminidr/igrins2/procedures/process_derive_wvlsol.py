import pandas as pd
import numpy as np


def _convert2wvlsol(p, orders_w_solutions):

    # derive wavelengths.
    xx = np.arange(2048)
    wvl_sol = []
    for o in orders_w_solutions:
        oo = np.empty_like(xx)
        oo.fill(o)
        wvl = p(xx, oo) / o
        wvl_sol.append(list(wvl))

    return wvl_sol


from .ecfit import fit_2dspec

def _fit_2d(xl, yl, zlo, xdeg=4, ydeg=3):
    """
    df: pixels, order, wavelength"""


    x_domain = [0, 2047]
    y_domain = [min(yl)-2, max(yl)+2]
    # x_degree, y_degree = 4, 3
    # x_degree, y_degree = 3, 2

    msk = np.isfinite(xl)

    fit_params = dict(x_degree=xdeg, y_degree=ydeg,
                      x_domain=x_domain, y_domain=y_domain)

    p, m = fit_2dspec(xl[msk], yl[msk], zlo[msk], **fit_params)

    from .astropy_poly_helper import serialize_poly_model
    poly_2d = serialize_poly_model(p)
    fit_results = dict(xyz=[xl[msk], yl[msk], zlo[msk]],
                       fit_params=fit_params,
                       fitted_model=poly_2d,
                       fitted_mask=m)

    return p, fit_results


def fit_wvlsol(df, xdeg=4, ydeg=3):
    """
    df: pixels, order, wavelength"""
    from .ecfit import fit_2dspec

    xl = df["fitted_pixel"].values
    yl = df["order"].values
    zl = df["wavelength"].values
    zlo = zl * yl
    # xl : pixel
    # yl : order
    # zlo : wvl * order

    p, fit_results = _fit_2d(xl, yl, zlo, xdeg=xdeg, ydeg=ydeg)
    return p, fit_results

    # x_domain = [0, 2047]
    # y_domain = [min(yl)-2, max(yl)+2]
    # # x_degree, y_degree = 4, 3
    # # x_degree, y_degree = 3, 2

    # msk = np.isfinite(xl)

    # fit_params = dict(x_degree=xdeg, y_degree=ydeg,
    #                   x_domain=x_domain, y_domain=y_domain)

    # p, m = fit_2dspec(xl[msk], yl[msk], zlo[msk], **fit_params)

    # from .astropy_poly_helper import serialize_poly_model
    # poly_2d = serialize_poly_model(p)
    # fit_results = dict(xyz=[xl[msk], yl[msk], zlo[msk]],
    #                    fit_params=fit_params,
    #                    fitted_model=poly_2d,
    #                    fitted_mask=m)

    # return p, fit_results


def derive_wvlsol(obsset):

    d = obsset.load("SKY_FITTED_PIXELS_JSON")
    df = pd.DataFrame(**d)

    msk = df["slit_center"] == 0.5
    dfm = df[msk]

    p, fit_results = fit_wvlsol(dfm)

    from ..igrins_libs.resource_helper_igrins import ResourceHelper
    helper = ResourceHelper(obsset)
    orders = helper.get("orders")

    wvl_sol = _convert2wvlsol(p, orders)
    d = dict(orders=orders,
             wvl_sol=wvl_sol)

    obsset.store("SKY_WVLSOL_JSON", d)

    fit_results["orders"] = orders
    obsset.store("SKY_WVLSOL_FIT_RESULT_JSON",
                 fit_results)


# if 0:

#     ordermap_fits = caldb.load_resource_for(basename,
#                                             ("sky", "ordermap_fits"))

#     slitposmap_fits = caldb.load_resource_for(basename,
#                                               ("sky", "slitposmap_fits"))

#     # slitoffset_fits = caldb.load_resource_for(basename,
#     #                                           ("sky", "slitoffset_fits"))

#     yy, xx = np.indices(ordermap_fits[0].data.shape)

#     msk = np.isfinite(ordermap_fits[0].data) & (ordermap_fits[0].data > 0)
#     pixels, orders, slit_pos = (xx[msk], ordermap_fits[0].data[msk],
#                                 slitposmap_fits[0].data[msk])

#     # load coeffs
#     # This needs to be fixed
#     names = ["pixel", "order", "slit"]

#     in_df = pd.read_json("coeffs.json", orient="split")
#     in_df = in_df.set_index(names)

#     poly, coeffs = NdPolyNamed.from_pandas(in_df)

#     cc0 = slit_pos - 0.5
#     values = dict(zip(names, [pixels, orders, cc0]))
#     offsets = poly.multiply(values, coeffs) # * cc0

#     offset_map = np.empty(ordermap_fits[0].data.shape, dtype=np.float64)
#     offset_map.fill(np.nan)
#     offset_map[msk] = offsets * cc0 # dd["offsets"]


# def process_band(utdate, recipe_name, band, obsids, config_name):

#     from igrins.libs.recipe_helper import RecipeHelper
#     helper = RecipeHelper(config_name, utdate, recipe_name)

#     derive_wvlsol(helper, band, obsids)


# if __name__ == "__main__":

#     utdate = "20140709"
#     obsids = [62, 63]

#     utdate = "20140525"
#     obsids = [29]

#     utdate = "20150525"
#     obsids = [52]


#     recipe_name = "SKY"


#     # utdate = "20150525"
#     # obsids = [32]

#     # recipe_name = "THAR"

#     band = "K"

#     #helper = RecipeHelper("../recipe.config", utdate)
#     config_name = "../recipe.config"

#     process_band(utdate, recipe_name, band, obsids, config_name)
