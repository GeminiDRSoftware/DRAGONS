from __future__ import print_function

# from collections import namedtuple
import warnings

import numpy as np
import matplotlib  # Affine class is used
# import scipy.ndimage as ni

# from astropy.io.fits import Card

from .. import DESCS
from ..igrins_libs.resource_helper_igrins import ResourceHelper
from ..utils.load_fits import get_first_science_hdu

from .aperture_helper import get_simple_aperture_from_obsset

# from .sky_spec import make_combined_image_sky, extract_spectra
# these are used by recipes

from .trace_flat import (get_smoothed_order_spec,
                         get_order_boundary_indices)

from .smooth_continuum import get_smooth_continuum


def _get_ref_spec_name(recipe_name):

    # if recipe_name is None:
    #     recipe_name = self.recipe_name

    if (recipe_name in ["SKY"]) or recipe_name.endswith("_AB"):
        ref_spec_key = "SKY_REFSPEC_JSON"
        ref_identified_lines_key = "SKY_IDENTIFIED_LINES_V0_JSON"

    elif recipe_name in ["THAR"]:
        ref_spec_key = "THAR_REFSPEC_JSON"
        ref_identified_lines_key = "THAR_IDENTIFIED_LINES_V0_JSON"

    else:
        raise ValueError("Recipe name of '%s' is unsupported."
                         % recipe_name)

    return ref_spec_key, ref_identified_lines_key


def _match_order(src_spectra, ref_spectra):

    orders_ref = ref_spectra["orders"]
    s_list_ref = ref_spectra["specs"]

    s_list_ = src_spectra["specs"]
    s_list = [np.array(s, dtype=np.float64) for s in s_list_]

    # match the orders of s_list_src & s_list_dst
    from .match_orders import match_orders
    delta_indx, orders = match_orders(orders_ref, s_list_ref,
                                      s_list)

    return orders


def identify_orders(obsset):

    ref_spec_key, _ = _get_ref_spec_name(obsset.recipe_name)
    # from igrins.libs.master_calib import load_ref_data
    # ref_spectra = load_ref_data(helper.config, band,

    ref_spec_path, ref_spectra = obsset.rs.load_ref_data(ref_spec_key,
                                                         get_path=True)

    src_spectra = obsset.load(DESCS["ONED_SPEC_JSON"])

    new_orders = _match_order(src_spectra, ref_spectra)

    from ..igrins_libs.logger import info
    info("          orders: {}...{}".format(new_orders[0], new_orders[-1]))

    src_spectra["orders"] = new_orders
    obsset.store(DESCS["ONED_SPEC_JSON"],
                 data=src_spectra)

    aperture_basename = src_spectra["aperture_basename"]
    obsset.store(DESCS["ORDERS_JSON"],
                 data=dict(orders=new_orders,
                           aperture_basename=aperture_basename,
                           ref_spec_path=ref_spec_path))


def _get_offset_transform(thar_spec_src, thar_spec_dst):

    from scipy.signal import correlate
    offsets = []
    cor_list = []
    center = 2048/2.

    for s_src, s_dst in zip(thar_spec_src, thar_spec_dst):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=FutureWarning)

            cor = correlate(s_src, s_dst, mode="same")

        cor_list.append(cor)
        offset = center - np.argmax(cor)
        offsets.append(offset)

    # from skimage.measure import ransac, LineModel
    from .skimage_measure_fit import ransac, LineModel

    xi = np.arange(len(offsets))
    data = np.array([xi, offsets]).T
    model_robust, inliers = ransac(data,
                                   LineModel, min_samples=3,
                                   residual_threshold=2, max_trials=100)

    outliers_indices = xi[inliers == False]
    offsets2 = [o for o in offsets]
    for i in outliers_indices:
        # reduce the search range for correlation peak using the model
        # prediction.
        ym = int(model_robust.predict_y(i))
        x1 = int(max(0, (center - ym) - 20))
        x2 = int(min((center - ym) + 20 + 1, 2048))
        # print i, x1, x2
        ym2 = center - (np.argmax(cor_list[i][x1:x2]) + x1)
        # print ym2
        offsets2[i] = ym2

    def get_offsetter(o):
        def _f(x, o=o):
            return x+o
        return _f
    sol_list = [get_offsetter(offset_) for offset_ in offsets2]

    return dict(sol_type="offset",
                sol_list=sol_list,
                offsets_orig=offsets,
                offsets_revised=offsets2)


def _get_offset_transform_between_2spec(ref_spec, tgt_spec):

    orders_ref = ref_spec["orders"]
    s_list_ref = ref_spec["specs"]

    orders_tgt = tgt_spec["orders"]
    s_list_tgt = tgt_spec["specs"]

    s_list_tgt = [np.array(s) for s in s_list_tgt]

    orders_intersection = set(orders_ref).intersection(orders_tgt)
    orders_intersection = sorted(orders_intersection)

    def filter_order(orders, s_list, orders_intersection):
        s_dict = dict(zip(orders, s_list))
        s_list_filtered = [s_dict[o] for o in orders_intersection]
        return s_list_filtered

    s_list_ref_filtered = filter_order(orders_ref, s_list_ref,
                                       orders_intersection)
    s_list_tgt_filtered = filter_order(orders_tgt, s_list_tgt,
                                       orders_intersection)

    offset_transform = _get_offset_transform(s_list_ref_filtered,
                                             s_list_tgt_filtered)

    return orders_intersection, offset_transform


def identify_lines(obsset):

    _ = _get_ref_spec_name(obsset.recipe_name)
    ref_spec_key, ref_identified_lines_key = _

    ref_spec = obsset.rs.load_ref_data(ref_spec_key)

    tgt_spec = obsset.load(DESCS["ONED_SPEC_JSON"])
    # tgt_spec_path = obsset.query_item_path("ONED_SPEC_JSON")
    # tgt_spec = obsset.load_item("ONED_SPEC_JSON")

    intersected_orders, d = _get_offset_transform_between_2spec(ref_spec,
                                                                tgt_spec)

    # REF_TYPE="OH"
    # fn = "../%s_IGRINS_identified_%s_%s.json" % (REF_TYPE, band,
    #                                             helper.refdate)
    l = obsset.rs.load_ref_data(ref_identified_lines_key)
    # l = json.load(open(fn))
    # ref_spectra = load_ref_data(helper.config, band, kind="SKY_REFSPEC_JSON")

    offsetfunc_map = dict(zip(intersected_orders, d["sol_list"]))

    from .identified_lines import IdentifiedLines

    identified_lines_ref = IdentifiedLines(l)
    ref_map = identified_lines_ref.get_dict()

    identified_lines_tgt = IdentifiedLines(l)
    identified_lines_tgt.update(dict(wvl_list=[], ref_indices_list=[],
                                     pixpos_list=[], orders=[],
                                     groupname=obsset.groupname))

    from .line_identify_simple import match_lines1_pix

    for o, s in zip(tgt_spec["orders"], tgt_spec["specs"]):
        if (o not in ref_map) or (o not in offsetfunc_map):
            wvl, indices, pixpos = [], [], []
        else:
            pixpos, indices, wvl = ref_map[o]
            pixpos = np.array(pixpos)
            msk = (pixpos >= 0)

            ref_pix_list = offsetfunc_map[o](pixpos[msk])
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'Degrees of freedom')
                pix_list, dist = match_lines1_pix(np.array(s), ref_pix_list)

            pix_list[dist > 1] = -1
            pixpos[msk] = pix_list

        identified_lines_tgt.append_order_info(o, wvl, indices, pixpos)

    # REF_TYPE = "OH"
    # fn = "%s_IGRINS_identified_%s_%s.json" % (REF_TYPE, band, helper.utdate)
    # item_path = caldb.query_item_path((band, master_obsid),
    #                                   "IDENTIFIED_LINES")
    # item_path = caldb.query_item_path((band, master_obsid),
    #                                   "IDENTIFIED_LINES")
    obsset.store(DESCS["IDENTIFIED_LINES_JSON"],
                 identified_lines_tgt.data)

# def update_db(obsset):

#     obsset_off = obsset.get_subset("OFF")
#     obsset_on = obsset.get_subset("ON")

#     obsset_off.add_to_db("flat_off")
#     obsset_on.add_to_db("flat_on")


###

# def process_band(utdate, recipe_name, band,
#                  groupname,
#                  obsids, frametypes, aux_infos,
#                  config_name, **kwargs):

#     if recipe_name.upper() != "SKY_AB":
#         if recipe_name.upper().endswith("_AB") and not kwargs.pop("do_ab"):
#             logger.info("ignoring {}:{}".format(recipe_name, groupname))
#             return

#     from .. import get_caldb, get_obsset
#     caldb = get_caldb(config_name, utdate)
#     obsset = get_obsset(caldb, band, recipe_name, obsids, frametypes)

#     # STEP 1 :
#     # make combined image

#     if recipe_name.upper() in ["SKY"]:
#         pass
#     elif recipe_name.upper().endswith("_AB"):
#         pass
#     elif recipe_name.upper() in ["THAR"]:
#         pass
#     else:
#         msg = ("recipe_name {} not supported "
#                "for this recipe").format(recipe_name)
#         raise ValueError(msg)

#     if recipe_name.upper() in ["THAR"]:
#         make_combined_image_thar(obsset)
#     else:
#         make_combined_image_sky(obsset)

#     # Step 2

#     # load simple-aperture (no order info; depends on

#     extract_spectra(obsset)

#     ## aperture trace from Flat)

#     ## extract 1-d spectra from ThAr

#     # Step 3:
#     ## compare to reference ThAr data to figure out orders of each strip
#     ##  -  simple correlation w/ clipping

#     identify_orders(obsset)

#     # Step 4:
#     ##  - For each strip, measure x-displacement from the reference
#     ##    spec. Fit the displacement as a function of orders.
#     ##  - Using the estimated displacement, identify lines from the spectra.
#     identify_lines(obsset)

#     # Step 6:

#     ## load the reference echellogram, and find the transform using
#     ## the identified lines.

#     from .find_affine_transform import find_affine_transform
#     find_affine_transform(obsset)

#     from ..libs.transform_wvlsol import transform_wavelength_solutions
#     transform_wavelength_solutions(obsset)

#     # Step 8:

#     ## make order_map and auxilary files.

#     save_orderflat(obsset)

#     # save figures

#     save_figures(obsset)

#     save_db(obsset)

def find_affine_transform(obsset):

    # As register.db has not been written yet, we cannot use
    # obsset.get("orders")
    orders = obsset.load(DESCS["ORDERS_JSON"])["orders"]

    ap = get_simple_aperture_from_obsset(obsset, orders)

    lines_data = obsset.load(DESCS["IDENTIFIED_LINES_JSON"])

    from .identified_lines import IdentifiedLines
    identified_lines_tgt = IdentifiedLines.load(lines_data)

    xy_list_tgt = identified_lines_tgt.get_xy_list_from_pixlist(ap)

    from .echellogram import Echellogram

    echellogram_data = obsset.rs.load_ref_data(kind="ECHELLOGRAM_JSON")

    echellogram = Echellogram.from_dict(echellogram_data)

    xy_list_ref = identified_lines_tgt.get_xy_list_from_wvllist(echellogram)

    assert len(xy_list_tgt) == len(xy_list_ref)

    from .fit_affine import fit_affine_clip
    affine_tr, mm = fit_affine_clip(np.array(xy_list_ref),
                                    np.array(xy_list_tgt))

    d = dict(xy1f=xy_list_ref, xy2f=xy_list_tgt,
             affine_tr_matrix=affine_tr.get_matrix(),
             affine_tr_mask=mm)

    obsset.store(DESCS["ALIGNING_MATRIX_JSON"],
                 data=d)
# from .find_affine_transform import find_affine_transform


def _get_wavelength_solutions(affine_tr_matrix, zdata,
                              new_orders):
    """
    new_orders : output orders

    convert (x, y) of zdata (where x, y are pixel positions and z
    is wavelength) with affine transform, then derive a new wavelength
    solution.

    """
    from .ecfit import get_ordered_line_data, fit_2dspec  # , check_fit

    affine_tr = matplotlib.transforms.Affine2D()
    affine_tr.set_matrix(affine_tr_matrix)

    d_x_wvl = {}
    for order, z in zdata.items():
        xy_T = affine_tr.transform(np.array([z.x, z.y]).T)
        x_T = xy_T[:, 0]
        d_x_wvl[order] = (x_T, z.wvl)

    _xl, _ol, _wl = get_ordered_line_data(d_x_wvl)
    # _xl : pixel
    # _ol : order
    # _wl : wvl * order

    x_domain = [0, 2047]
    # orders = igrins_orders[band]
    # y_domain = [orders_band[0]-2, orders_band[-1]+2]
    y_domain = [new_orders[0], new_orders[-1]]
    p, m = fit_2dspec(_xl, _ol, _wl, x_degree=4, y_degree=3,
                      x_domain=x_domain, y_domain=y_domain)

    # if 0:
    #     import matplotlib.pyplot as plt
    #     fig = plt.figure(figsize=(12, 7))
    #     orders_band = sorted(zdata.keys())
    #     check_fit(fig, xl, yl, zl, p, orders_band, d_x_wvl)
    #     fig.tight_layout()

    xx = np.arange(2048)
    wvl_sol = []
    for o in new_orders:
        oo = np.empty_like(xx)
        oo.fill(o)
        wvl = p(xx, oo) / o
        wvl_sol.append(list(wvl))

    # if 0:
    #     json.dump(wvl_sol,
    #               open("wvl_sol_phase0_%s_%s.json" % \
    #                    (band, igrins_log.date), "w"))

    return wvl_sol


def transform_wavelength_solutions(obsset):

    # load affine transform

    # As register.db has not been written yet, we cannot use
    # obsset.get("orders")

    helper = ResourceHelper(obsset)
    orders = helper.get("orders")

    d = obsset.load(DESCS["ALIGNING_MATRIX_JSON"])

    affine_tr_matrix = d["affine_tr_matrix"]

    # load echellogram
    echellogram_data = obsset.rs.load_ref_data(kind="ECHELLOGRAM_JSON")

    from .echellogram import Echellogram
    echellogram = Echellogram.from_dict(echellogram_data)

    wvl_sol = _get_wavelength_solutions(affine_tr_matrix,
                                        echellogram.zdata,
                                        orders)

    obsset.store(DESCS["WVLSOL_V0_JSON"],
                 data=dict(orders=orders, wvl_sol=wvl_sol))

    return wvl_sol


def _make_order_flat(flat_normed, flat_mask, orders,
                     order_map, mode="median", extract_mask=None):

    f_reduce = dict(median=np.nanmedian, mean=np.nanmean)[mode]

    import scipy.ndimage as ni
    slices = ni.find_objects(order_map)

    # ordermap for extraction with optional mask applied
    if extract_mask is not None:
        order_map2 = np.ma.array(order_map, mask=~extract_mask).filled(0)
    else:
        order_map2 = order_map

    mean_order_specs = []
    mask_list = []
    for o in orders:
        sl = (slices[o-1][0], slice(0, 2048))

        mmm = order_map2[sl] == o

        d_sl = flat_normed[sl].copy()
        d_sl[~mmm] = np.nan

        f_sl = flat_mask[sl].copy()
        f_sl[~mmm] = np.nan
        ff = np.nanmean(f_sl, axis=0)
        mask_list.append(ff)

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')

            ss = f_reduce(d_sl, axis=0)

        mean_order_specs.append(ss)

    s_list = [get_smoothed_order_spec(s) for s in mean_order_specs]
    i1i2_list = [get_order_boundary_indices(s, s0)
                 for s, s0 in zip(mean_order_specs, s_list)]
    s2_list = [get_smooth_continuum(s) for s, (i1, i2)
               in zip(s_list, i1i2_list)]

    # make flat
    # x = np.arange(len(s_list[-1]))
    flat_im = np.ones(flat_normed.shape, "d")
    # flat_im.fill(np.nan)

    fitted_responses = []

    for o, px in zip(orders, s2_list):
        sl = (slices[o-1][0], slice(0, 2048))
        d_sl = flat_normed[sl].copy()
        msk = (order_map[sl] == o)
        # d_sl[~msk] = np.nan

        d_div = d_sl / px
        px2d = px * np.ones_like(d_div)  # better way to broadcast px?
        with np.errstate(invalid="ignore"):
            d_div[px2d < 0.05*px.max()] = 1.

        flat_im[sl][msk] = (d_sl / px)[msk]
        fitted_responses.append(px)

    with np.errstate(invalid="ignore"):
        flat_im[flat_im < 0.5] = np.nan

    order_flat_dict = dict(orders=orders,
                           fitted_responses=fitted_responses,
                           i1i2_list=i1i2_list,
                           mean_order_specs=mean_order_specs)

    return flat_im, order_flat_dict


def _make_order_flat_deprecated(flat_normed, flat_mask, orders, order_map):

    # from storage_descriptions import (FLAT_NORMED_DESC,
    #                                   FLAT_MASK_DESC)

    # flat_normed  = flaton_products[FLAT_NORMED_DESC][0].data
    # flat_mask = flaton_products[FLAT_MASK_DESC].data

    import scipy.ndimage as ni
    slices = ni.find_objects(order_map)

    mean_order_specs = []
    mask_list = []
    for o in orders:
        # if slices[o-1] is None:
        #     continue
        sl = (slices[o-1][0], slice(0, 2048))
        d_sl = flat_normed[sl].copy()
        d_sl[order_map[sl] != o] = np.nan

        f_sl = flat_mask[sl].copy()
        f_sl[order_map[sl] != o] = np.nan
        ff = np.nanmean(f_sl, axis=0)
        mask_list.append(ff)

        mmm = order_map[sl] == o

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Mean of empty slice')

            ss = [np.nanmean(d_sl[2:-2][:, i][mmm[:, i][2: -2]])
                  for i in range(2048)]

        mean_order_specs.append(ss)

    s_list = [get_smoothed_order_spec(s) for s in mean_order_specs]
    i1i2_list = [get_order_boundary_indices(s, s0)
                 for s, s0 in zip(mean_order_specs, s_list)]
    # p_list = [get_order_flat1d(s, i1, i2) for s, (i1, i2) \
    #          in zip(s_list, i1i2_list)]
    from .smooth_continuum import get_smooth_continuum
    s2_list = [get_smooth_continuum(s) for s, (i1, i2)
               in zip(s_list, i1i2_list)]

    # make flat
    # x = np.arange(len(s_list[-1]))
    flat_im = np.ones(flat_normed.shape, "d")
    # flat_im.fill(np.nan)

    fitted_responses = []

    for o, px in zip(orders, s2_list):
        sl = (slices[o-1][0], slice(0, 2048))
        d_sl = flat_normed[sl].copy()
        msk = (order_map[sl] == o)
        # d_sl[~msk] = np.nan

        d_div = d_sl / px
        px2d = px * np.ones_like(d_div)  # better way to broadcast px?
        with np.errstate(invalid="ignore"):
            d_div[px2d < 0.05*px.max()] = 1.

        flat_im[sl][msk] = (d_sl / px)[msk]
        fitted_responses.append(px)

    with np.errstate(invalid="ignore"):
        flat_im[flat_im < 0.5] = np.nan

    # from storage_descriptions import (ORDER_FLAT_IM_DESC,
    #                                   ORDER_FLAT_JSON_DESC)

    # r = PipelineProducts("order flat")
    # r.add(ORDER_FLAT_IM_DESC, PipelineImageBase([], flat_im))
    # r.add(ORDER_FLAT_JSON_DESC,
    #       PipelineDict(orders=orders,
    #                    fitted_responses=fitted_responses,
    #                    i1i2_list=i1i2_list,
    #                    mean_order_specs=mean_order_specs))

    order_flat_dict = dict(orders=orders,
                           fitted_responses=fitted_responses,
                           i1i2_list=i1i2_list,
                           mean_order_specs=mean_order_specs)

    return flat_im, order_flat_dict


def save_orderflat(obsset):

    helper = ResourceHelper(obsset)
    orders = helper.get("orders")

    from .aperture_helper import get_simple_aperture_from_obsset

    ap = get_simple_aperture_from_obsset(obsset, orders=orders)

    order_map = ap.make_order_map()
    extract_mask = ap.make_order_map(0.2, 0.8) > 0

    hdul = obsset.load_resource_for("flat_normed")
    flat_normed = get_first_science_hdu(hdul).data

    flat_mask = obsset.load_resource_for("flat_mask")

    # from ..libs.process_flat import make_order_flat
    order_flat_im, order_flat_json = _make_order_flat(flat_normed,
                                                      flat_mask,
                                                      orders, order_map,
                                                      extract_mask=extract_mask)

    hdul = obsset.get_hdul_to_write(([], order_flat_im))
    obsset.store(DESCS["order_flat_im"], hdul)

    obsset.store(DESCS["order_flat_json"], order_flat_json)

    order_map2 = ap.make_order_map(mask_top_bottom=True)
    bias_mask = flat_mask & (order_map2 > 0)

    obsset.store(DESCS["bias_mask"], bias_mask)


def update_db(obsset):

    # save db
    obsset.add_to_db("register")


# from ..pipeline.steps import Step


# steps = [Step("Make Combined Sky", make_combined_image_sky),
#          Step("Extract Simple 1d Spectra", extract_spectra),
#          Step("Identify Orders", identify_orders),
#          Step("Identify Lines", identify_lines),
#          Step("Find Affine Transform", find_affine_transform),
#          Step("Derive transformed Wvl. Solution", transform_wavelength_solutions),
#          Step("Save Order-Flats, etc", save_orderflat),
#          Step("Update DB", update_db),
# ]


# if __name__ == "__main__":
#     pass
