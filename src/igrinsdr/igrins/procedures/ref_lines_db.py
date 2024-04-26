import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from ..utils.list_utils import flatten


def get_group_flag_generator():
    import itertools
    p = itertools.chain([1], itertools.repeat(0))
    return p


def get_group_flags(l):
    return [f for r1 in l for r_, f in zip(r1, get_group_flag_generator())]


def get_ref_wavelengths(ohlines, line_indices):

    ref_wvl = [ohlines.um[l] for l in line_indices]

    return ref_wvl


def get_ref_pixels(ref_wvl, wvlsol0, x=None):
    """
    Given the list of wavelengths tuples, return expected pixel
    positions from the initial wavelength solution of wvlsol0.

    """

    if x is None:
        x = np.arange(len(wvlsol0))
    um2pixel = interp1d(wvlsol0, x, bounds_error=False)

    ref_pixel = [um2pixel(w) for w in ref_wvl]

    # there could be cases when the ref lines fall out of bounds,
    # resulting nans.
    nan_filter = [np.all(np.isfinite(p)) for p in ref_pixel]
    valid_list = [[np.all(np.isfinite(p))]*len(p) for p in ref_pixel]

    group_flags = get_group_flags(ref_wvl)
    df = pd.DataFrame(dict(wavelength=flatten(ref_wvl),
                           valid=flatten(valid_list),
                           group_flag=group_flags,
                           group_id=np.add.accumulate(group_flags)))

    ref_pixel_filtered = [r for r, m in zip(ref_pixel, nan_filter) if m]
    df2 = df.join(pd.DataFrame(dict(pixel=flatten(ref_pixel_filtered)),
                               index=df.index[flatten(valid_list)]))

    return df2


def ref_lines_update_pixel(ref_lines, res,
                           colname_i="pixel", colname_f="pixel"):

    _p = ref_lines.groupby(["group_id"])[colname_i]
    x_list0 = [params[0] for params in res]

    # update pixel of ref_lines
    d_shift_list = x_list0 - _p.first()

    pixel_f = [row.values+shift for (v, row), shift
               in zip(_p, d_shift_list)]
    ref_lines[colname_f] = flatten(pixel_f)

    return ref_lines


def ref_lines_update_sigma_pixel(ref_lines, res,
                                 colname_i="sigma_pixel",
                                 colname_f="sigma_pixel"):

    _p = ref_lines.groupby(["group_id"])[colname_i]
    sigma_list = [params[1] for params in res]

    sigma_pixel_f = [[sig] * len(row.values) for (v, row), sig
                     in zip(_p, sigma_list)]

    ref_lines[colname_f] = flatten(sigma_pixel_f)

    return ref_lines


def ref_lines_nested(ref_lines, colname):
    return [row[colname].values for group_id, row
            in ref_lines.groupby(["group_id"])]


def ref_lines_reidentify(ref_lines, s, x,
                         fitted_lines=None,
                         colname_pixel="pixel", colname_params="params"):

    fitted_lines = fitted_lines_init(ref_lines)

    fitted_lines_reidentify(fitted_lines, ref_lines, s, x,
                            colname_pixel="pixel", colname_params="params")

    return fitted_lines


def get_ref_list1(ohlines, line_indices,
                  wvlsol0, x=None):
    ref_wvl = get_ref_wavelengths(ohlines, line_indices)
    line_list = get_ref_pixels(ref_wvl, wvlsol0, x=x)

    return line_list


def fitted_lines_init(ref_lines):

    orders = np.unique(ref_lines["order"])
    if len(orders) == 1:
        order = orders[0]
    elif len(orders) == 0:
        # if ref_lines has no entries (len(orders) == 0), just return empty DF.
        fitted_lines = pd.DataFrame(dict(wavelength=[],
                                         d_cent=[],
                                         cent_pixel0=[],
                                         order=[]))
        
        return fitted_lines
        #order = None
    else:
        raise ValueError("")

    _p = ref_lines.groupby(["group_id"])
    _p_pixel = _p["pixel"]

    # calculate shift to centroid pixel positions
    mean_pixel = _p_pixel.mean()
    d_cent_list = mean_pixel - _p_pixel.first()

    # centroid wavelength
    wvl_list = _p["wavelength"].mean()

    fitted_lines = pd.DataFrame(dict(wavelength=wvl_list.values,
                                     d_cent=d_cent_list.values,
                                     cent_pixel0=mean_pixel,
                                     order=order))

    return fitted_lines


def fitted_lines_reidentify(fitted_lines, ref_lines, s, x,
                            colname_pixel="pixel", colname_params="params",
                            ref_sigma0=1.5):

    ref_lines_groupby = ref_lines.groupby("group_id")

    ref_pixels = [row["pixel"].values for group_id, row
                  in ref_lines_groupby]

    if "sigma_pixel" in ref_lines:
        ref_sigma = [row["sigma_pixel"].values[0] for group_id, row
                     in ref_lines_groupby]
    else:
        ref_sigma = ref_sigma0

    from .reidentify import reidentify
    res = reidentify(s, ref_pixels, x=x, sigma_init=ref_sigma)

    params = [p for p, _, _ in res]
    x_list0 = [p[0] for p in params]
    x_list = x_list0 + fitted_lines["d_cent"]

    fitted_pixels = pd.DataFrame(dict(pixels=x_list,
                                      params=params))

    return fitted_pixels


    # fitted_lines_update_pixel(fitted_lines, fitted_pixels,
    #                           colname_pixel=colname_pixel,
    #                           colname_params=colname_params)


# def fitted_lines_update_pixel(fitted_lines, res,
#                               colname_pixel="pixel",
#                               colname_params=None):
#     x_list0 = [params[0] for params, _, _ in res]
#     x_list = x_list0 + fitted_lines["d_cent"]

#     fitted_lines[colname_pixel] = x_list

#     if colname_params is not None:
#         fitted_lines[colname_params] = [p for p, _, _ in res]


# ref_lines : pandas DataFrame

class RefLinesDBBase:
    def __init__(self, ref_loader):
        self._refdata = None
        self.ref_loader = ref_loader
        # from .igrins_config import get_config
        # self.config = get_config(config)

    def _get_refdata(self):

        if self._refdata is not None:
            return self._refdata

        refdata = self._load_refdata()
        self._refdata = refdata

        return refdata

    def _load_refdata(self):
        pass

    def get_ref_lines(self, o, wvl, x):
        pass

    def get_ref_lines_collection(self, order_list, wvl_list, x):
        """
        return RefLinesCollection instance
        """
        ref_lines_list = [self.get_ref_lines(o, wvl, x)
                          for o, wvl in zip(order_list, wvl_list)]

        ref_lines_coll = RefLinesCollection(ref_lines_list)
        return ref_lines_coll

    def identify_single_order(self, o, spec):
        """
        spec : an object with wvl_map & s_map attributes. Both are
        dictionary contains wavelength and spectra for each order
        """

        ref_lines_db = self

        wvl, s = spec.wvl_map[o], spec.s_map[o]
        x = np.arange(len(wvl))

        ref_lines = ref_lines_db.get_ref_lines(o, wvl, x)
        # ref_lines.line_centroids

        fitted_pixels_ = ref_lines.fit(s, x, update_self=True)

        fitted_pixels = pd.concat([ref_lines.line_centroids,
                                   fitted_pixels_],
                                  axis=1, levels=["order", "wavelength"])

        return fitted_pixels

    def identify(self, spec, ref_sigma=1.5):

        ref_lines_db = self
        order_list = sorted(spec.wvl_map.keys())
        wvl_list = [spec.wvl_map[o_] for o_ in order_list]
        s_list = [spec.s_map[o_] for o_ in order_list]

        x = np.arange(len(wvl_list[0]))

        ref_lines_col = ref_lines_db.get_ref_lines_collection(order_list,
                                                              wvl_list, x)

        fitted_pixels_col = ref_lines_col.fit(s_list, x, update_self=True,
                                              ref_sigma=ref_sigma)

        line_centroids_master = pd.concat([_r.line_centroids for _r
                                           in ref_lines_col.ref_lines_list])

        fitted_pixels_master = pd.concat([line_centroids_master,
                                          pd.concat(fitted_pixels_col)],
                                         axis=1)

        nic = ["order", "wavelength"]  # new_index_columns
        fitted_pixels_master = fitted_pixels_master.set_index(nic)

        return fitted_pixels_master


class SkyLinesDB(RefLinesDBBase):
    def _load_refdata(self):
        from .ref_data_sky import load_sky_ref_data
        # sky_refdata = load_sky_ref_data(self.config, band)
        sky_refdata = load_sky_ref_data(self.ref_loader)
        return sky_refdata

    def get_ref_lines(self, o, wvl, x):
        """
        return RefLines instance
        """
        sky_ref_data = self._get_refdata()

        ohlines_db = sky_ref_data["ohlines_db"]
        try:
            line_indices = sky_ref_data["ohline_indices"][o]
        except KeyError:
            line_indices = []

        _ref_lines = get_ref_list1(ohlines_db, line_indices,
                                  wvl, x=x)

        _ref_lines["order"] = o

        ref_lines = RefLines(_ref_lines)

        return ref_lines


class HitranSkyLinesDB(RefLinesDBBase):
    def _load_refdata(self):
        if self.ref_loader.band != "K":
            raise ValueError("only K band is supported")

        refdata0 = self.ref_loader.load("HITRAN_BOOTSTRAP_K")

        refdata = dict((int(k), v) for k, v in refdata0.items())

        return refdata

    def get_ref_lines(self, o, wvl, x):
        """
        return RefLines instance
        """
        ref_data = self._get_refdata()

        try:
            ref_wvl = ref_data[o]["wavelength_grouped"]
        except KeyError:
            ref_wvl = []

        _ref_pixels = get_ref_pixels(ref_wvl, wvl, x)
        _ref_pixels["sigma_pixel"] = 5
        _ref_pixels["order"] = o

        ref_lines = RefLines(_ref_pixels)

        return ref_lines


class Test:
    def __init__(self, config):
        from .igrins_config import get_config
        config = get_config(config)
        self.config = config

    def update_K(self, reidentified_lines_map,
                 orders_w_solutions,
                 wvl_solutions, s_list):
        # fn = "hitran_bootstrap_K_%s.json" % self.refdate
        # bootstrap_name = master_calib.get_master_calib_abspath(fn)
        # import json
        # bootstrap = json.load(open(bootstrap_name))

        from .master_calib import load_ref_data
        bootstrap = load_ref_data(config, band="K",
                                  kind="HITRAN_BOOTSTRAP_K")


        from . import hitran
        r, ref_pixel_list = hitran.reidentify(orders_w_solutions,
                                              wvl_solutions, s_list,
                                              bootstrap)
        # json_name = "hitran_reidentified_K_%s.json" % igrins_log.date
        # r = json.load(open(json_name))
        for i, s in r.items():
            ss = reidentified_lines_map[int(i)]
            ss0 = np.concatenate([ss[0], s["pixel"]])
            ss1 = np.concatenate([ss[1], s["wavelength"]])
            reidentified_lines_map[int(i)] = (ss0, ss1)

        return reidentified_lines_map


class RefLines:
    def __init__(self, ref_lines):
        self._ref_lines = ref_lines
        self.line_centroids = fitted_lines_init(ref_lines)

    def fit(self, s, x, update_self=True, ref_sigma=1.5):

        fitted_lines = self.line_centroids  # .copy()
        fitted_pixels = fitted_lines_reidentify(fitted_lines,
                                                self._ref_lines, s, x,
                                                ref_sigma0=ref_sigma)

        if update_self:
            ref_lines_update_pixel(self._ref_lines,
                                   fitted_pixels["params"])

        if update_self and "sigma_pixel" in self._ref_lines:
            ref_lines_update_sigma_pixel(self._ref_lines,
                                         fitted_pixels["params"])

        return fitted_pixels


class RefLinesCollection:
    def __init__(self, ref_lines_list):
        self.ref_lines_list = ref_lines_list

    def fit(self, s_list, x, update_self=True, ref_sigma=1.5):
        fitted_pixels_list = []
        for ref_lines, s in zip(self.ref_lines_list, s_list):
            _ = ref_lines.fit(s, x, update_self, ref_sigma=ref_sigma)
            fitted_pixels_list.append(_)

        return fitted_pixels_list


# class SpecReidentify:
#     def __init__(self, ref_lines_db):

#         self.wvl_map, self.s_map = {}, {}
#         self.ref_lines_db = ref_lines_db

#     def _load_wvl(self, band):

#         fn = "../calib/primary/20140525/SDC%s_20140525_0003.wvlsol_v0.json" % band
#         wvl_solution = json.load(open(fn))
#         wvl_map = dict(zip(wvl_solution["orders"],
#                            wvl_solution["wvl_sol"]))

#         return wvl_map

#     def _load_spec(self, band):

#         fn = "../calib/primary/20140525/SDC%s_20140525_0029.oned_spec.json" % band
#         s_list = json.load(open(fn))
#         s_map = dict(zip(s_list["orders"],
#                          s_list["specs"]))

#         return s_map

#     def get_spec_map(self, band):
#         if band not in self.wvl_map:
#             self.wvl_map[band] = self._load_wvl(band)

#         if band not in self.s_map:
#             self.s_map[band] = self._load_spec(band)

#         wvl_map = self.wvl_map[band]
#         s_map = self.s_map[band]

#         return wvl_map, s_map

#     def reidentify_order(self, band, o, wvl, s, x=None):

#         if x is None:
#             x = np.arange(len(wvl))

#         ref_lines0 = self.ref_lines_db.get_ref_lines(band, o, wvl, x)

#         fitted_lines = fitted_lines_init(ref_lines0)

#         self.reidentify_order_add(ref_lines0, s, x,
#                                   colname_pixel="pixel",
#                                   colname_params="params",
#                                   fitted_lines=fitted_lines)

#         ref_lines = ref_lines0.copy()
#         ref_lines_update_pixel(ref_lines,
#                                fitted_lines["params"])

#         return ref_lines, fitted_lines

#     def reidentify_order_add(self, ref_lines, s, x,
#                              colname_pixel,
#                              colname_params,
#                              fitted_lines=None):

#         if fitted_lines is None:
#             fitted_lines = fitted_lines_init(ref_lines)

#         fitted_lines_reidentify(fitted_lines, ref_lines, s, x,
#                                 colname_pixel=colname_pixel,
#                                 colname_params=colname_params)

#         return fitted_lines

#     def reidentify_band(self, band):

#         wvl_map, s_map = self.get_spec_map(band)
#         ref_lines_list, fitted_lines_list = [], []

#         for o in sorted(wvl_map.keys()):

#             wvl = wvl_map[o]
#             s = s_map[o]
#             x = np.arange(len(wvl))

#             _ = self.reidentify_order(band, o, wvl, s, x=x)
#             ref_lines, fitted_lines = _

#             if 0:
#                 self.reidentify_order_add(ref_lines, s, x,
#                                           colname_pixel,
#                                           colname_params,
#                                           fitted_lines=fitted_lines)

#             ref_lines_list.append(ref_lines)
#             fitted_lines_list.append(fitted_lines)

#         ref_lines_master = pd.concat(ref_lines_list)
#         fitted_lines_master = pd.concat(fitted_lines_list)

#         return ref_lines_master, fitted_lines_master


# from .recipe_helper import RecipeHelper


def helper_load_spec(helper, band, obsid):

    caldb = helper.get_caldb()
    #orders = caldb.load_resource_for((band, obsid), "orders")["orders"]

    basename = (band, obsid)

    s_list = caldb.load_item_from(basename, "ONED_SPEC_JSON")

    s_map = dict(zip(s_list["orders"],
                     s_list["specs"]))

    wvlsol_ = caldb.load_resource_for(basename, "wvlsol_v0")

    wvl_map = dict(zip(wvlsol_["orders"],
                       wvlsol_["wvl_sol"]))

    return s_map, wvl_map


class SampleSpec:
    def __init__(self, config_name, utdate, band, obsid):
        self.config_name = config_name
        self.utdate = utdate
        self.obsid = obsid
        self.band = band

        self.helper = RecipeHelper(config_name, utdate)

        self.s_map, self.wvl_map = helper_load_spec(self.helper, band, obsid)


def test_H():
    config_name = "../recipe.config"
    utdate = 20151130
    obsid = 50
    band = "H"

    spec = SampleSpec(config_name, utdate, band, obsid)
    ref_lines_db = SkyLinesDB(config=config_name)

    # test single order
    o = 115
    fitted_pixels = ref_lines_db.identify_single_order(band, o, spec)

    fitted_pixels_master = ref_lines_db.identify(band, spec)

    # just to prevent never-used warning
    fitted_pixels, fitted_pixels_master


def test_K():
    config_name = "../recipe.config"
    utdate = 20151130
    obsid = 50
    band = "K"

    spec = SampleSpec(config_name, utdate, band, obsid)
    ref_lines_db = SkyLinesDB(config=config_name)

    # test single order
    o = 75
    fitted_pixels = ref_lines_db.identify_single_order(band, o, spec)

    fitted_pixels_master = ref_lines_db.identify(band, spec)

    # just to prevent never-used warning
    fitted_pixels, fitted_pixels_master


def test_K_Hitran():
    config_name = "../recipe.config"
    utdate = 20151130
    obsid = 50
    band = "K"

    spec = SampleSpec(config_name, utdate, band, obsid)
    ref_lines_db = HitranSkyLinesDB(config=config_name)

    # test single order
    o = 75
    fitted_pixels = ref_lines_db.identify_single_order(band, o, spec)

    fitted_pixels_master = ref_lines_db.identify(band, spec)

    # just to prevent never-used warning
    fitted_pixels, fitted_pixels_master




def draw_fig():
    min_o, max_o = min(order_list), max(order_list)
    orders = range(min_o, max_o+1)

    no = len(orders)
    nx = max(len(_l) for _l in s_list)

    arr = np.empty((no, nx), dtype="float32")
    arr.fill(np.nan)

    for o, s in zip(order_list, s_list):
        indx = orders.index(o)
        arr[indx] = s


    vmax = np.percentile(arr[np.isfinite(arr)], 99)
    vmin = np.percentile(arr[np.isfinite(arr)], 1)

    ax = subplot(111)
    im = ax.imshow(s_list, 
                   extent=[-0.5, nx-0.5, min_o-0.5, max_o+0.5],
                   origin="lower",
                   aspect="auto", interpolation="none", 
                   cmap="gist_heat_r", vmax=vmax, vmin=vmin)

    peak = [r[2] for r in fitted_pixels_master["params"]]
    sct = ax.scatter(fitted_pixels_master["pixels"],
                     fitted_pixels_master["order"], 
                     c=peak, norm=im.norm, cmap=im.cmap,
                     edgecolor="w")


    peak = [r[2] for r in fitted_pixels_master["params"]]
    sct = ax.scatter(fitted_pixels_master["pixels"],
                     fitted_pixels_master["order"], 
                     c=peak, norm=im.norm, cmap=im.cmap,
                     edgecolor="y")


if 0:
    ll = []
    for r0, r1 in zip(ref_lines_col.ref_lines_list,
                      ref_lines_col1.ref_lines_list):
        if len(r0._ref_lines):
            ll.extend(r0._ref_lines["pixel"] - r1._ref_lines["pixel"])


if 0:
    test_H()
    test_K()

    test_K_Hitran()

    # t = TestFitH()
    # t.test_single_order()
    # t.test_all_order()

# if 1:

#     t = TestFitK()
#     t.test_single_order()
#     t.test_all_order()


# if 0:
#         utdate = 20140526
#         obsid = 156
#         band = "K"

#         config_name = "../recipe.config"
#         spec = SampleSpec(config_name, utdate, band, obsid)

#         from hitran import Hitran
#         med_filter = Hitran.get_median_filtered_spec

#         o = 74

#         s0 = spec.s_map[74]
#         wvl = spec.wvl_map[74]
#         s = med_filter(wvl, s0)
#         x = np.arange(len(s))

#         from igrins_config import get_config
#         config = get_config("../recipe.config")

#         from master_calib import load_ref_data
#         bootstrap = load_ref_data(config, band="K",
#                                   kind="HITRAN_BOOTSTRAP_K")

#         ref_wvl = bootstrap[str(o)]["wavelength_grouped"]
#         _ref_pixels = get_ref_pixels(ref_wvl, wvl, x=None)
#         _ref_pixels["sigma_pixel"] = 5
#         _ref_pixels["order"] = o

#         ref_lines = RefLines(_ref_pixels)
#         fitted_lines0 = ref_lines.fit(s, x, update_self=True)
#         fitted_lines = ref_lines.fit(s, x, update_self=True)

# if 0:
#         ref_lines_db = SkyLinesDB(config=config_name)

#         # test 1
#         o = 115

#         wvl, s = spec.wvl_map[o], spec.s_map[o]
#         x = np.arange(len(wvl))

#         ref_lines = ref_lines_db.get_ref_lines(band, o, wvl, x)
#         ref_lines.line_centroids

#         fitted_pixels = ref_lines.fit(s, x, update_self=True)

#         fitted_pixels

# if 0:
#     #####


#     wvlsol_2d = fitted_pixels_col.fit()

#     fitted_pixels_col.diff(wvlsol_2d)


# if 0:
#     plot(wvl_map[o], s_map[o])

#     ref_wvl = ref_lines_nested(ref_lines, "wavelength")
#     for w in ref_wvl:
#         plot(w, np.zeros_like(w), "ro-")

# if 0:
#     dd = {}
#     for i, f in pd.groupby(fitted_pixels_master, "order"):
#         dd[int(i)] = f["cent_pixel0"], f["cent_pixel0"] - f["pixels"]
#         plot(f["pixels"], f["cent_pixel0"] - f["pixels"], "o", 
#              label="%d" % i)
