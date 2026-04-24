# import astropy.io.fits as pyfits
import numpy as np
import pandas as pd

from astropy.table import Table
import astrodata
import igrins_instruments

from igrinsdr.igrins.primitives_igrins import get_ref_spectra, get_ref_data

# from igrinsdr.igrins.procedures.procedures_register import _get_offset_transform_between_2spec

from igrinsdr.igrins.procedures.identified_lines import IdentifiedLines
from igrinsdr.igrins.procedures.apertures import Apertures
from igrinsdr.igrins.procedures.echellogram import Echellogram
# from igrinsdr.igrins.primitives_igrins import _get_wavelength_solutions

# fn = "N20240429S0204_K.fits"
band = "H"
# band = "K"

fnroot = "N20240429S0204"

# fn = f"{fnroot}_{band}.fits"
# ad = astrodata.open(fn)
# adlist = [ad]

fnout = f"../../test_i2/{fnroot}_{band}_arc.fits"
adout = astrodata.open(fnout)
adinputs = [adout]

from collections import namedtuple
Spec = namedtuple("Spec", ["s_map", "wvl_map"])

from igrinsdr.igrins.procedures.reidentify import reidentify
from scipy.interpolate import interp1d

def flatten(l):
    return [r_ for r1 in l for r_ in r1]


def get_group_flag_generator():
    import itertools
    p = itertools.chain([1], itertools.repeat(0))
    return p


def get_group_flags(l):
    return [f for r1 in l for r_, f in zip(r1, get_group_flag_generator())]


def identify_lines_from_spec(orders, spec_data, wvlsol,
                             ref_lines_db, ref_lines_db_hitrans,
                             ref_sigma=1.5):
    small_list = []
    small_keys = []

    spec = Spec(dict(zip(orders, spec_data)),
                dict(zip(orders, wvlsol)))

    fitted_pixels_oh = ref_lines_db.identify(spec, ref_sigma=ref_sigma)
    small_list.append(fitted_pixels_oh)
    small_keys.append("OH")

    # if obsset.band == "K":
    if ref_lines_db_hitrans is not None:
        fitted_pixels_hitran = ref_lines_db_hitrans.identify(spec)
        small_list.append(fitted_pixels_hitran)
        small_keys.append("Hitran")

    fitted_pixels = pd.concat(small_list,
                              keys=small_keys,
                              names=["kind"],
                              axis=0)

    return fitted_pixels

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
    def __init__(self, ref_file):
        self._refdata = None
        self._ref_file = ref_file

    def _load_refdata(self):

        tbl = Table.read(self._ref_file, format="fits") # "ref_lines_oh.fits"
        df = tbl.to_pandas()

        ref_wvl_dict = {}
        for o, grouped_by_order in df.groupby("order"):
            ref_wvl_dict[o] = list(v.values for gid, v in grouped_by_order.groupby("gid")["um"])

        return ref_wvl_dict

    def get_ref_lines(self, o, wvl, x):
        """
        return RefLines instance
        """
        ref_wvl_dict = self._get_refdata()

        ref_wvl = ref_wvl_dict.get(o, [])
        _ref_lines = get_ref_pixels(ref_wvl, wvl, x=x)

        _ref_lines["order"] = o

        ref_lines = RefLines(_ref_lines)

        return ref_lines

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

class RefLinesCollection:
    def __init__(self, ref_lines_list):
        self.ref_lines_list = ref_lines_list

    def fit(self, s_list, x, update_self=True, ref_sigma=1.5):
        fitted_pixels_list = []
        for ref_lines, s in zip(self.ref_lines_list, s_list):
            _ = ref_lines.fit(s, x, update_self, ref_sigma=ref_sigma)
            fitted_pixels_list.append(_)

        return fitted_pixels_list

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

def ref_lines_update_sigma_pixel(ref_lines, res,
                                 colname_i="sigma_pixel",
                                 colname_f="sigma_pixel"):

    _p = ref_lines.groupby(["group_id"])[colname_i]
    sigma_list = [params[1] for params in res]

    sigma_pixel_f = [[sig] * len(row.values) for (v, row), sig
                     in zip(_p, sigma_list)]

    ref_lines[colname_f] = flatten(sigma_pixel_f)

    return ref_lines

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

    res = reidentify(s, ref_pixels, x=x, sigma_init=ref_sigma)

    params = [p for p, _, _ in res]
    x_list0 = [p[0] for p in params]
    x_list = x_list0 + fitted_lines["d_cent"]

    fitted_pixels = pd.DataFrame(dict(pixels=x_list,
                                      params=params))

    return fitted_pixels

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


if True:
        from igrinsdr.igrins.primitives_igrins import (get_ref_path,
                                                       identify_lines_from_spec)
        ad = adinputs[0]

        wvlsol0 = ad[0].WVLSOL0
        orders, wvlsol = wvlsol0["orders"], wvlsol0["wavelengths"]
        wvlsol_by_order = dict(zip(orders, wvlsol))

        ref_file = get_ref_path(band, "ref_lines_oh") # "ref_lines_oh.fits"
        ref_lines_db = SkyLinesDB(ref_file.open("rb"))
        ref_lines_db._load_refdata()

        from itertools import count
        counter = count(1)
        ref_data = [(o, gid, line)
                    for o, lines_list in ref_lines_db._get_refdata().items()
                    for gid, lines in zip(counter, lines_list)
                    for line in lines]

        df_ref_data0 = pd.DataFrame(ref_data, columns=["order", "gid", "wavelength"])

        from scipy.interpolate import interp1d
        x = np.arange(2048)
        # for each order, add pixel coordinate from the initial wvlsol
        for order, grouped in df_ref_data0.groupby("order"):
            wvl = wvlsol_by_order.get(order, None)
            if wvl is not None:
                knots = interp1d(wvl, x,
                                 bounds_error=False, assume_sorted=True, fill_value=np.nan)
                df_ref_data0.loc[grouped.index, "pixel"] = knots(grouped["wavelength"])

        # flags groups that any of the line in the group has a pixel value of nan.
        msk = df_ref_data0.groupby("gid")["pixel"].apply(lambda pixels:
                                                        np.all(np.isfinite(pixels)))
        # msk has an index of "gid". We will filter the dataframe using this mask.
        # Note that there can be multiple rows wit same gid, and indexing with mask
        # gives a warning of

        # Boolean Series key will be reindexed to match DataFrame index

        # FIXME check if there is a better way of doing this.
        df_ref_data = df_ref_data0.set_index("gid")[msk]

        # The filtered df_ref_data should only have valid pixels.
        def prepare_gaussian_group(x: np.ndarray, s: np.ndarray, lines: np.ndarray,
                                   sigma_init=1.5,
                                   max_sigma_scale=2,
                                   fitrange_scale=2.5):
            pass

        # spectrum at the slit center
        multi_spec = ad[0].SPEC1D_MULTI
        slit_centers = multi_spec["slit_centers"][0].astype("float32")
        i_slit_center = len(slit_centers) // 2

        spec_data = multi_spec["multispec"][:, i_slit_center, :]
        spec_by_order = dict(zip(multi_spec["orders"], spec_data))

        # we prepare a dataframe index of (order, gid)
        grouped = df_ref_data.groupby(["order", "gid"])
        df_fit = pd.DataFrame(dict(mean_pixel=grouped["pixel"].mean(),
                                   mean_wvl=grouped["wavelength"].mean()))

        # For each group, we fit the sliced data with multiple gaussian.
        for (o, gid), grp in grouped:
            if (s := spec_by_order.get(o, None)) is not None:
                r = fit_gaussian_group(x, s, grp["pixel"])
                df_fit.loc[(o, gid), ["shift", "sigma", "height", "baseline"]] = r[0]



        wvlsol0 = ad[0].WVLSOL0
        orders, wvlsol = wvlsol0["orders"], wvlsol0["wavelengths"]

        multi_spec = ad[0].SPEC1D_MULTI
        spec_data = multi_spec["multispec"][:, 2, :]

        spec = Spec(dict(zip(orders, spec_data)),
                    dict(zip(orders, wvlsol)))

        order_list = sorted(spec.wvl_map.keys())
        wvl_list = [spec.wvl_map[o_] for o_ in order_list]
        s_list = [spec.s_map[o_] for o_ in order_list]

        x = np.arange(len(wvl_list[0]))

        ref_lines_col = ref_lines_db.get_ref_lines_collection(order_list,
                                                              wvl_list, x)

        ref_line = ref_lines_col.ref_lines_list[10]

        """
In [74]: ref_line._ref_lines
Out[74]: 
   wavelength  valid  group_flag  group_id        pixel  order
0    1.650224   True           1         1   187.400457    108
1    1.650249   True           0         1   189.359747    108
2    1.655372   True           1         2   618.168197    108
3    1.655391   True           0         2   619.750039    108
4    1.658580   True           1         3   900.909711    108
5    1.658685   True           1         4   910.361408    108
6    1.660999   True           1         5  1121.499613    108
7    1.661205   True           1         6  1140.556964    108
8    1.668878   True           1         7  1881.514941    108
9    1.668962   True           0         7  1890.014533    108

        """

        # for a single specta of an oder.
        s = s_list[10]
        update_self = False
        ref_sigma = 4
        r = ref_line.fit(s, x, update_self, ref_sigma=ref_sigma)

        # test data for test_fit_ohlines.py
        j = dict(s=s.tolist(), ref_line=ref_line._ref_lines.to_dict())
        import json
        json.dump(j, open("test_data_for_fit_ohlines.json", "w"))

if True:
        #
        from igrinsdr.igrins.primitives_igrins import (get_ref_path,
                                                       identify_lines_from_spec)
        # from igrinsdr.igrins.procedures.ref_lines_db import SkyLinesDB

        ad = adinputs[0]

        # multi_spec = obsset.load("multi_spec_fits")
        multi_spec = ad[0].SPEC1D_MULTI
        slit_centers = multi_spec["slit_centers"][0].astype("float32")

        # # just to retrieve order information
        # wvlsol_v0 = obsset.load_resource_for("wvlsol_v0")
        # orders = wvlsol_v0["orders"]
        # wvlsol = wvlsol_v0["wvl_sol"]

        wvlsol0 = ad[0].WVLSOL0
        orders, wvlsol = wvlsol0["orders"], wvlsol0["wavelengths"]

        # ref_lines_db = SkyLinesDB(config=obsset.get_config())
        ref_file = get_ref_path(band, "ref_lines_oh") # "ref_lines_oh.fits"
        ref_lines_db = SkyLinesDB(ref_file.open("rb"))

        ref_lines_db_hitrans = None
        # if obsset.rs.get_resource_spec()[1] == "K":
        #     ref_lines_db_hitrans = HitranSkyLinesDB(obsset.rs.master_ref_loader)
        # else:
        #     ref_lines_db_hitrans = None

        # keys = []
        fitted_pixels_list = []


        for i, slit_center in enumerate(slit_centers):
            d = multi_spec["multispec"][:, i, :]
            fitted_pixels_ = identify_lines_from_spec(orders, d, wvlsol,
                                                      ref_lines_db,
                                                      ref_lines_db_hitrans)

            fitted_pixels_list.append(fitted_pixels_)

        # for hdu in multi_spec:
        #     slit_center = hdu.header["FSLIT_CN"]
        #     keys.append(slit_center)

        #     fitted_pixels_ = identify_lines_from_spec(orders, hdu.data, wvlsol,
        #                                               ref_lines_db,
        #                                               ref_lines_db_hitrans)

        #     fitted_pixels_list.append(fitted_pixels_)

        # concatenate collected list of fitted pixels.
        fitted_pixels_master = pd.concat(fitted_pixels_list,
                                         keys=slit_centers,
                                         names=["slit_center"],
                                         axis=0)

        # storing multi-index seems broken. Enforce reindexing.
        tbl = Table.from_pandas(fitted_pixels_master.reset_index())

        # tbl["params"] is type of object. We need to transform it to float[4]
        tbl["params"] = np.array(list(tbl["params"]))

        ad[0].LINEFIT = tbl
