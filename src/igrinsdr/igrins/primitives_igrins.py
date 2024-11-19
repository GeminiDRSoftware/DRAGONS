#
#                                                                       DRAGONS
#
#                                                         primitives_igrins.py
# ------------------------------------------------------------------------------

import numpy as np
import json
import pandas as pd
from collections import namedtuple
from importlib.resources import files

from astropy.io import fits
from astropy.table import Table

import astrodata
from gempy.gemini import gemini_tools as gt

from geminidr.gemini.primitives_gemini import Gemini
from geminidr.core.primitives_nearIR import NearIR
from geminidr.gemini.lookups import DQ_definitions as DQ

from . import parameters_igrins

from .lookups import timestamp_keywords as igrins_stamps

from .json_helper import dict_to_table

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------

from .procedures.procedure_dark import (make_guard_n_bg_subtracted_images,
                                        estimate_amp_wise_noise)

from .procedures.trace_flat import trace_flat_edges, table_to_poly
from .procedures.iter_order import iter_order
from .procedures.reference_pixel import fix_pattern_using_reference_pixel

from .procedures.trace_flat import table_to_poly
from .procedures.iter_order import iter_order
from .procedures.apertures import Apertures
from .procedures.match_orders import match_orders

import matplotlib
import warnings

from .procedures.procedures_register import _get_offset_transform_between_2spec
from .procedures.line_identify_simple import match_lines1_pix
from .procedures.identified_lines import IdentifiedLines
from .procedures.echellogram import Echellogram
from .procedures.fit_affine import fit_affine_clip
from .procedures.ecfit import get_ordered_line_data, fit_2dspec  # , check_fit
from .procedures.ref_lines_db import SkyLinesDB

from .procedures.sky_spec_helper import _get_slices
from .procedures.process_wvlsol_volume_fit import (_append_offset,
                                                   _filter_points,
                                                   _volume_poly_fit)

from .procedures.nd_poly import NdPolyNamed
from .procedures.process_derive_wvlsol import fit_wvlsol, _convert2wvlsol

from .procedures.readout_pattern_helper import remove_pattern


from .procedures.slit_profile import (extract_slit_profile,
                                      _get_norm_profile_ab,
                                      _get_norm_profile,
                                      _get_profile_func_from_dict,
                                      make_slitprofile_map)
# from igrinsdr.igrins.json_helper import dict_to_table

from .procedures.spec_extract_w_profile import extract_spec_using_profile

from .procedures.astropy_poly_helper import deserialize_poly_model
from .procedures.iraf_helper import get_wat_spec, default_header_str
from .procedures.iraf_helper import invert_order

from .procedures.correct_distortion import get_rectified_2dspec
from .procedures.shifted_images import ShiftedImages
from .procedures.badpixel_mask import make_igrins_hotpixel_mask, make_igrins_deadpixel_mask

def _get_wavelength_solutions(affine_tr_matrix, zdata,
                              new_orders):
    """
    new_orders : output orders

    convert (x, y) of zdata (where x, y are pixel positions and z
    is wavelength) with affine transform, then derive a new wavelength
    solution.

    """
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


def get_ref_data(band):
    pkgroot = files("igrinsdr") # FIXME okay to use hardcoded module name?
    dataroot = pkgroot / "igrins/lookups/ref_data"

    # FIXME use hardcoded file names for now.
    j = dict(
        ref_spec=json.load(open(dataroot / f"SDC{band}_20140525_0029.oned_spec.json")),
        identified_lines_v0=json.load(open(dataroot / f"SKY_SDC{band}_20140525.identified_lines_v0.json")),
        echellogram_data=json.load(open(dataroot / f"SDC{band}_20140525.echellogram.json")),
    )
    return j

def get_ref_line_path():
    pkgroot = files("igrinsdr") # FIXME okay to use hardcoded module name?
    return pkgroot / "igrins/lookups/ref_data" / "ref_lines_oh.fits"

Spec = namedtuple("Spec", ["s_map", "wvl_map"])

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

# for spliiting A & B
def groupby_pq(pql):
    """
    pql : list of (p, q)
    """

    pql = np.array(pql)

    xy0 = np.percentile(pql, 25, axis=0)
    xy1 = np.percentile(pql, 75, axis=0)

    dxy = xy1 - xy0

    dd = np.dot(pql - xy0, dxy)
    dd /= (dd.max() - dd.min())

    return [d1 < 0.5 for d1 in dd]

assert groupby_pq([(0, 0), (2.5, 2.5), (2.4, 2.6), (0, 0.1)]) == [True, False, False, True]

def splitAB(adinputs):
    adinputsA, adinputsB = [], []

    ad0 = adinputs[0]
    if "FRMTYPE" in ad0.phu:
        group_by = "frametype"
    else:
        group_by = "pq"

    if group_by == "frametype":
        frametypes = [ad.phu["FRMTYPE"] for ad in adinputs]
        frametype_set = set(frametypes)
        assert len(frametype_set) == 2

        if "A" in frametype_set:
            groupA_name = "A"
        elif "ON" in frametype_set:
            groupA_name = "ON"
        else:
            groupA_name = frametypes[0]

        flags = [ft == groupA_name for ft in frametypes]

    else:
        # we groupby pq
        pql = [(ad.phu["POFFSET"], ad.phu["QOFFSET"]) for ad in adinputs]

        flags = groupby_pq(pql)

    for flag, ad in zip(flags, adinputs):
        if flag:
            adinputsA.append(ad)
        else:
            adinputsB.append(ad)

    return adinputsA, adinputsB


def subtract_ab(dataA, dataB, varA, varB, mask,
                # allow_no_b_frame=False,
                remove_level=2,
                remove_amp_wise_var=False,
                # interactive=False,
                # cache_only=False
                ):

    if remove_level == "auto":
        remove_level = 2

    if remove_amp_wise_var == "auto":
        remove_amp_wise_var = False

    # if interactive:
    #     params = run_interactive(obsset,
    #                              data_minus_raw, data_plus, bias_mask,
    #                              remove_level, remove_amp_wise_var)

    #     print("returned", params)
    #     if not params["to_save"]:
    #         print("canceled")
    #         return

    #     remove_level = params["remove_level"]
    #     remove_amp_wise_var = params["amp_wise"]

    data = remove_pattern(dataA - dataB, mask=mask,
                          remove_level=remove_level,
                          remove_amp_wise_var=remove_amp_wise_var)

    var = remove_pattern(varA + varB, remove_level=1,
                        remove_amp_wise_var=False)

    return data, var


def get_wat_cards(fit_results_tbl):

    fit_results_map = dict(zip(fit_results_tbl["key"], fit_results_tbl["encoded"]))
    fitted_model_encoded = fit_results_map["fitted_model"]
    orders = json.loads(fit_results_map["orders"])
    modeul_name, class_name, serialized = json.loads(fitted_model_encoded)

    p = deserialize_poly_model(modeul_name, class_name, serialized)

    # save as WAT fits header
    xx = np.arange(0, 2048)
    xx_plus1 = np.arange(1, 2048+1)

    from astropy.modeling import models, fitting

    # We convert 2d chebyshev solution to a seriese of 1d
    # chebyshev.  For now, use naive (and inefficient)
    # approach of refitting the solution with 1d. Should be
    # reimplemented.

    p1d_list = []
    for o in orders:
        oo = np.empty_like(xx)
        oo.fill(o)
        wvl = p(xx, oo) / o * 1.e4  # um to angstrom

        p_init1d = models.Chebyshev1D(domain=[1, 2048],
                                      degree=p.x_degree)
        fit_p1d = fitting.LinearLSQFitter()
        p1d = fit_p1d(p_init1d, xx_plus1, wvl)
        p1d_list.append(p1d)

    wat_list = get_wat_spec(orders, p1d_list)

    cards = [fits.Card.fromstring(l.strip())
             for l in default_header_str]

    wat = "wtype=multispec " + " ".join(wat_list)
    char_per_line = 68
    num_line, remainder = divmod(len(wat), char_per_line)
    for i in range(num_line):
        k = "WAT2_%03d" % (i+1,)
        v = wat[char_per_line*i:char_per_line*(i+1)]
        c = fits.Card(k, v)
        cards.append(c)

    if remainder > 0:
        i = num_line
        k = "WAT2_%03d" % (i+1,)
        v = wat[char_per_line*i:]
        c = fits.Card(k, v)
        cards.append(c)

    return cards


def get_wat_header(wat_table, wavelength_increasing_order=False):

    cards = [fits.Card.fromstring(s) for s in wat_table['cards']]
    header = fits.Header(cards)

    # hdu = obsset.load_resource_sci_hdu_for("wvlsol_fits")
    if wavelength_increasing_order:
        header = invert_order(header)

        def convert_data(d):
            return d[::-1]
    else:

        def convert_data(d):
            return d

    return header, convert_data




@parameter_override
class Igrins(Gemini, NearIR):
    """
    This class inherits from the level above.  Any primitives specific
    to IGRINS can go here.
    """

    tagset = {"GEMINI", "IGRINS"}

    def _initialize(self, adinputs, **kwargs):
        self.inst_lookups = 'geminidr.igrins.lookups'
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_igrins)
        # Add IGRINS specific timestamp keywords
        self.timestamp_keys.update(igrins_stamps.timestamp_keys)

    def selectFrame(self, adinputs=None, **params):
        """Filter the adinputs by its FRMTYPE value in the header.
        """
        frmtype = params["frmtype"]
        adoutputs = [ad for ad in adinputs
                     if frmtype in ad.hdr['FRMTYPE']]
        return adoutputs

    def streamPatternCorrected(self, adinputs=None, **params):
        """
        make images with Readout pattern corrected. And add them to streams.

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        rpc_mode = params.get("rpc_mode")
        assert rpc_mode == "full"
        # FIXME: only 'full' mode is supported for now, which will create
        # images using the methods of ['guard', 'level2', 'level3']

        dlist = [ad[0].data for ad in adinputs]
        hdu_list = make_guard_n_bg_subtracted_images(dlist,
                                                     rpc_mode=rpc_mode,
                                                     bias_mask=None,
                                                     log=log)
        for (name, dlist) in hdu_list:
            # name: the name of the correction method applied. One of ["GUARD",
            # "LEVEL2", "LEVEL3"]
            # dlist : list of numpy images
            adoutputs = []
            for ad0, d in zip(adinputs, dlist):
                # we create new astrodata object based on the input's header.
                hdu = fits.ImageHDU(data=d, header=ad0[0].hdr,
                                    name='SCI')
                ad = astrodata.create(ad0.phu, [hdu])
                gt.mark_history(ad, primname=self.myself(),
                                keyword="RPC")

                adoutputs.append(ad)

            self.streams[f"RPC_{name}"] = adoutputs

        return adinputs

    def estimateNoise(self, adinputs=None, **params):
        """Estimate the noise characteriscs for images in each streams. The resulting
        table is added to a 'ESTIMATED_NOISE' stream
        """

        # filenames that will be used in the table.
        filenames = [ad.filename for ad in adinputs]

        kdlist = [(k[4:], [ad[0].data for ad in adlist])
                  for k, adlist in self.streams.items()
                  if k.startswith("RPC_")]

        df = estimate_amp_wise_noise(kdlist, filenames=filenames)
        # df : pandas datafrome object.

        # Convert it to astropy.Table and then to an astrodata object.
        tbl = Table.from_pandas(df)
        phu = fits.PrimaryHDU()
        ad = astrodata.create(phu)

        astrodata.add_header_to_table(tbl)
        ad.EST_NOISE = tbl
        # ad.append(tbl, name='EST_NOISE')

        self.streams["ESTIMATED_NOISE"] = [ad]

        return adinputs

    def selectStream(self, adinputs=None, **params):
        stream_name = params["stream_name"]
        return self.streams[stream_name]

    def addNoiseTable(self, adinputs=None, **params):
        """
        The table from the 'EST_NOISE' stream will be appended to the input.
        """
        # adinputs should contain a single ad of stacked dark. We attach table
        # to the stacked dark.

        ad = adinputs[0]

        ad_noise_table = self.streams["ESTIMATED_NOISE"][0]
        del self.streams["ESTIMATED_NOISE"]

        ad.EST_NOISE = ad_noise_table.EST_NOISE
        # ad.append(ad_noise_table.EST_NOISE, name="EST_NOISE")

        return adinputs

    def setSuffix(self, adinputs=None, **params):
        suffix = params["suffix"]

        # Doing this also makes the output files saved.
        adinputs = self._markAsCalibration(adinputs, suffix=suffix,
                                           primname=self.myself(),
                                           keyword="NOISETABLE")

        return adinputs

    def someStuff(self, adinputs=None, **params):
        """
        Write message to screen.  Test primitive.

        Parameters
        ----------
        adinputs
        params

        Returns
        -------

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        for ad in adinputs:
            log.status('I see '+ad.filename)

            gt.mark_history(ad, primname=self.myself(), keyword="TEST")
            ad.update_filename(suffix=params['suffix'], strip=True)

        return adinputs


    @staticmethod
    def _has_valid_extensions(ad):
        """ Check that the AD has a valid number of extensions. """

        # this needs to be updated at appropriate.
        return len(ad) in [1]

    def determineSlitEdges(self, adinputs=None, **params):

        ad = adinputs[0]

        # ll = trace_flat_edges(ad[0].data)
        # print(ad.info())
        # tbl = Table(ll)

        # ad.SLITEDGE = tbl

        for ext in ad:
            ll = trace_flat_edges(ext.data)
            tbl = Table(ll)

            ext.SLITEDGE = tbl

        return adinputs

    def maskBeyondSlit(self, adinputs=None, **params):

        ad = adinputs[0]

        for ext in ad:
            tbl = ext.SLITEDGE

            pp = table_to_poly(tbl)

            from geminidr.gemini.lookups import DQ_definitions as DQ
            mask = np.empty((2048, 2048), dtype=DQ.datatype)
            mask.fill(DQ.unilluminated)
            for o, sl, m in iter_order(pp):
                mask[sl][m] = 0

            ext.mask |= mask

        return adinputs

    def normalizeFlat(self, adinputs=None, **params):

        ad = adinputs[0]

        for ext in ad:
            tbl = ext.SLITEDGE

            pp = table_to_poly(tbl)

            ext.FLAT_ORIGINAL = ext.data.copy()
            d = ext.data
            dq_mask = (ext.mask & DQ.unilluminated).astype(bool)
            d[dq_mask] = np.nan

            # Very primitive normarlization. Should be improved.
            for o, sl, m in iter_order(pp):
                dn = np.ma.array(d[sl], mask=~m).filled(np.nan)
                s = np.nanmedian(dn,
                                 axis=0)
                d[sl][m] = (dn / s)[m]

        return adinputs

    def fixIgrinsHeader(self, adinputs, **params):
        ad = adinputs[0]

        for ad in adinputs:
            for ext in ad:
                for desc in ('saturation_level', 'non_linear_level'):
                    kw = ad._keyword_for(desc)
                    if kw not in ext.hdr:
                        ext.hdr[kw] = (1.e5, "Test")
                        # print ("FIX", kw, ext.hdr.comments[kw])


        return adinputs

    def referencePixelsCorrect(self, adinputs, **params):
        for ad in adinputs:
            for ext in ad:
                # FIXME we may want different default behavior btw IG1 and IG2
                if params["apply_reference_pixels_correction"]:
                    ext.data = fix_pattern_using_reference_pixel(ext.data)

        return adinputs

    def _get_ad_flat(self, ad):
        calreturns = self.caldb.get_calibrations([ad, ad], caltype="processed_flat")
        for fn, mode in zip(*calreturns):
            assert mode == "user_cals"  # for now we assume userdb.
            return astrodata.open(fn)

    def _get_ad_sky(self, ad):
        calreturns = self.caldb.get_calibrations([ad, ad], caltype="processed_arc")
        for fn, mode in zip(*calreturns):
            assert mode == "user_cals"  # for now we assume userdb.
            return astrodata.open(fn)

    def extractSimpleSpec(self, adinputs, **params):
        # from recipe_system import cal_service
        # caldb = cal_service.set_local_database()
        # procmode = 'sq' if self.mode == 'sq' else None
        # c = caldb.get_calibrations(adinputs, caltype="processed_flat", procmode=procmode)

        ad = adinputs[0]

        ad_flat = self._get_ad_flat(ad)

        tbl = ad_flat[0].SLITEDGE
        ap = Apertures(tbl)

        # FIXME we need to apply the badpixel mask.
        s = ap.extract_spectra_simple(ad[0].data, f1=0., f2=1.)

        # t = Table(s,
        #           names=(f"{o}" for o in ap.orders_to_extract))
        ss = [np.array(s1, dtype='float32') for s1 in s]
        t = Table([ap.orders_to_extract, ss], names=['orders_initial', 'specs'])

        ad[0].SPEC1D = t
        ad[0].SLITEDGE = tbl

        return adinputs


    def identifyOrders(self, adinputs):
        ad = adinputs[0]
        spec1d = ad[0].SPEC1D

        s_list_ = spec1d["specs"]
        s_list = [np.array(s, dtype=np.float64) for s in s_list_]

        band = ad.band() # phu["BAND"]

        ref_spectra = get_ref_data(band)["ref_spec"]
        # ref_spectra = json.load(open(f"SKY_{band}.oned_spec.json"))

        orders_ref = ref_spectra["orders"]
        s_list_ref = ref_spectra["specs"]

        # match the orders of s_list_src & s_list_dst
        delta_indx, new_orders = match_orders(orders_ref, s_list_ref,
                                              s_list)

        spec1d["orders"] = new_orders
        order_map = dict(zip(spec1d["orders_initial"], new_orders))
        ad[0].SLITEDGE["order"] = [order_map[o] for o in ad[0].SLITEDGE["order"]]

        # ad[0].SPEC1D_NEW = spec1d
        return adinputs

    def identifyLinesAndGetWvlsol(self, adinputs, **params):
        ad = adinputs[0]

        tgt_spec = ad[0].SPEC1D

        band = ad.band() # phu["BAND"]

        ref_data = get_ref_data(band)

        ref_spec = ref_data["ref_spec"]

        intersected_orders, d = _get_offset_transform_between_2spec(ref_spec,
                                                                    tgt_spec)

        l = ref_data["identified_lines_v0"]
        offsetfunc_map = dict(zip(intersected_orders, d["sol_list"]))

        identified_lines_ref = IdentifiedLines(l)
        ref_map = identified_lines_ref.get_dict()

        identified_lines_tgt = IdentifiedLines(l)
        identified_lines_tgt.update(dict(wvl_list=[], ref_indices_list=[],
                                         pixpos_list=[], orders=[]))

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

            identified_lines_tgt.append_order_info(int(o), wvl, indices, pixpos)

        # We now fit affine transform

        ap = Apertures(ad[0].SLITEDGE)

        xy_list_tgt = identified_lines_tgt.get_xy_list_from_pixlist(ap)

        # echellogram_data = json.load(open(f"echellogram_{band}.json"))
        echellogram_data = ref_data["echellogram_data"]

        echellogram = Echellogram.from_dict(echellogram_data)

        xy_list_ref = identified_lines_tgt.get_xy_list_from_wvllist(echellogram)

        assert len(xy_list_tgt) == len(xy_list_ref)

        affine_tr, mm = fit_affine_clip(np.array(xy_list_ref),
                                        np.array(xy_list_tgt))

        d = dict(xy1f=xy_list_ref, xy2f=xy_list_tgt,
                 affine_tr_matrix=affine_tr.get_matrix(),
                 affine_tr_mask=mm)

        # we now get new wavelength solution
        affine_tr_matrix = d["affine_tr_matrix"]

        orders = tgt_spec["orders"]
        wvl_sol = _get_wavelength_solutions(affine_tr_matrix,
                                            echellogram.zdata,
                                            orders)

        ad[0].WVLSOL0 = Table([orders, wvl_sol], names=['orders', 'wavelengths'])

        ad.update_filename(suffix=params['suffix'], strip=True)

        return adinputs

    def extractSpectraMulti(self, adinputs, **params):

        ad = adinputs[0]

        n_slice_one_direction = 2
        slice_center, slice_up, slice_down = _get_slices(n_slice_one_direction)

        data = ad[0].data

        ap = Apertures(ad[0].SLITEDGE)

        slices = slice_down[::-1] + [slice_center] + slice_up
        slit_centers = [0.5*(s1+s2) for (s1, s2) in slices]

        ss = []
        for s1, s2 in slices:
            s = ap.extract_spectra_simple(data, s1, s2)
            ss.append(s)
        ss = np.array(ss)

        orders = ap.orders_to_extract
        tbl = Table([orders, [ss[:, i, :] for i in range(ss.shape[1])], [slit_centers]*len(orders)],
                    names=["orders", "multispec", "slit_centers"])

        ad[0].SPEC1D_MULTI = tbl

        return adinputs

    def identifyMultiline(self, adinputs, **params):

        ad = adinputs[0]

        # multi_spec = obsset.load("multi_spec_fits")
        multi_spec = ad[0].SPEC1D_MULTI

        # # just to retrieve order information
        # wvlsol_v0 = obsset.load_resource_for("wvlsol_v0")
        # orders = wvlsol_v0["orders"]
        # wvlsol = wvlsol_v0["wvl_sol"]

        orders = ad[0].WVLSOL0["orders"]
        wvlsol = ad[0].WVLSOL0["wavelengths"]


        # ref_lines_db = SkyLinesDB(config=obsset.get_config())
        ref_file = get_ref_line_path() # "ref_lines_oh.fits"
        ref_lines_db = SkyLinesDB(ref_file)

        ref_lines_db_hitrans = None
        # if obsset.rs.get_resource_spec()[1] == "K":
        #     ref_lines_db_hitrans = HitranSkyLinesDB(obsset.rs.master_ref_loader)
        # else:
        #     ref_lines_db_hitrans = None

        # keys = []
        fitted_pixels_list = []

        slit_centers = multi_spec["slit_centers"][0]
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

        return adinputs

    def volumeFit(self, adinputs, **params):

        # fn = "./SDCH_20190412_0040_wvl0.fits"
        # ad = astrodata.open(fn)
        ad = adinputs[0]

        linefit = ad[0].LINEFIT
        # d = obsset.load("SKY_FITTED_PIXELS_JSON")
        # .remove_column("params") # params column is a multi-d data,
        #                                       # not supported for conversion. We
        #                                       # just drop it.
        colnames = linefit.colnames

        df = linefit[colnames[:-1]].to_pandas()

        index_names = ["kind", "order", "wavelength"]
        df = df.set_index(index_names)[["slit_center", "pixels"]]

        dd = _append_offset(df)
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
        ad[0].VOLUMEFIT_COEFFS = Table.from_pandas(out_df)
        # d = out_df.to_dict(orient="split")
        # obsset.store("VOLUMEFIT_COEFFS_JSON", d)

        return adinputs

    def attachWatTable(self, adinputs, **params):

        ad = adinputs[0]
        fit_results_tbl = ad[0].WVLFIT_RESULTS

        cards = get_wat_cards(fit_results_tbl)
        tbl = Table([[c.image for c in cards]], names=["cards"])

        ad[0].WAT_HEADER = tbl

        return adinputs

    def makeSpectralMaps(self, adinputs, **params):

        ad = adinputs[0]

        ap = Apertures(ad[0].SLITEDGE)

        ordermap = ad[0].ORDERMAP = ap.make_order_map()
        slitposmap = ad[0].SLITPOSMAP = ap.make_slitpos_map()

        # # FIXME Do not remember why we needed this.
        # order_map2 = ap.make_order_map(mask_top_bottom=True)

        # We now make slitoffset map. It could be refactored to become a separate function.
        yy, xx = np.indices(ordermap.shape)

        msk = np.isfinite(ordermap) & (ordermap > 0)
        pixels, orders, slitpos = (xx[msk], ordermap[msk],
                                   slitposmap[msk])

        tbl = ad[0].VOLUMEFIT_COEFFS
        in_df = tbl.to_pandas() # pd.DataFrame(**d)

        names = ["pixel", "order", "slit"]

        # # pixel, order, slit : saved as float, needt to be int. Not needed for dragons.
        # for n in names:
        #     in_df[n] = in_df[n].astype("i")

        in_df = in_df.set_index(names)
        poly, coeffs = NdPolyNamed.from_pandas(in_df)

        cc0 = slitpos - 0.5
        values = dict(zip(names, [pixels, orders, cc0]))
        offsets = poly.multiply(values, coeffs) # * cc0

        offset_map = np.empty(ordermap.shape, dtype=np.float64)
        offset_map.fill(np.nan)
        offset_map[msk] = offsets * cc0 # dd["offsets"]

        ad[0].SLITOFFSETMAP = offset_map

        # Derive wavelength solution

        linefit = ad[0].LINEFIT
        colnames = [n for n in linefit.colnames if n != "params"]
        dfm = linefit[colnames].to_pandas().query("slit_center == 0.5")

        p, fit_results = fit_wvlsol(dfm)

        # from ..igrins_libs.resource_helper_igrins import ResourceHelper
        # helper = ResourceHelper(obsset)
        # orders = helper.get("orders")

        wvl_sol = _convert2wvlsol(p, ap.orders)

        ad[0].WVLSOL = Table([ap.orders, wvl_sol], names=["orders", "wavelengths"])

        fit_results["orders"] = ap.orders
        ad[0].WVLFIT_RESULTS = dict_to_table(fit_results)

        return adinputs

    def makeAB(self, adinputs, **params):
        adinputsA, adinputsB = splitAB(adinputs)
        stackedA = self.stackFrames(adinputsA)
        stackedB = self.stackFrames(adinputsB)

        ad = adinputs[0]
        ad_sky = self._get_ad_sky(ad)

        mask = ad_sky[0].ORDERMAP != 0

        data, var = subtract_ab(stackedA[0][0].data, stackedB[0][0].data,
                                stackedA[0][0].variance, stackedB[0][0].variance,
                                mask,
                                remove_level=params["remove_level"],
                                remove_amp_wise_var=params["remove_amp_wise_var"],
                                )

        # FIXME should we better to create a new instance of AstroData?
        ad = stackedA[0]
        ad[0].data = data
        ad[0].variance = var

        return [ad]

    def estimateSlitProfile(self, adinputs, **params):
        """
        return a profile function

        def profile(order, x_pixel, y_slit_pos):
            return profile_value

        """

        ad = adinputs[0]

        ad_flat = self._get_ad_flat(ad)
        ad_sky = self._get_ad_sky(ad)

        orderflat = ad_flat[0].data

        data_minus = ad[0].data
        data_minus_flattened = data_minus / orderflat

        ap = Apertures(ad_sky[0].SLITEDGE)
        # from .aperture_helper import get_aperture_from_obsset
        # orders = helper.get("orders")
        # ap = get_aperture_from_obsset(obsset, orders=orders)

        ordermap = ad_sky[0].ORDERMAP
        # ordermap_bpixed = helper.get("ordermap_bpixed")
        slitpos_map = ad_sky[0].SLITPOSMAP

        ordermap_bpixed = np.ma.array(ordermap, mask=ad_flat[0].mask > 0).filled(0)

        x1, x2 = params["slit_profile_range"]
        _ = extract_slit_profile(ap,
                                 ordermap_bpixed, slitpos_map,
                                 data_minus_flattened,
                                 x1=x1, x2=x2)
        bins, hh0, slit_profile_list = _

        if params["do_ab"]:
            profile_x, profile_y = _get_norm_profile_ab(bins, hh0)
            # profile = get_profile_func_ab(profile_x, profile_y)
        else:
            profile_x, profile_y = _get_norm_profile(bins, hh0)
            # profile = get_profile_func(profile_x, profile_y)

        slit_profile_dict = dict(orders=ap.orders,
                                 ab_mode=params["do_ab"],
                                 slit_profile_list=slit_profile_list,
                                 profile_x=profile_x,
                                 profile_y=profile_y)

        tbl = dict_to_table(slit_profile_dict)
        ad[0].SLITPROFILE = tbl

        profile = _get_profile_func_from_dict(slit_profile_dict)
        profile_map = make_slitprofile_map(ap, profile,
                                           ordermap, slitpos_map,
                                           frac_slit=params["frac_slit"])

        ad[0].SLITPROFILE_MAP = profile_map

        return adinputs

    def _get_spec1d(self, ad, ad_sky):

        ad_out = astrodata.create(ad.phu)

        # from astrodata.nddata import NDAstroData as NDDataObject

        # from astropy.table import Table
        fit_results_tbl = ad_sky[0].WVLFIT_RESULTS
        cards = get_wat_cards(fit_results_tbl)
        # tbl = Table([[c.image for c in cards]], names=["cards"])

        header = fits.Header(cards)

        hdu = fits.ImageHDU(header=header,
                            data=np.array(ad[0].SPEC1D["spec"]))
        ad_out.append(hdu)

        ad_out[0].variance = np.array(ad[0].SPEC1D["variance"])
        ad_out[0].WAVELENGTHS = np.array(ad[0].SPEC1D["wavelengths"])
        ad_out[0].SN_PER_RESEL = np.array(ad[0].SPEC1D["sn_per_res_element"])

        return ad_out

    def extractStellarSpec(self, adinputs, **params):
        # extraction_mode="optimal",
        #                      conserve_2d_flux=True,
        #                      pixel_per_res_element=None):

        extraction_mode = params["extraction_mode"]
        pixel_per_res_element = params["pixel_per_res_element"]

        ad = adinputs[0]

        ad_flat = self._get_ad_flat(ad)
        ad_sky = self._get_ad_sky(ad)

        ap = Apertures(ad_sky[0].SLITEDGE)

        orderflat = ad_flat[0].data
        data_minus = ad[0].data
        data_minus_flattened = data_minus / orderflat

        # if False:
        #     variance_map = obsset.load_fits_sci_hdu("combined_variance1",
        #                                             postfix=postfix).data
        #     variance_map0 = obsset.load_fits_sci_hdu("combined_variance0",
        #                                              postfix=postfix).data

        variance_map = ad[0].variance + 2**2 # FIXME figure out the readout
                                             # noize and properly update it
        variance_map0 = None # FIXME original plp used this to update variance
                             # while doing the iterationin optima extraction. We
                             # simply ignore this by setting it to NaN.

        ordermap = ad_sky[0].ORDERMAP
        # ordermap_bpixed = helper.get("ordermap_bpixed")
        slitpos_map = ad_sky[0].SLITPOSMAP

        ordermap_bpixed = np.ma.array(ordermap, mask=ad_flat[0].mask > 0).filled(0)

        slitoffset_map = ad_sky[0].SLITOFFSETMAP
        # slitoffset_map = helper.get("slitoffsetmap")

        # ordermap = helper.get("ordermap")
        # ordermap_bpixed = helper.get("ordermap_bpixed")
        # slitpos_map = helper.get("slitposmap")

        # gain = float(obsset.rs.query_ref_value("gain"))
        gain = 1.

        profile_map = ad[0].SLITPROFILE_MAP

        # profile_map = obsset.load_fits_sci_hdu("slitprofile_fits",
        #                                        postfix=postfix).data

        _ = extract_spec_using_profile(ap, profile_map,
                                       variance_map,
                                       variance_map0,
                                       data_minus_flattened,
                                       orderflat,
                                       ordermap, ordermap_bpixed,
                                       slitpos_map,
                                       slitoffset_map,
                                       gain,
                                       extraction_mode=extraction_mode,
                                       debug=False)

        s_list, v_list, cr_mask, aux_images = _

        wvl_solutions_map = dict(zip(ad_sky[0].WVLSOL["orders"], ad_sky[0].WVLSOL["wavelengths"]))

        wvl_solutions = []
        sn_list = []
        for o, s, v in zip(ap.orders_to_extract,
                           s_list, v_list):
            wvl = wvl_solutions_map[o]
            # if pixel_per_res_element is None:
            if pixel_per_res_element == 0.:
                dw = np.gradient(wvl)
                _pixel_per_res_element = (wvl/40000.)/dw
            else:
                _pixel_per_res_element = float(pixel_per_res_element)

            # print pixel_per_res_element[1024]
            # len(pixel_per_res_element) = 2047. But we ignore it.

            with np.errstate(invalid="ignore"):
                sn = (s/v**.5)*(_pixel_per_res_element**.5)

            sn_list.append(sn)
            wvl_solutions.append(wvl)

        from astropy.table import Table
        tbl = Table([ap.orders_to_extract, wvl_solutions, s_list, v_list, sn_list],
                    names=["orders", "wavelengths", "spec", "variance", "sn_per_res_element"])

        ad[0].SPEC1D = tbl

        shifted = aux_images["shifted"]
        tbl = Table([shifted._fields, list(shifted)],
                    names=["type", "array"])

        ad[0].WVLCOR = tbl

        ad_1dspec = self._get_spec1d(ad, ad_sky)
        ad_1dspec.update_filename(suffix="_spec1d", strip=True)

        self.streams["debug"] = adinputs

        return [ad_1dspec]

    def checkCALDB(self, adinputs, **params):
        for caltype in params["caltypes"]:
            calibrations = self.caldb.get_calibrations(adinputs, caltype)
            if calibrations.files[0] is None:
                raise RuntimeError(f"calibration file of {caltype} need to be specified")

        return adinputs

    def fixHeader(self, adinputs, **params):
        for ad in adinputs:
            forced_tags = set(ad.phu.get("TAG_FORCED", "").split())
            for tag in params["tags"]:
                if tag not in forced_tags:
                    forced_tags.add(tag)
            ad.phu["TAG_FORCED"] = " ".join(forced_tags)

            gt.mark_history(ad, primname=self.myself(), keyword="fixHeader")
            ad.update_filename(suffix=params['suffix'], strip=False)

        return adinputs

    def saveTwodspec(self, adinputs, **params):

        height_2dspec = params["height_2dspec"]
        conserve_flux = True
        # height_2dspec = 100 # obsset.get_recipe_parameter("height_2dspec")
        wavelength_increasing_order = params["wavelength_increasing_order"]

        ad = self.streams["debug"][0]

        ad_sky = self._get_ad_sky(ad)

        shifted = ShiftedImages.from_table(ad[0].WVLCOR)
        data_shft = shifted.image
        variance_map_shft = shifted.variance

        wat_table = ad_sky[0].WAT_HEADER

        # make sure you apply convert_data to the output. If get_wat_header is
        # called with wavelength_increasing_order=True, convert_data will rearrange
        # the data to the correct order.
        wvl_header, convert_data = get_wat_header(wat_table,
                                                  wavelength_increasing_order)

        ordermap = ad_sky[0].ORDERMAP
        # FIXME we should use proper badpixel mask.
        ordermap_bpixed = np.ma.array(ordermap, mask=ad_sky[0].mask).filled(0)

        ap = Apertures(ad_sky[0].SLITEDGE)

        order_map = ad_sky[0].ORDERMAP

        _ = get_rectified_2dspec(data_shft, ordermap_bpixed, ap, # bottom_up_solutions,
                                 conserve_flux=conserve_flux, height=height_2dspec)


        d0_shft_list, msk_shft_list, height = _

        with np.errstate(invalid="ignore"):
            d = np.array(d0_shft_list) / np.array(msk_shft_list)

        d = convert_data(d.astype("float32"))

        hdu_spec2d = fits.ImageHDU(header=wvl_header, data=d)

        ad_out = astrodata.create(ad.phu)
        ad_out.append(hdu_spec2d)

        _ = get_rectified_2dspec(variance_map_shft, order_map, ap, # bottom_up_solutions,
                                 conserve_flux=conserve_flux, height=height)

        d0_shft_list, msk_shft_list, _ = _

        with np.errstate(invalid="ignore"):
            d = np.array(d0_shft_list) / np.array(msk_shft_list)

        ad_out[0].variance = d

        ad_out[0].WAVELENGTHS = np.array(ad_sky[0].WVLSOL["wavelengths"])

        ad_out.update_filename(suffix="_spec2d", strip=True)
        ad_out.write(overwrite=True)

        return adinputs

    def saveDebugImage(self, adinputs, **params):
        if params["save_debug"]:
            ad_debug = self.streams["debug"]
            ad_debug[0].update_filename(suffix="_spec_debug", strip=True)
            ad_debug[0].write(overwrite=True)

        return adinputs

    # For FLAT recipes : make_hotpix_mask, make_deadpix_mask

    def makeIgrinsBPM(self, adinputs, **params):
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        sigma_clip1 = params['hotpix_sigma_clip1']
        sigma_clip2 = params['hotpix_sigma_clip2']
        deadpix_thresh = params['deadpix_thresh']
        smooth_size = params['deadpix_smooth_size']

        ad_flatoff = self.streams['flat-off'][0]
        ad_flaton = self.streams['flat-on'][0]

        for flatoff_ext, flaton_ext in zip(ad_flatoff, ad_flaton):
            flat_off = flatoff_ext.data

            bg_std, hotpix_mask = make_igrins_hotpixel_mask(flat_off,
                                                            sigma_clip1=sigma_clip1,
                                                            sigma_clip2=sigma_clip2,
                                                            medfilter_size=None)

            flat_on = flaton_ext.data
            flat_std = flaton_ext.variance**.5

            deadpix_mask = make_igrins_deadpixel_mask(flat_on, flat_std, deadpix_thresh, smooth_size)


            flatoff_ext.reset((hotpix_mask | deadpix_mask).astype(np.int16), mask=None, variance=None)

        ad_flatoff.update_filename(suffix="_badpixel", strip=True)
        ad_flatoff.phu.set('OBJECT', 'BadPixel')

        return [ad_flatoff]
