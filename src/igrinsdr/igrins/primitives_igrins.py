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

from .procedures.readout_pattern.readout_pattern_helper import (remove_readout_pattern_flat_off,
                                                                remove_readout_pattern_from_guard)

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

from .procedures.procedures_register import _get_offset_transform_between_two_specs
from .procedures.line_identify_simple import match_lines1_pix
from .procedures.identified_lines import IdentifiedLines
from .procedures.echellogram import Echellogram
from .procedures.fit_affine import fit_affine_clip
from .procedures.ecfit import get_ordered_line_data, fit_2dspec  # , check_fit

from .procedures.sky_spec_helper import _get_slices
from .procedures.process_wvlsol_volume_fit import (_append_offset,
                                                   _filter_points,
                                                   _volume_poly_fit)

from .procedures.nd_poly import NdPolyNamed
from .procedures.process_derive_wvlsol import fit_wvlsol, _convert2wvlsol

from .procedures.readout_pattern.readout_pattern_helper import remove_pattern


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

from .procedures.procedures_register import get_offset_transform_between_two_specs


def _get_wavelength_solutions(zdata, affine_tr_matrix,
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

def get_ref_path(band, kind):
    "returns a path-like object returned by importlib.resources.files"
    from . import lookups
    dataroot = files(lookups) / "ref_data"

    k = dict(
        ref_spec=dataroot / f"SDC{band}_20140525_0029.oned_spec.json",
        identified_lines_v0=dataroot / f"SKY_SDC{band}_20140525.identified_lines_v0.json",
        # echellogram_data=json.load((dataroot / f"SDC{band}_20140525.echellogram.json").open()),
        echellogram_data=dataroot / f"SDC{band}_20240721.echellogram.json",
        ref_lines_oh=dataroot / "ref_lines_oh.fits"
    )

    return k[kind]

def get_ref_data(band, kind):

    p = get_ref_path(band, kind)

    return json.load(p.open())


def get_ref_spectra(band, source="__package__"):
    # The default is to fetch the reference spectra that is included in the package.
    # We may support using a reduced sky spectra from the other night.

    if source == "__package__":
        # ref_spectra = get_ref_data(band)["ref_spec"]
        ref_spectra = get_ref_data(band, "ref_spec")

        orders_ref = ref_spectra["orders"]
        s_list_ref = ref_spectra["specs"]
    else:
        # using the output of sky recipe.
        ad = astrodata.open(source)
        spec1d = ad[0].SPEC1D

        orders_ref = spec1d["orders"]
        s_list_ref = spec1d["specs"]

    return orders_ref, s_list_ref


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


def get_xy_of_ref_n_tgt(df_identified_lines, tgt_ap: Apertures, ref_echellogram: Echellogram):
    df = df_identified_lines.loc[df_identified_lines["xpos"] >= 0]

    dfout = df.copy()
    for o, grouped in dfout.groupby("order"):
        dfout.loc[grouped.index, "ypos"] = tgt_ap.get_y(o, grouped["xpos"])

    # Using the wavelength, get the original x,y from the echellogram.
    for o, grouped in dfout.groupby("order"):
        xpos0, ypos0 = ref_echellogram.get_xy_from_wvl(o, grouped["wvl"])

        dfout.loc[grouped.index, "xpos0"] = xpos0
        dfout.loc[grouped.index, "ypos0"] = ypos0

    return dfout


def get_wvlsol_from_transformed_echellogram(echellogram: Echellogram, affine_tr_matrix,
                                            new_orders):
    """
    new_orders : output orders

    convert (x, y) of zdata (where x, y are pixel positions and z
    is wavelength) with affine transform, then derive a new wavelength
    solution.

    """

    # zdata is basically, x, y and wvl by o.

    from matplotlib.transforms import Affine2D
    affine_tr = Affine2D()
    affine_tr.set_matrix(affine_tr_matrix)

    df_echellogram = echellogram.get_df()

    transformed_x = affine_tr.transform(df_echellogram[["x", "y"]].values)[:, 0]

    p, m = fit_2dspec(transformed_x,
                      df_echellogram["order"].values,
                      df_echellogram["wvl"].values * df_echellogram["order"].values,
                      x_degree=4, y_degree=3,
                      x_domain=[0, 2047], y_domain=[new_orders[0], new_orders[-1]])

    xx, oo = np.meshgrid(np.arange(2048), new_orders)
    wvl_sol = p(xx, oo) / oo

    return wvl_sol


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

# for fitting the liness

import numpy as np
import pandas as pd

from numpy.typing import ArrayLike

def _gauss0_w_dcenters(xx, params, lines):
    """ Returns a gaussian function with the given parameters"""
    shift, sigma, height, background = params

    y_models = []
    with np.errstate(divide="ignore"):
        for line in lines:
            y_models.append(np.exp(-(((xx - (line + shift))/sigma)**2*0.5)))

    return height*np.array(y_models).sum(axis=0) + background


# def _gauss_w_dcenters_chi2(xx, yy, params, lines):
#     return np.sum((yy - _gauss0_w_dcenters(xx, params, lines))**2)


FIT_FAIL_RETURN_VALUE = [np.nan] * 4, None, None


def prepare_gaussian_group(x: np.ndarray, s: np.ndarray, lines: np.ndarray,
                           sigma_init=1.5,
                           max_sigma_scale=2,
                           fitrange_scale=2.5):
    """
    Prepare the fit that fits the spectrum with a group of gaussian that shares same offset and width.
    It returns sliced spectrum, fitting function, initial and bounds of the fit parameters.

    lines : initial x-coordinate of lines to fit
    sigma_init : initial sigma. A single value is given which will be shared with multiple lines. 
    """

    lines = np.array(lines)
    lines.sort()

    if not np.all(np.isfinite(lines)):  # if any of the position has nan
        return FIT_FAIL_RETURN_VALUE

    max_sigma = max_sigma_scale * sigma_init

    # The spectrum will be sliced and fit. We calculate the boundary of the slice.
    # The slice need to be larger than the bounds of x parameter.

    # The shift of x is bound to [-fitrange_scale, +fitrange_scale] * max_sigma.
    # The slice will be as large as this padded with addtional 2*max_sigma.

    xshift_max = fitrange_scale * max_sigma
    xmin = lines[0] - xshift_max - 2 * max_sigma
    xmax = lines[-1] + xshift_max + 2 * max_sigma

    # find the slice
    imin, imax = np.clip(np.searchsorted(x, [xmin, xmax]), 0, len(x))

    if imax - imin < 3:
        return FIT_FAIL_RETURN_VALUE

    sl = slice(imin, imax)

    xx = x[sl]
    yy = s[sl]

    # initial estimation of the height
    ymin, ymax = yy.min(), yy.max()
    yheight = ymax - ymin

    # dcenters0 = lines - lines[0]

    def _gauss(params, xx=xx, lines=lines):
        # return _gauss_w_dcenters_chi2(xx, yy, params, lines)
        return _gauss0_w_dcenters(xx, params, lines)

    # initial parameter and bounds
    params_ = [(0, (-xshift_max, xshift_max)),
               (sigma_init, (0, max_sigma)),
               (yheight, (0, 2*yheight)),
               (ymin, (ymin, ymax)) # baseline
               ]

    params0 = np.array([p for p, _ in params_])
    param_bounds = np.array([b for _, b in params_])

    return xx, yy, _gauss, params0, param_bounds


def fit_gaussian_group(x: np.ndarray, s: np.ndarray,
                       lines: np.ndarray,
                       sigma_init=1.5,
                       max_sigma_scale=2,
                       # drange_scale=5,
                       fitrange_scale=2.5):
    """
    Fit the spectrum with a group of gaussian that shares same offset and width.


    Parameters
    ----------
    x : x
    s : spectrum
    lines : initial x-coordinate of lines to fit
    sigma_init : initial sigma. A single value is given which will be
        shared with multiple lines. 

    Returns
    -------
    x : ndarray
        The solution. (offset, sigma, height, baseline)
    nfeval : int
        The number of function evaluations.
    rc : int
        Return code from the fitter, scipy.fmin_tnc


    """


    xx, yy, _gauss, params0, param_bounds = prepare_gaussian_group(x, s, lines,
                                                                   sigma_init=sigma_init,
                                                                   max_sigma_scale=max_sigma_scale,
                                                                   fitrange_scale=fitrange_scale)



    from scipy.optimize import fmin_tnc

    def chi2(params, gauss=_gauss, yy=yy):
        return np.sum((yy - gauss(params))**2)

    try:
        sol_ = fmin_tnc(chi2, params0,
                        bounds=param_bounds,
                        approx_grad=True, disp=0,
                        )
    except ValueError:
        raise

    return sol_




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

            mask = np.empty((2048, 2048), dtype=DQ.datatype)
            mask.fill(DQ.unilluminated)
            for o, sl, m in iter_order(pp):
                mask[sl][m] = 0

            ext.mask |= mask

        return adinputs

    def normalizeFlat(self, adinputs=None, **params):

        from .procedures.normalize_flat import (get_initial_spectrum_for_flaton,
                                                get_normalize_spectrum_for_flaton)

        ad = adinputs[0]

        for ext in ad:
            tbl = ext.SLITEDGE

            slitedge_polyfit = table_to_poly(tbl)

            ext.FLAT_ORIGINAL = ext.data.copy()

            # dq_mask = (ext.mask & DQ.unilluminated).astype(bool)
            # d[dq_mask] = np.nan

            d = ext.data
            mask = ext.mask > 0

            s = get_initial_spectrum_for_flaton(d, mask, slitedge_polyfit)
            s_list, i1i2_list, s2_list = get_normalize_spectrum_for_flaton(s)

            flat_im = np.ones(d.shape, "d")

            for (o, sl, m), s2 in zip(iter_order(slitedge_polyfit), s2_list):
                if s2 is None:  # some order may have little valid pixels and
                                # spectrum is None. We just skip these.
                    continue

                # subim = np.ma.array(d[sl], mask=~m).filled(np.nan)
                # d_div = subim / s2
                subim = d[sl]
                flat_im[sl][m] = (subim / s2)[m]

            with np.errstate(invalid="ignore"):
                flat_im[flat_im < 0.5] = np.nan

            ext.data = flat_im

            order_flat_dict = dict(#orders=orders,
                                   fitted_responses=s2_list,
                                   i1i2_list=i1i2_list,
                                   mean_order_specs=s)

            tbl = dict_to_table(order_flat_dict)

            ad.FLATNORM = tbl

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

    def readoutPatternCorrectSky(self, adinputs, **params):

        # We assume that ad instance has only a single extension.
        data_list = [ad[0].data for ad in adinputs]
        band = adinputs[0][0].band()
        assert band in "HK"

        data_list_fixed = remove_readout_pattern_flat_off(data_list, band=band,
                                                          rp_remove_mode=0)

        for ad, d in zip(adinputs, data_list_fixed):
            ad[0].data = d
            ad.update_filename(suffix="_rpc", strip=True)

        return adinputs


    def readoutPatternCorrectFlatOff(self, adinputs, **params):
        lamp_off_list = self.selectFromInputs(adinputs, tags='LAMPOFF')

        # We assume that ad instance has only a single extension.
        data_list = [ad[0].data for ad in lamp_off_list]
        band = lamp_off_list[0][0].band()
        assert band in "HK"

        flat_off_1st_pattern_removal_mode = params["flat_off_1st_pattern_removal_mode"]
        flat_off_2nd_pattern_removal_mode = params["flat_off_2nd_pattern_removal_mode"]
        if flat_off_2nd_pattern_removal_mode == "auto":
           rp_remove_mode = None
        else:
           rp_remove_mode = int(flat_off_2nd_pattern_removal_mode)


        data_list_fixed = remove_readout_pattern_flat_off(data_list, band=band,
                                                          flat_off_pattern_removal=flat_off_1st_pattern_removal_mode,
                                                          rp_remove_mode=rp_remove_mode)

        for ad, d in zip(lamp_off_list, data_list_fixed):
            ad[0].data = d
            ad.update_filename(suffix="rp_corrected", strip=True)

        return adinputs

    def readoutPatternCorrectFlatOn(self, adinputs, **params):

        if "IGRINS-2" in adinputs[0].tags:
            # do nothing for IGRINS-2. The reference pixel values in IGRINS-2
            # detectors can make things worse. FIXME The data taken after
            # 202410 can be okay. Need to check.
            return adinputs

        lamp_on_list = self.selectFromInputs(adinputs, tags='LAMPON')
        for ad in lamp_on_list:
            ad.data = remove_readout_pattern_from_guard(ad.data)

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

        # FIXME we simply apply mask from ad_flat. Maybe we should we have flatCorrect prmitive?
        d = np.ma.array(ad[0].data, mask=(ad[0].mask | ad_flat[0].mask) > 0).filled(np.nan)
        s = ap.extract_spectra_simple(d, f1=0.1, f2=0.9)

        # t = Table(s,
        #           names=(f"{o}" for o in ap.orders_to_extract))
        ss = [np.array(s1, dtype='float32') for s1 in s]
        t = Table([ap.orders_to_extract, ss], names=['orders_initial', 'specs'])

        ad[0].SPEC1D = t
        ad[0].SLITEDGE = tbl

        return adinputs


    def identifyOrders(self, adinputs):
        # given the extracted spectrum, we compare this with reference spectrum
        # to figure which aperture corresponds to which order. We basically
        # cross-correlat the spectrum of a given aperture with all the spectra
        # in the reference spectra, and found the order that gives a maximum
        # cross-correlation. To preven the spectrum to sensitive to the bright
        # lines, we filter the spectrum before the cross-correlation. The
        # filtering basically clips large values. For each aperture, what we
        # get is an delta order from the initial guess. The delta order is
        # estimated for about ~10 aperture near the center, and we delta order
        # with maximum occurence given that number of occurence is larger than
        # the threshold. As a result, we identify which aperture corresponds to
        # which order and how much shift need to be applied to the reference
        # spectra to match the given spectra. The shift is measure for the ~10 order
        # in the center, though.

        ad = adinputs[0]
        ext = ad[0]
        spec1d = ext.SPEC1D

        s_list_ = spec1d["specs"]
        s_list = [np.array(s, dtype=np.float64) for s in s_list_]

        band = ext.band() # phu["BAND"]

        orders_ref, s_list_ref = get_ref_spectra(band)

        # match the orders of s_list_src & s_list_dst
        new_orders, indx_shift_dict = match_orders(orders_ref, s_list_ref,
                                                   s_list)

        # indx in indx_shift_dict should be a real order
        spec1d["orders"] = new_orders
        # to make it a shift from the reference spectrum, we negate the shift.
        spec1d["shift_from_ref"] = [-indx_shift_dict.get(o, np.nan) for o in spec1d["orders"]]

        order_map = dict(zip(spec1d["orders_initial"], new_orders))
        ext.SLITEDGE["order"] = [order_map[o] for o in ext.SLITEDGE["order"]]

        # ad[0].SPEC1D_NEW = spec1d
        return adinputs


    def identifyLines(self, adinputs, **params):
        # Given the already identified line in position and wavelength per order,
        # we try to reidentify lines from the spectrum. To do this, we need to
        # provide an initial transform function that transform pixel axis in the
        # reference spectra to that of target spectrum. For now, we use
        # cross-corrlation of reference spectra to that of the target spectra.

        ad = adinputs[0]
        ext = ad[0]
        tgt_spec = ext.SPEC1D

        band = ext.band() # phu["BAND"]

        ref_spec = get_ref_data(band, "ref_spec")

        tr_ref_to_tgt = get_offset_transform_between_two_specs(ref_spec, tgt_spec)
        # a dictionary of transforms by orders.

        l = get_ref_data(band, "identified_lines_v0")
        identified_lines_ref = IdentifiedLines(l)

        identified_lines_tgt = identified_lines_ref.reidentify_specs(tgt_spec["orders"],
                                                                     tgt_spec["specs"],
                                                                     tr_ref_to_tgt)

        tbl = Table.from_pandas(identified_lines_tgt.get_df())
        ad[0].LINEID = tbl

        return adinputs

    def getInitialWvlsol(self, adinputs, **params):
        ad = adinputs[0]
        ext = ad[0]
        tgt_spec = ext.SPEC1D

        df_identified_lines = ad[0].LINEID.to_pandas()

        ap = Apertures(ad[0].SLITEDGE)

        band = ext.band() # phu["BAND"]

        echellogram_data = get_ref_data(band, "echellogram_data")
        echellogram = Echellogram.from_dict(echellogram_data)

        # We may use the xpos and ypos to fit the wavelength solution. However,
        # it is assumed that we do not have many lines to cover whole detector
        # area thus doing that will may give unstable wavelength solution.
        # Therefore, we fit affine transform from reference to the target,
        # transform each orders' x, y position of echellogram (length of 2048;
        # y position is not actually used), and then fit that to derive a new
        # wavelength solution.

        dfout = get_xy_of_ref_n_tgt(df_identified_lines, ap, echellogram)
        xy_list_ref = dfout[["xpos0", "ypos0"]].values # from reference echellogram
        xy_list_tgt = dfout[["xpos", "ypos"]].values # idntified from the target.

        # find the affine transform.
        affine_tr, mm = fit_affine_clip(xy_list_ref, xy_list_tgt)

        affine_tr_matrix = affine_tr.get_matrix()

        # we now transform the echellogram with the affine transform.

        orders = tgt_spec["orders"]
        wvl_sol = get_wvlsol_from_transformed_echellogram(echellogram,
                                                          affine_tr_matrix,
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
        from operator import itemgetter
        from scipy.interpolate import interp1d
        from igrinsdr.igrins.primitives_igrins import get_ref_path

        ad = adinputs[0]
        band = ad[0].band()

        # prepare line fitting. Reference lines are read and we will define
        # _fit function which will fit lines given a list of spectrum. _fit
        # will be applied to spectra of different slit positions.

        wvlsol0 = ad[0].WVLSOL0
        orders, wvlsol = wvlsol0["orders"], wvlsol0["wavelengths"]
        wvlsol_by_order = dict(zip(orders, wvlsol))

        ref_file = get_ref_path(band, "ref_lines_oh") # "ref_lines_oh.fits"
        tbl = Table.read(ref_file.open("rb"), format="fits") # "ref_lines_oh.fits"
        df_ref_data0 = tbl.to_pandas()

        from scipy.interpolate import interp1d
        x = np.arange(2048)
        # for each order, add pixel coordinate from the initial wvlsol
        for order, grouped in df_ref_data0.groupby("order"):
            wvl = wvlsol_by_order.get(order, None)
            if wvl is not None:
                knots = interp1d(wvl, x,
                                 bounds_error=False, assume_sorted=True, fill_value=np.nan)
                df_ref_data0.loc[grouped.index, "pixel"] = knots(grouped["um"])

        # flags groups that any of the line in the group has a pixel value of nan.
        msk = df_ref_data0.groupby("gid")["pixel"].apply(lambda pixels:
                                                        np.all(np.isfinite(pixels)))
        # msk has an index of "gid". We will filter the dataframe using this mask.
        # Note that there can be multiple rows wit same gid, and indexing with mask
        # gives a warning of

        # Boolean Series key will be reindexed to match DataFrame index

        # FIXME check if there is a better way of doing this.
        df_ref_data = df_ref_data0.set_index("gid")[msk].reset_index()

        # The filtered df_ref_data should only have valid pixels.


        def _fit(df_ref_data, spec_by_order):
            # we prepare a dataframe index of (order, gid)
            grouped = df_ref_data.groupby(["order", "gid"])
            df_fit = pd.DataFrame(dict(initial_mean_pixel=grouped["pixel"].mean(),
                                       wavelength=grouped["um"].mean()))

            fitted_line_location = pd.Series(index=df_ref_data.index) # initially set to nan

            # For each group, we fit the sliced data with multiple gaussian.
            for (o, gid), grp in grouped:
                if (s := spec_by_order.get(o, None)) is not None:
                    r = fit_gaussian_group(x, s, grp["pixel"])
                    # add column for the fit parameter
                    df_fit.loc[(o, gid), ["shift", "sigma", "height", "baseline"]] = r[0]
                    # add column for fitted pixel position
                    df_fit.loc[(o, gid), "fitted_pixel"] = df_fit.loc[(o, gid), "initial_mean_pixel"] + r[0][0]
                    fitted_line_location[grp.index] = grp["pixel"] + r[0][0]

            return df_fit, fitted_line_location

        # fit lines in the spectrum of the slit center
        multi_spec = ad[0].SPEC1D_MULTI
        slit_centers = multi_spec["slit_centers"][0].astype("float32")
        i_slit_center = len(slit_centers) // 2

        spec_data = multi_spec["multispec"][:, i_slit_center, :]
        spec_by_order = dict(zip(multi_spec["orders"], spec_data))

        df_fit_list = []
        df_fit, fitted_line_location0 = _fit(df_ref_data, spec_by_order)
        df_fit_list.append((slit_centers[i_slit_center], df_fit))

        # Now we do lower and upper part of the slit
        for i_range in [range(0, i_slit_center)[::-1], # lower part of the slit
                        range(i_slit_center+1, len(slit_centers)) # upper part of the slit
                        ]:
            # for the start of upper and lower parts, the initial location of
            # lines are from the slit center.
            fitted_line_location = fitted_line_location0
            for i in i_range:
                # we update the pixel location from the previous fit
                df_ref_data_updated = df_ref_data.copy(deep=False)
                df_ref_data_updated["pixel"] = fitted_line_location

                spec_data = multi_spec["multispec"][:, i, :]
                spec_by_order = dict(zip(multi_spec["orders"], spec_data))

                df_fit, fitted_line_location = _fit(df_ref_data, spec_by_order)
                df_fit_list.append((slit_centers[i], df_fit))


        df_fit_list.sort(key=itemgetter(0))
        df_fit_oh = pd.concat([df_fit for _, df_fit in df_fit_list],
                              keys=[c for c, _ in df_fit_list],
                              names=["slit_center"],
                              axis=0)

        df_fit_master = pd.concat([df_fit_oh],
                                  keys=["oh"],
                                  names=["kind"],
                                  axis=0)


        tbl = Table.from_pandas(df_fit_master.reset_index())

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
