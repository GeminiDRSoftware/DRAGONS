#
#                                                                  gemini_python
#
#                                                               primitives_qa.py
# ------------------------------------------------------------------------------
import numpy as np
import math
import operator
from copy import deepcopy
from functools import partial
from collections import namedtuple

from astropy.stats import sigma_clip
from scipy.special import j1

from gemini_instruments.gmos.pixel_functions import get_bias_level

from gempy.gemini import gemini_tools as gt
from gempy.gemini import qap_tools as qap
from gempy.utils import logutils

from .lookups import DQ_definitions as DQ
from .lookups import qa_constraints as qa

from geminidr import PrimitivesBASE
from . import parameters_qa

from recipe_system.utils.decorators import parameter_override

QAstatus = namedtuple('QAstatus', 'band req warning info')
Measurement = namedtuple('Measurement', 'value std samples')
# ------------------------------------------------------------------------------
@parameter_override
class QA(PrimitivesBASE):
    """
    This is the class containing the QA primitives.
    """
    tagset = {"GEMINI"}

    def __init__(self, adinputs, **kwargs):
        super().__init__(adinputs, **kwargs)
        self._param_update(parameters_qa)

    def measureBG(self, adinputs=None, **params):
        """
        This primitive measures the sky background level for an image by
        sampling the non-object unflagged pixels in each extension.

        The count levels are then converted to a flux using the nominal
        (*not* measured) Zeropoint values - the point being you want to measure
        the actual background level, not the flux incident on the top of the
        cloud layer necessary to produce that flux level.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        remove_bias: bool
            remove the bias level (if present) before measuring background?
        separate_ext: bool
            report one value per extension, instead of a global value?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        suffix = params["suffix"]
        remove_bias = params.get("remove_bias", False)
        separate_ext = params["separate_ext"]

        for ad in adinputs:
            bias_level = None
            # First check if the bias level has already been subtracted
            if remove_bias:
                if not {'BIASIM', 'DARKIM',
                   self.timestamp_keys['subtractOverscan']}.intersection(ad.phu):
                    try:
                        bias_level = get_bias_level(adinput=ad, estimate=False)
                    except NotImplementedError:
                        bias_level = None

                    if bias_level is None:
                        log.warning("Bias level not found for {}; "
                                    "approximate bias will not be removed "
                                    "from the sky level".format(ad.filename))
                        bias_level = [None] * len(ad)

            # Get the filter name and the corresponding BG band definition
            # and the requested band
            filter = ad.filter_name(pretty=True)
            if filter in ['k(short)', 'kshort', 'K(short)', 'Kshort']:
                filter = 'Ks'
            try:
                bg_band_limits = qa.bgBands[filter]
            except KeyError:
                bg_band_limits = None

            report = qap.BGReport(ad, log=self.log, limit_dict=bg_band_limits)
            bunit = report.bunit

            for ext, bias in zip(ad, bias_level):
                report.measure(ext, bias_level=bias)
                results = report.calculate_metric()

                # Write sky background to science header
                bg = results['bg']
                if bg is not None:
                    ext.hdr.set("SKYLEVEL", bg,
                            comment=f"{self.keyword_comments['SKYLEVEL']} [{bunit}]")

                if separate_ext:
                    report.report(results,
                                  header=f"{ad.filename}:{ext.hdr['EXTVER']}")

            # Collapse extension-by-extension numbers if multiple extensions
            if len(ad) > 1:
                results = report.calculate_metric('all')

            if not separate_ext:
                report.report(results)

            # Write mean background to PHU if averaging all together
            # (or if there's only one science extension)
            if (len(ad) == 1 or not separate_ext) and results.get('bg') is not None:
                ad.phu.set("SKYLEVEL", results['bg'], comment="{} [{}]".
                            format(self.keyword_comments['SKYLEVEL'], bunit))

            # Report measurement to the adcc
            if results.get('mag'):
                qad = {"band": report.band,
                       "brightness": results["mag"],
                       "brightness_error": results["mag_std"],
                       "requested": report.reqband,
                       "comment": report.comments}
                qap.adcc_report(ad, "bg", qad)

            # Report measurement to fitsstore
            if self.upload and "metrics" in self.upload:
                qap.fitsstore_report(ad, "sb", report.info_list(),
                                     self.calurl_dict, self.mode, upload=True)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def measureCC(self, adinputs=None, suffix=None):
        """
        This primitive will determine the zeropoint by looking at sources in
        the OBJCAT for which a reference catalog magnitude has been determined

        It will also compare the measured zeropoint against the nominal
        zeropoint for the instrument and the nominal atmospheric extinction
        as a function of airmass, to compute the estimated cloud attenuation.

        This function is for use with SExtractor-style source-detection.
        It relies on having already added a reference catalog and done the
        cross match to populate the refmag column of the objcat

        The reference magnitudes (refmag) are straight from the reference
        catalog. The measured magnitudes (mags) are straight from the object
        detection catalog.

        We correct for atmospheric extinction at the point where we
        calculate the zeropoint, ie we define::

            actual_mag = zeropoint + instrumental_mag + extinction_correction

        where in this case, actual_mag is the refmag, instrumental_mag is
        the mag from the objcat, and we use the nominal extinction value as
        we don't have a measured one at this point. ie  we're actually
        computing zeropoint as::

            zeropoint = refmag - mag - nominal_extinction_correction

        Then we can treat zeropoint as::

            zeropoint = nominal_photometric_zeropoint - cloud_extinction

        to estimate the cloud extinction.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            nom_phot_zpt = ad.nominal_photometric_zeropoint()
            if not any(nom_phot_zpt):
                log.warning("No nominal photometric zeropoint available "
                            "for {}, filter {}".format(ad.filename,
                             ad.filter_name(pretty=True)))
                continue

            qad = {'zeropoint': {}}
            if not any(hasattr(ext, 'OBJCAT') for ext in ad):
                log.warning("No OBJCATs found in {}".format(ad.filename))
                continue

            # We really want to check for the presence of reference mags
            # in the objcats at this point, but we can more easily do a
            # quick check for the presence of reference catalogs, which are
            # a pre-requisite for this and not bother with
            # any of this if there are no reference catalogs
            if not hasattr(ad, 'REFCAT'):
                log.warning("No REFCAT present - not attempting"
                            " to measure photometric zeropoints")
                continue

            nom_at_ext = ad.nominal_atmospheric_extinction()
            if nom_at_ext is None:
                log.warning("Cannot get atmospheric extinction. Assuming zero.")
                nom_at_ext = 0.0
            exptime = ad.exposure_time()

            # If it's a funky nod-and-shuffle imaging acquistion,
            # then need to scale exposure time
            if "NODANDSHUFFLE" in ad.tags:
                log.warning("Imaging Nod-And-Shuffle. Photometry may be dubious")
                # AFAIK the number of nod_cycles isn't actually relevant -
                # there's always 2 nod positions, thus the exposure
                # time for any given star is half the total
                exptime /= 2.0

            all_zp = []
            all_cloud = []
            info_list = []

            for ext, npz in zip(ad, nom_phot_zpt):
                extver = ext.hdr['EXTVER']
                ext_info = {}
                try:
                    objcat = ext.OBJCAT
                except AttributeError:
                    log.warning("No OBJCAT in {}:{}".format(ad.filename,extver))
                    all_zp.append(Measurement(None, None, 0))
                    continue

                # Incrementally cull the catalog: remove sources without mags
                good_obj = objcat[~np.logical_or(objcat['MAG_AUTO'] == -999,
                                                 objcat['MAG_AUTO'] > 90)]
                if len(good_obj) == 0:
                    log.warning("No magnitudes found in {}[OBJCAT,{}]".format(
                                ad.filename,extver))
                    all_zp.append(Measurement(None, None, 0))
                    continue

                # Remove sources without reference mags
                good_obj = good_obj[~np.logical_or.reduce(
                    [good_obj['REF_MAG'] == -999, np.isnan(good_obj['REF_MAG']),
                     np.isnan(good_obj['REF_MAG_ERR'])])]
                if len(good_obj) == 0:
                    log.warning("No reference magnitudes found in {}[OBJCAT,{}]".
                        format(ad.filename,extver))
                    all_zp.append(Measurement(None, None, 0))
                    continue

                # Sources must be free of SExtractor flags and unsaturated, and
                # <2% of pixels be otherwise flagged (typically bad/non-linear)
                good_obj = good_obj[np.logical_and.reduce([good_obj['FLAGS'] == 0,
                        good_obj['NIMAFLAGS_ISO'] < 0.02*good_obj['ISOAREA_IMAGE'],
                        good_obj['IMAFLAGS_ISO'] & DQ.saturated == 0])]

                zps = good_obj['REF_MAG'] - nom_at_ext - (good_obj['MAG_AUTO'] +
                                                         2.5*math.log10(exptime))
                zperrs = np.sqrt(good_obj['REF_MAG_ERR']**2 +
                                 good_obj['MAGERR_AUTO']**2)

                # There shouldn't be any NaN left
                assert sum(np.logical_or(np.isnan(zps), np.isnan(zperrs))) == 0

                # TODO: weight instead?
                # Trim out where zeropoint error > err_threshold
                if len([z for z in zps if z is not None]) <= 5:
                    # 5 sources or less. Beggars are not choosers.
                    ok = zperrs<0.2
                else:
                    ok = zperrs<0.1

                # Ensure these are regular floats for JSON (thanks to PH)
                zps = [Measurement(float(zp), float(zperr), 1) for zp, zperr
                       in zip(zps[ok], zperrs[ok])]

                if len(zps) == 0:
                    log.warning("No good photometric sources found in "
                                "{}[OBJCAT,{}]".format(ad.filename,extver))
                    all_zp.append(Measurement(None, None, 0))
                    continue

                # Collapse all the Measurements to a single value + error
                if len(zps) > 2:
                    # TODO: 1-sigma clip is crap!
                    stats = _stats(zps)
                    m, s = stats.value, stats.std
                    zps = [z for z in zps if abs(z.value - m) < s]

                ext_zp = _stats(zps, weights='variance') if len(zps)>1 else zps[0]

                # Write the zeropoint to the SCI extension header
                ext.hdr.set("MEANZP", ext_zp.value, self.keyword_comments["MEANZP"])

                # Report average extinction measurement
                ext_cloud = _arith(_arith(ext_zp, 'sub', npz), 'mul', -1)
                comments = _cc_report(ext, ext_zp, ext_cloud, None)

                # Individual extinction measurements for all sources
                all_cloud.extend([_arith(_arith(zp, 'sub', npz), 'mul', -1)
                                   for zp in zps])
                all_zp.append(ext_zp)

                # Store the number in the QA dictionary to report to the RC
                ampname = ext.hdr.get("AMPNAME", 'amp{}'.format(extver))
                qad['zeropoint'].update({ampname: {'value': ext_zp.value,
                                                   'error': ext_zp.std}})

                # Compose a dictionary in the format the fitsstore record wants
                ext_info.update({"mag": ext_zp.value, "mag_std": ext_zp.std,
                        "cloud": ext_cloud.value, "cloud_std": ext_cloud.std,
                        "nsamples": ext_zp.samples})
                info_list.append(ext_info)

            # Only if we've managed to measure at least one zeropoint
            if any(zp.value for zp in all_zp):
                avg_cloud = _stats(all_cloud, weights='variance')
                qastatus = _get_qa_band('cc', ad, avg_cloud, qa.ccBands, simple=False)

                comments = _cc_report(ad, all_zp, avg_cloud, qastatus)

                # For QA dictionary
                qad.update({'band': qastatus.band, 'comment': comments,
                            'extinction': float(avg_cloud.value),
                            'extinction_error': float(avg_cloud.std)})
                qap.adcc_report(ad, "cc", qad)

                # Add band and comment to the info_list
                [info.update({"percentile_band": qad["band"],
                              "comment": qad["comment"]}) for info in info_list]

                # Also report to fitsstore
                if self.upload and "metrics" in self.upload:
                    fitsdict = qap.fitsstore_report(ad, "zp", info_list,
                                                    self.calurl_dict, self.mode,
                                                    upload=True)
            else:
                log.stdinfo("    Filename: {}".format(ad.filename))
                log.stdinfo("    Could not measure zeropoint - no catalog sources associated")

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def measureIQ(self, adinputs=None, **params):
        """
        This primitive is for use with sextractor-style source-detection.
        FWHM (from _profile_sources()) and CLASS_STAR (from SExtractor)
        are already in OBJCAT; this function does the clipping and reporting
        only. Measured FWHM is converted to zenith using airmass^(-0.6).

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        remove_bias: bool [only for some instruments]
            remove the bias level (if present) before displaying?
        separate_ext: bool
            report one value per extension, instead of a global value?
        display: bool
            display the images?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        suffix = params["suffix"]
        separate_ext = params["separate_ext"]
        display = params["display"]

        # remove_bias doesn't always exist in display() (only for GMOS)
        display_params = {"tile": not separate_ext}
        try:
            remove_bias = params["remove_bias"]
        except KeyError:
            remove_bias = False
        else:
            display_params["remove_bias"] = remove_bias

        frame = 1
        for ad in adinputs:
            iq_overlays = []
            is_ao = ad.is_ao()
            try:
                wvband = 'AO' if is_ao else ad.wavelength_band()
                iq_band_limits = qa.iqBands[wvband]
            except KeyError:
                iq_band_limits = None
            report = qap.IQReport(ad, log=self.log, limit_dict=iq_band_limits)

            if report.measure:
                has_sources = False
                for ext in ad:
                    extid = f"{ad.filename}:{ext.hdr['EXTVER']}"
                    nsources = report.add_measurement(ext, strehl_fn=partial(_strehl, ext))
                    if nsources == 0:
                        if separate_ext:
                            self.log.warning(f"No good sources found in {extid}")
                        continue

                    has_sources = True
                    results = report.calculate_metric()
                    if separate_ext:
                        report.report(results, header=extid)
                        if not is_ao:
                            fwhm, ellip = results["fwhm"], results["elip"]
                            if fwhm:
                                ext.hdr.set("MEANFWHM", fwhm,
                                            comment=self.keyword_comments["MEANFWHM"])
                            if ellip:
                                ext.hdr.set("MEANELLP", ellip,
                                            comment=self.keyword_comments["MEANELLP"])

                    # If displaying, make a mask to display along with image
                    # that marks which stars were used
                    if display:
                        iq_overlays.append(_iq_overlay(src, ext.data.shape))

                # Need one of these in order to make a report
                if has_sources or report.ao_seeing:
                    if len(ad) > 1:
                        results = report.calculate_metric('all')

                    if not separate_ext:
                        report.report(results)

                    qad = {"band": report.band, "requested": report.reqband,
                           "delivered": results["fwhm"],
                           "delivered_error": results["fwhm_std"],
                           "ellipticity": results["elip"],
                           "ellip_error": results["elip_std"],
                           "zenith": results["zfwhm"],
                           "zenith_error": results["zfwhm_std"],
                           "is_ao": is_ao, "ao_seeing": results["ao_seeing"],
                           "strehl": results["strehl"],
                           "comment": report.comments}
                    qap.adcc_report(ad, "iq", qad)

                    # Report measurement to fitsstore
                    if self.upload and "metrics" in self.upload:
                        qap.fitsstore_report(ad, "iq", report.info_list(),
                                             self.calurl_dict, self.mode, upload=True)

                    # Store measurements in the PHU if desired
                    if (len(ad) == 1 or not separate_ext) and not is_ao:
                        fwhm, ellip = results["fwhm"], results["elip"]
                        if fwhm:
                            ext.hdr.set("MEANFWHM", fwhm,
                                        comment=self.keyword_comments["MEANFWHM"])
                        if ellip:
                            ext.hdr.set("MEANELLP", ellip,
                                        comment=self.keyword_comments["MEANELLP"])

            if display:

                # We may need to tile the image (and OBJCATs) so make an
                # adiq object for such purposes
                if not separate_ext and len(ad) > 1:
                    adiq = deepcopy(ad)
                    if remove_bias and display:
                        # Set the remove_bias parameter to False so it doesn't
                        # get removed again when display is run; leave it at
                        # default if no tiling is being done at this point,
                        # so the display will handle it later
                        remove_bias = False

                        if (ad.phu.get('BIASIM') or ad.phu.get('DARKIM') or
                                any(v is not None for v in ad.hdr.get('OVERSCAN'))):
                            log.fullinfo("Bias level has already been "
                                         "removed from data; no approximate "
                                         "correction will be performed")
                        else:
                            try:
                                # Get the bias level
                                bias_level = get_bias_level(adinput=ad,
                                                            estimate=False)
                            except NotImplementedError:
                                bias_level = None

                            if bias_level is None:
                                log.warning("Bias level not found for {}; "
                                            "approximate bias will not be removed "
                                            "from the sky level".format(ad.filename))
                            else:
                                # Subtract the bias level from each extension
                                log.stdinfo("Subtracting approximate bias level "
                                            "from {} for display".format(ad.filename))
                                log.stdinfo(" ")
                                log.fullinfo("Bias levels used: {}".
                                             format(bias_level))
                                for ext, bias in zip(adiq, bias_level):
                                    ext.subtract(np.float32(bias))

                    log.fullinfo("Tiling extensions together in order to compile "
                                 "IQ data from all extensions")
                    adiq = self.tileArrays([adiq], tile_all=True)[0]
                else:
                    # No further manipulation, so can use a reference to the
                    # original AD object instead of making a copy
                    adiq = ad
                adiq = ad

                # If separate_ext is True, we want the tile parameter
                # for the display primitive to be False
                self.display([adiq], overlay=iq_overlays if iq_overlays else None,
                             frame=frame, **display_params)
                frame += len(adiq)
                if any(ov is not None for ov in iq_overlays):
                    log.stdinfo("Sources used to measure IQ are marked "
                                "with blue circles.")
                    log.stdinfo("")

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################
def _arith(m, op, operand):
    """Performs an arithmetic operation on a value and its uncertainty"""
    if op in ['mul', 'div', 'truediv']:
        return Measurement(getattr(operator, op)(m.value, operand),
                getattr(operator, op)(m.std, abs(operand)) if m.std else m.std,
                m.samples)
    else:
        return Measurement(getattr(operator, op)(m.value, operand),
                           m.std, m.samples)

def _stats(stats_list, weights='sample'):
    """
    Estimates overall mean and standard deviation from measurements that have
    already been compressed, so the original data don't exist

    Parameters
    ----------
    stats_list: list of Measurements
        The input statistics
    weights: 'variance'/'sample'/None
        how to weight the measurements

    Returns
    -------
    Measurement: mean, standard deviation, total number of measurements
    """
    try:
        use_list = [m for m in stats_list if m.value is not None]
        if weights == 'variance':
            wt = [1.0 / (m.std * m.std) for m in use_list]
        elif weights == 'sample':
            wt = [m.samples for m in use_list]
        else:
            wt = [1.0] * len(use_list)
        total_samples = sum(m.samples for m in use_list)
        mean = np.average([m.value for m in use_list], weights=wt)
        var1 = np.average([(m.value - mean)**2 for m in use_list],
                                   weights = wt)
        var2 = sum(w*m.std*m.std for w, m in zip(wt, use_list))/sum(wt)
        sigma = np.sqrt(var1 + var2)
    except:
        return Measurement(None, None, 0)
    return Measurement(mean, sigma, total_samples)

def _get_qa_band(metric, ad, quant, limit_dict, simple=True):
    """
    Calculates the QA band by comparing a measurement and its uncertainty with
    a dict of {value: limit} entries. This uses the dict to work out whether
    low numbers or high numbers are "good".

    Parameters
    ----------
    metric: str
        name of the metric
    ad: AstroData
        the AD object being investigated
    quant: Measurement
        value and uncertainty in the quantity measured
    limit_dict: dict
        dict of QA boundaries and values
    simple: bool
        do a simple test (ignoring uncertainty)? (otherwise hypothesis test)

    Returns
    -------
    QAstatus: band (int/list), reqband, warning (list), info (str)
        actual band(s), requested, warning comment/[], useful string for later
        presentation.

    """
    log = logutils.get_logger(__name__)

    # In cmp lambda fn, "-" is deprecated for in numpy objects. Because the
    # lambda needs to work on other types, and because we need a numerical
    # value returned, the bitwise-or recommendation does not apply.

    # We coerce the numpy.float64 object to native type @L785:: float(quant.value)
    cmp = lambda x, y: (x > y) - (x < y)
    try:
        reqband = getattr(ad, 'requested_{}'.format(metric.lower()))()
    except:
        reqband = None

    fmt1 = '95% confidence test indicates worse than CC{}'
    fmt2 = '95% confidence test indicates borderline CC{} or one band worse'
    fmt3 = '95% confidence test indicates CC{} or better'
    info = ''
    warning = ''

    if limit_dict is None:
        return QAstatus(None, reqband, warning, info)

    if quant is None or quant.value is None:
        qaband = None
    else:
        if simple:
            # Straightfoward determination of which band the measured value
            # lies in. The uncertainty on the measurement is ignored.
            bands, limits = list(zip(*sorted(limit_dict.items(),
                                             key=lambda k_v: k_v[0], reverse=True)))
            sign = cmp(limits[1], limits[0])
            inequality = '<' if sign > 0 else '>'
            qaband = 100
            info = '{}{}'.format(inequality, limits[0])
            for i in range(len(bands)):
                if cmp(float(quant.value), limits[i]) == sign:
                    qaband = bands[i]
                    info = '{}-{}'.format(*sorted(limits[i:i+2])) if \
                        i<len(bands)-1 else '{}{}'.format(
                        '<>'.replace(inequality,''), limits[i])
            if reqband is not None and qaband > reqband:
                warning = '{} requirement not met'.format(metric.upper())
        else:
            # Assumes the measured value and uncertainty represent a Normal
            # distribution, and works out the probability that the true value
            # lies in each band
            bands, limits = list(zip(*sorted(limit_dict.items(), key=lambda k_v: k_v[1])))
            bands = (100,)+bands if bands[0]>bands[1] else bands+(100,)
            # To Bayesian this, prepend (0,) to limits and not to probs
            # and renormalize (divide all by 1-probs[0]) and add prior
            norm_limits = [(l - float(quant.value))/quant.std for l in limits]
            cum_probs = [0] + [0.5*(1+math.erf(s/math.sqrt(2))) for
                           s in norm_limits] + [1]
            probs = np.diff(cum_probs)
            if bands[0] > bands[1]:
                bands = bands[::-1]
                probs = probs[::-1]
            qaband = [b for b, p in zip(bands, probs) if p>0.05]

            cum_prob = 0.0
            for b, p in zip(bands[:-1], probs[:-1]):
                cum_prob += p
                if cum_prob < 0.05:
                    log.fullinfo(fmt1.format(b))
                    if b == reqband:
                        warning = 'CC requirement not met at the 95% confidence level'
                elif cum_prob < 0.95:
                    log.fullinfo(fmt2.format(b))
                else:
                    log.fullinfo(fmt3.format(b))

    return QAstatus(qaband, reqband, warning, info)

def _cc_report(ad, zpt, cloud, qastatus):
    """
    Logs the formatted output of a measureCC report. Single-extension
    reports go to fullinfo, reports for an entire image to stdinfo.

    Parameters
    ----------
    ad: AstroData
        AD objects or slice
    zpt: Measurement/list
        zeropoint measurement(s) and uncertainty(ies)
    cloud: Measurement/list
        extinction measurement(s) and uncertainty(ies)
    qastatus: QAstatus namedtuple
        information about the actual band

    Returns
    -------
    list: list of comments to be passed to the FITSstore report
    """
    single_ext = ad.is_single
    comments = []
    headstr = 'Filename: {}'.format(ad.filename)
    if single_ext:
        headstr += ':{}'.format(ad.hdr['EXTVER'])
    header = [headstr]
    header.append('{} sources used to measure zeropoint'.format(cloud.samples))

    filt = ad.filter_name(pretty=True)
    if single_ext:
        # Never called on a single extension unless there's a measurement
        logtype = 'fullinfo'
        body = [('Zeropoint measurement ({}-band):'.format(filt),
                 '{:.2f} +/- {:.2f}'.format(zpt.value, zpt.std))]
        npz = cloud.value + zpt.value  # Rather than call descriptor again
        body.append(('Nominal zeropoint:', '{:.2f}'.format(npz)))
    else:
        logtype = 'stdinfo'
        for zp in zpt:
            rstr = '{:.2f} +/- {:.2f}'.format(zp.value, zp.std) if zp.value \
                else 'not measured'
            try:
                body.append(('', rstr))
            except NameError:
                body = [('Zeropoints by detector ({}-band):'.format(filt), rstr)]

    body.append(('Estimated cloud extinction:',
                 '{:.2f} +/- {:.2f} mag'.format(cloud.value, cloud.std)))

    if qastatus and not single_ext:
        if isinstance(qastatus.band, int):
            bands = [qastatus.band]
        else:
            bands = qastatus.band
        body.append(('CC bands consistent with this:', ', '.join(['CC{}'.
               format(x if x < 100 else 'Any') for x in bands])))
        if qastatus.req:
            body.append(('Requested CC:', 'CC{}'.format('Any' if
                                    qastatus.req==100 else qastatus.req)))
            if qastatus.warning:
                body.append(('WARNING: {}'.format(qastatus.warning), ''))
                comments.append(qastatus.warning)
        else:
            body.append(('(Requested CC could not be determined)', ''))

    _qa_report(header, body, 32, 26, logtype)
    return comments

def _qa_report(header, body, llen, rlen, logtype='stdinfo'):
    """
    Outputs a formatted QA report to the log.

    Parameters
    ----------
    header: list of str
        things to print in the header
    body: list of (str, str)
        things to print in the body
    llen: int
        width of left-justified part of body
    rlen: int
        width of right-justified part of body
    logtype: str
        how to log the report
    """
    log = logutils.get_logger(__name__)
    logit = getattr(log, logtype)
    indent = ' ' * logutils.SW
    logit('')
    for line in header:
        logit(indent + line)
    logit(indent + '-'*(llen+rlen))
    for lstr, rstr in body:
        if len(rstr) > rlen and not lstr:
            logit(indent + rstr.rjust(llen+rlen))
        else:
            logit(indent + lstr.ljust(llen) + rstr.rjust(rlen))
    logit(indent + '-'*(llen+rlen))
    logit('')
    return

def _gsaoi_iq_estimate(ad, fwhm, strehl):
    """
    Attempts to estimate the natural seeing for a GSAOI image from
    the observed FWHMs of objects.

    Parameters
    ----------
    ad: AstroData
        AD object being studied
    fwhm: Measurement
        measured FWHMs of stellar sources
    strehl: Measurement
        measured Strehl ratios of sources
    Returns
    -------
    Measurement: estimate of the seeing
    """
    log = logutils.get_logger(__name__)
    wavelength = ad.central_wavelength(asMicrometers=True)
    magic_number = np.log10(strehl.value * fwhm.value ** 1.5 /
                            wavelength ** 2.285)
    # Final constant is ln(10)
    magic_number_std = np.sqrt((strehl.std / strehl.value) ** 2 +
            (1.5 * fwhm.std / fwhm.value) ** 2 + 0.15 ** 2) / 2.3026
    if magic_number_std == 0.0:
        magic_number_std = 0.1
    if fwhm.value > 0.2:
        log.warning("Very poor image quality")
    elif abs((magic_number + 3.00) / magic_number_std) > 3:
        log.warning("Strehl and FWHM estimates are inconsistent")
    # More investigation required here
    return _arith(fwhm, 'mul', 7.0)

def _iq_overlay(stars, data_shape):
    """
    Generates a tuple of numpy arrays that can be used to mask a display with
    circles centered on the stars' positions and radii that reflect the
    measured FWHM.
    Eg. data[iqmask] = some_value

    The circle definition is based on numdisplay.overlay.circle, but circles
    are two pixels wide to make them easier to see.

    Parameters
    ----------
    stars: Table
        information (from OBJCAT) of the sources used for IQ measurement
    data_shape: 2-tuple
        shape of the data being displayed

    Returns
    -------
    tuple: arrays of x and y coordinates for overlay
    """
    xind = []
    yind = []
    width = data_shape[1]
    height = data_shape[0]
    for x0, y0 in zip(stars['x'], stars['y']):
        #radius = star["fwhm"]
        radius = 16
        r2 = radius*radius
        quarter = int(math.ceil(radius * math.sqrt (0.5)))

        for dy in range(-quarter,quarter+1):
            dx = math.sqrt(r2 - dy**2) if r2>dy*dy else 0
            j = int(round(dy+y0))
            i = int(round(x0-dx))           # left arc
            if i>=0 and j>=0 and i<width and j<height:
                xind.extend([i-1,i-2])
                yind.extend([j-1,j-1])
            i = int(round(x0+dx))           # right arc
            if i>=0 and j>=0 and i<width and j<height:
                xind.extend([i-1,i])
                yind.extend([j-1,j-1])

        for dx in range(-quarter, quarter+1):
            dy = math.sqrt(r2 - dx**2) if r2>dx*dx else 0
            i = int(round(dx + x0))
            j = int(round(y0 - dy))           # bottom arc
            if i>=0 and j>=0 and i<width and j<height:
                xind.extend([i-1,i-1])
                yind.extend([j-1,j-2])
            j = int (round (y0 + dy))           # top arc
            if i>=0 and j>=0 and i<width and j<height:
                xind.extend([i-1,i-1])
                yind.extend([j-1,j])

    iqmask = (np.array(yind),np.array(xind))
    return iqmask

def _strehl(ext, sources):
    """
    Calculate the Strehl ratio for each source in a list.

    Parameters
    ----------
    ext : AstroData slice
        image for which we're measuring the Strehl ratio
    sources: Table
        sources appropriate for measuring the Strehl ratio
    """
    wavelength = ext.effective_wavelength()
    # Leave if there's no wavelength information
    if wavelength == 0.0 or wavelength is None:
        return
    strehl_list = []
    pixel_scale = ext.pixel_scale()
    for star in sources:
        psf = _quick_psf(star['x'], star['y'], pixel_scale, wavelength,
                         8.1, 1.2)
        strehl = float(star['flux_max'] / star['flux'] / psf)
        strehl_list.append(strehl)
    return strehl_list


def _quick_psf(xc, yc, pixscale, wavelength, diameter, obsc_diam=0.0):
    """
    Calculate the peak pixel flux (normalized to total flux) for a perfect
    diffraction pattern due by a circular aperture of a given diameter
    with a central obscuration.

    Parameters
    ----------
    xc, yc:
        Pixel center (only subpixel location matters)
    pixscale:
        Pixel scale in arcseconds
    wavelength:
        Wavelength in metres
    diameter:
        Diameter of aperture in metres
    obsc_diam:
        Diameter of central obscuration in metres

    """
    xfrac = np.modf(float(xc))[0]
    yfrac = np.modf(float(yc))[0]
    if xfrac > 0.5:
        xfrac -= 1.0
    if yfrac > 0.5:
        yfrac -= 1.0
    # Accuracy improves with increased resolution, but subdiv=5
    # appears to give within 0.5% (always underestimated)
    subdiv = 5
    obsc = obsc_diam / diameter
    xgrid, ygrid = (np.mgrid[0:subdiv,0:subdiv]+0.5) / subdiv-0.5
    dr = np.sqrt((xfrac-xgrid)**2 + (yfrac-ygrid)**2)
    x = np.pi * diameter / wavelength * dr * pixscale / 206264.8
    sum = np.sum(np.where(x == 0, 1.0, 2*(j1(x) - obsc*j1(obsc*x))/x)**2)
    sum *= (pixscale/(206264.8 * subdiv) * (1 - obsc*obsc))**2
    return sum / (4 * (wavelength/diameter)**2 / np.pi)
