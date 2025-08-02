#
#                                                                  gemini_python
#
#                                                               primitives_qa.py
# ------------------------------------------------------------------------------
import numpy as np
import math
from copy import deepcopy
from functools import partial

from scipy.special import j1
from astropy.table import Table, vstack
from astropy.stats import sigma_clip

from gemini_instruments.gmos.pixel_functions import get_bias_level

from gempy.gemini import gemini_tools as gt
from gempy.gemini import qap_tools as qap
from gempy.utils import logutils

from .lookups import qa_constraints as qa

from geminidr import PrimitivesBASE
from . import parameters_qa
from .lookups import DQ_definitions as DQ

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class QA(PrimitivesBASE):
    """
    This is the class containing the QA primitives.
    """
    tagset = {"GEMINI"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
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
        suffix : str
            suffix to be added to output files
        remove_bias : bool
            remove the bias level (if present) before measuring background?
        separate_ext : bool
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
            if bias_level is None:
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

            report = BGReport(ad, log=self.log, limit_dict=bg_band_limits)
            bunit = report.bunit

            for ext, bias in zip(ad, bias_level):
                report.add_measurement(ext, bias_level=bias)
                results = report.calculate_metric()

                # Write sky background to science header
                bg = results.get("bg")
                if bg is not None:
                    ext.hdr.set("SKYLEVEL", bg,
                            comment=f"{self.keyword_comments['SKYLEVEL']} [{bunit}]")

                if separate_ext:
                    report.report(results,
                                  header=f"{ad.filename} extension {ext.id}")

            # Collapse extension-by-extension numbers if multiple extensions
            if len(ad) > 1:
                report.reset_text()  # clear comments for final report
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
                                     self.mode, upload=True)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def measureCC(self, adinputs=None, **params):
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
        suffix : str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        suffix = params["suffix"]
        separate_ext = params["separate_ext"]

        for ad in adinputs:
            nom_phot_zpt = ad.nominal_photometric_zeropoint()
            if not any(nom_phot_zpt):
                log.warning("No nominal photometric zeropoint available "
                            "for {}, filter {}"
                            .format(ad.filename, ad.filter_name(pretty=True)))
                continue

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

            report = CCReport(ad, log=self.log, limit_dict=qa.ccBands)

            qad = {'zeropoint': {}}
            all_results = []
            for ext in ad:
                extid = f"{ad.filename} extension {ext.id}"
                nsources = report.add_measurement(ext)
                results = report.calculate_metric()
                if separate_ext and nsources == 0:
                    if hasattr(ext, 'OBJCAT'):
                        log.warning("No suitable object and/or reference "
                                    f"magnitudes found in {extid}")
                    else:
                        log.warning(f"No OBJCAT in {extid}")
                    continue

                all_results.append(results)
                if not results:
                    if separate_ext:
                        log.warning(f"No good photometric sources found in {extid}")
                    continue

                # Write the zeropoint to the extension header
                zpt = results.get("mag")
                if zpt is not None:
                    ext.hdr.set("MEANZP", zpt, self.keyword_comments["MEANZP"])

                if separate_ext:
                    report.report(results,
                                  header=f"{ad.filename} extension {ext.id}")

                # Store the number in the QA dictionary to report to the ADCC
                ampname = ext.hdr.get("AMPNAME", f"amp{ext.id}")
                qad['zeropoint'].update({ampname: {'value': results["mag"],
                                                   'error': results["mag_std"]}})

            # Collapse extension-by-extension numbers if multiple extensions
            if len(ad) > 1:
                report.reset_text()  # clear comments for final report
                results = report.calculate_metric('all')

            if results:
                if not separate_ext:
                    report.report(results, all_results=all_results)

                # For QA dictionary
                qad.update({'band': report.band, 'comment': report.comments,
                            'extinction': results["cloud"],
                            'extinction_error': results["cloud_std"]})
                qap.adcc_report(ad, "cc", qad)

                # Also report to fitsstore
                if self.upload and "metrics" in self.upload:
                    qap.fitsstore_report(ad, "zp", report.info_list(),
                                         self.mode, upload=True)
            else:
                log.stdinfo(f"    Filename: {ad.filename}")
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
        suffix : str
            suffix to be added to output files
        remove_bias : bool [only for some instruments]
            remove the bias level (if present) before displaying?
        separate_ext : bool
            report one value per extension, instead of a global value?
        display : bool
            display the images?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        suffix = params["suffix"]
        display = params["display"]

        # If separate_ext is True, we want the tile parameter
        # for the display primitive to be False
        #display_params = {"tile": not separate_ext}
        display_params = {}
        # remove_bias doesn't always exist in display() (only for GMOS)
        if "remove_bias" in params:
            display_params["remove_bias"] = params["remove_bias"]

        frame = 1
        for ad in adinputs:
            measure_iq = True
            iq_overlays = []
            is_ao = ad.is_ao()
            separate_ext = params["separate_ext"] & (len(ad) > 1)

            if not {'IMAGE', 'SPECT'} & ad.tags:
                log.warning(f"{ad.filename} is not IMAGE or SPECT; "
                            "no IQ measurement will be performed")
                measure_iq = False

            # Check that the data is not an image with non-square binning
            if 'IMAGE' in ad.tags:
                xbin = ad.detector_x_bin()
                ybin = ad.detector_y_bin()
                if xbin != ybin:
                    log.warning("No IQ measurement possible, image {} is {} x "
                                "{} binned data".format(ad.filename, xbin, ybin))
                    measure_iq = False

            # We may still display the image
            if measure_iq:
                try:
                    wvband = 'AO' if is_ao else ad.wavelength_band()
                    iq_band_limits = qa.iqBands[wvband]
                except KeyError:
                    iq_band_limits = None

                if separate_ext or len(ad) == 1:
                    adiq = ad
                else:
                    log.fullinfo(f"Tiling {ad.filename} to compile IQ data "
                                 "from all extensions")
                    adiq = self.tileArrays([deepcopy(ad)]).pop()

                report = IQReport(adiq, log=self.log, limit_dict=iq_band_limits)

                if {'GSAOI', 'IMAGE'}.issubset(ad.tags):
                    ao_seeing_fn = partial(_gsaoi_iq_estimate, ad)
                else:
                    ao_seeing_fn = None

                has_sources = False
                for ext in adiq:
                    extid = f"{ad.filename} extension {ext.id}" if separate_ext else ad.filename
                    nsources = report.add_measurement(ext, strehl_fn=partial(_strehl, ext))
                    results = report.calculate_metric(ao_seeing_fn=ao_seeing_fn)
                    if nsources == 0:
                        if separate_ext:
                            self.log.warning(f"No good sources found in {extid}")
                        continue
                    else:
                        has_sources = True

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

                # Need one of these in order to make a report
                if has_sources or report.ao_seeing:
                    if len(ad) > 1:
                        report.reset_text()  # clear comments for final report
                        results = report.calculate_metric('all', ao_seeing_fn=ao_seeing_fn)

                    if not separate_ext:
                        report.report(results)

                    qad = {"band": report.band, "requested": report.reqband,
                           "delivered": results.get("fwhm"),
                           "delivered_error": results.get("fwhm_std"),
                           "ellipticity": results.get("elip"),
                           "ellip_error": results.get("elip_std"),
                           "zenith": results.get("zfwhm"),
                           "zenith_error": results.get("zfwhm_std"),
                           "is_ao": is_ao, "ao_seeing": results.get("ao_seeing"),
                           "strehl": results.get("strehl"),
                           "comment": report.comments}
                    qap.adcc_report(ad, "iq", qad)

                    # Report measurement to fitsstore
                    if self.upload and "metrics" in self.upload:
                        qap.fitsstore_report(ad, "iq", report.info_list(),
                                             self.mode, upload=True)

                    # Store measurements in the PHU if desired
                    if (len(ad) == 1 or not separate_ext) and not is_ao:
                        fwhm, ellip = results["fwhm"], results["elip"]
                        if fwhm:
                           ad.phu.set("MEANFWHM", fwhm,
                                      comment=self.keyword_comments["MEANFWHM"])
                        if ellip:
                            ad.phu.set("MEANELLP", ellip,
                                       comment=self.keyword_comments["MEANELLP"])
                else:
                    self.log.warning(f"No good sources found in {ad.filename}")

            if display:
                # If displaying, make a mask to display along with image
                # that marks which stars were used
                for ext, measurement in zip(ad, report.measurements):
                    try:  # columns won't exist if no objects
                        circles = [(x, y, 16) for x, y in zip(measurement["x"], measurement["y"])]
                    except KeyError:
                        iq_overlays.append(None)
                    else:
                        iq_overlays.append(circles)

                self.display([adiq], debug_overlay=tuple(iq_overlays) if iq_overlays else None,
                             frame=frame, **display_params)
                frame += len(ad) if separate_ext else 1
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
def _gsaoi_iq_estimate(ad, results):
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
    try:
        strehl, fwhm = results["strehl"], results["fwhm"]
    except KeyError:  # no sources
        return

    log = logutils.get_logger(__name__)
    wavelength = ad.central_wavelength(asMicrometers=True)
    magic_number = np.log10(strehl * fwhm ** 1.5 / wavelength ** 2.285)
    # Final constant is ln(10)
    magic_number_std = np.sqrt((results["strehl_std"] / strehl) ** 2 +
            (1.5 * results["fwhm_std"] / fwhm) ** 2 + 0.15 ** 2) / 2.3026
    if magic_number_std == 0.0:
        magic_number_std = 0.1
    if fwhm > 0.2:
        log.warning("Very poor image quality")
    elif abs((magic_number + 3.00) / magic_number_std) > 3:
        log.warning("Strehl and FWHM estimates are inconsistent")
    # More investigation required here
    return fwhm * 7, results["fwhm_std"] * 7


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


class QAReport:
    result1 = '95% confidence test indicates worse than {}{}'
    result2 = '95% confidence test indicates borderline {}{} or one band worse'
    result3 = '95% confidence test indicates {}{} or better'

    def __init__(self, ad, log=None, limit_dict=None):
        self.log = log or logutils.get_logger(__name__)
        self.filename = ad.filename
        self.filter_name = ad.filter_name(pretty=True)
        self.ad_tags = ad.tags
        self.num_ext = len(ad)  # for info_list validation
        self.metric = self.__class__.__name__[:2]
        self.limit_dict = limit_dict
        self.band = None
        self.measurements = []
        self.results = []
        self.fitsdict_items = ["percentile_band", "comment"]
        self.reset_text()

    def reset_text(self):
        """Clear comments and warning"""
        self.comments = []
        self.warning = ''

    @staticmethod
    def bandstr(band):
        return 'Any' if band == 100 else str(band)

    def calculate_qa_band(self, value, unc=None, simple=True):
        """
        Calculates the QA band(s) corresponding to the measured value (and its
        uncertainty, if simple=False) and populates various attributes of the
        QAReport object with information.

        Parameters
        ----------
        value : float
            measured value for the metric
        unc : float
            uncertainty on the measurement (used only if simple=True)
        simple : bool
            use the value only? (else perform a Bayesian analysis)

        Updates
        -------
        self.band : int
            QA band achieved by this observation
        self.band_info: str
            information about the limit(s) of the achieved QA band
        self.warning : str
            if the requested QA band requirements have not been met
        """
        cmp = lambda x, y: (x > y) - (x < y)

        log = self.log
        if self.limit_dict is None or value is None:
            return
        simple |= unc is None

        if simple:
            # Straightfoward determination of which band the measured value
            # lies in. The uncertainty on the measurement is ignored.
            bands, limits = list(zip(*sorted(self.limit_dict.items(),
                                             key=lambda k_v: k_v[0], reverse=True)))
            sign = cmp(limits[1], limits[0])
            inequality = '<' if sign > 0 else '>'
            self.band = 100
            self.band_info = f'{inequality}{limits[0]}'
            for i in range(len(bands)):
                if cmp(float(value), limits[i]) == sign:
                    self.band = bands[i]
                    self.band_info = ('{}-{}'.format(*sorted(limits[i:i+2])) if
                                 i < len(bands)-1 else '{}{}'.format(
                                 '<>'.replace(inequality,''), limits[i]))
            if self.reqband is not None and self.band > self.reqband:
                self.warning = f'{self.metric} requirement not met'
        else:
            # Assumes the measured value and uncertainty represent a Normal
            # distribution, and works out the probability that the true value
            # lies in each band
            bands, limits = list(zip(*sorted(self.limit_dict.items(),
                                             key=lambda k_v: k_v[1])))
            bands = (100,) + bands if bands[0] > bands[1] else bands + (100,)
            # To Bayesian this, prepend (0,) to limits and not to probs
            # and renormalize (divide all by 1-probs[0]) and add prior
            norm_limits = [(l - float(value))/unc for l in limits]
            cum_probs = [0.] + [0.5 * (1 + math.erf(s / math.sqrt(2))) for
                                s in norm_limits] + [1.]
            probs = np.diff(cum_probs)
            if bands[0] > bands[1]:
                bands = bands[::-1]
                probs = probs[::-1]
            self.band = [b for b, p in zip(bands, probs) if p > 0.05]

            cum_prob = 0.0
            for b, p in zip(bands[:-1], probs[:-1]):
                cum_prob += p
                if cum_prob < 0.05:
                    log.fullinfo(self.result1.format(self.metric, b))
                    if b == self.reqband:
                        self.warning = (f'{self.metric} requirement not met '
                                        'at the 95% confidence level')
                elif cum_prob < 0.95:
                    log.fullinfo(self.result2.format(self.metric, b))
                else:
                    log.fullinfo(self.result3.format(self.metric, b))

    def get_measurements(self, extensions):
        if extensions == 'last':
            extensions = slice(-1, None)
        elif extensions == 'all':
            extensions = slice(None, None)

        # Ensures a Table is returned, even if no measurements
        return vstack([Table()] + [m for m in self.measurements[extensions] if m],
                      metadata_conflicts='silent')

    def info_list(self, update_band_info=True):
        """
        Creates a list suitable for sending as a fitsstore_report from the
        measurements associated with this QAReport object.

        Parameters
        ----------
        update_band_info: bool
            update the "percentile_band" and "comment" items of all extensions
            with the new band and warning information.

        Returns
        -------
        info_list: list of dicts
        """
        info_list = []
        for result in self.results:
            info_list.append({k: result[k] for k in self.fitsdict_items
                              if k in result})
        if update_band_info:
            for item in info_list:
                if item:  # Don't add to an empty item
                    item.update({"percentile_band": self.band,
                                 "comment": self.comments})
        assert len(info_list) == self.num_ext, "Info List length does not match number of extensions"
        return info_list

    def output_log_report(self, header, body, llen=32, rlen=26, logtype='stdinfo'):
        """Formats the QA report for the log"""
        logit = getattr(self.log, logtype)
        indent = ' ' * logutils.SW
        logit('')
        for line in header:
            logit(indent + line)
        logit(indent + '-' * (llen + rlen))
        for lstr, rstr in body:
            if len(rstr) > rlen and not lstr:
                logit(indent + rstr.rjust(llen + rlen))
            else:
                logit(indent + lstr.ljust(llen) + rstr.rjust(rlen))
        logit(indent + '-' * (llen + rlen))
        logit('')


class BGReport(QAReport):
    def __init__(self, ad, log=None, limit_dict=None):
        super().__init__(ad, log=log, limit_dict=limit_dict)
        self.reqband = ad.requested_bg()
        self.bunit = 'ADU' if ad.is_in_adu() else 'electron'
        self.fitsdict_items.extend(["mag", "mag_std", "electrons",
                                    "electrons_std", "nsamples"])

    def add_measurement(self, ext, sampling=100, bias_level=None):
        """
        Get the data from this extension and add them to the
        data list of the BGReport object
        """
        log = self.log
        bg, bgerr, nsamples = gt.measure_bg_from_image(ext, sampling=sampling,
                                                       gaussfit=False)
        extid = f"{self.filename} extension {ext.id}"

        if bg is not None:
            log.fullinfo("{}: Raw BG level = {:.3f}".format(extid, bg))
            if bias_level is not None:
                bg -= bias_level
                log.fullinfo(" " * len(extid)+"  Bias-subtracted BG level = "
                                              "{:.3f}".format(bg))
        measurement = Table([[bg], [bgerr], [nsamples]],
                            names=('bg', 'bgerr', 'nsamples'))

        zpt = ext.nominal_photometric_zeropoint()
        if zpt is not None:
            if bg > 0:
                exptime = ext.exposure_time()
                pixscale = ext.pixel_scale()
                try:
                    fak = 1.0 / (exptime * pixscale * pixscale)
                except TypeError:  # exptime or pixscale is None
                    pass
                else:
                    bgmag = zpt - 2.5 * math.log10(bg * fak)
                    bgmagerr = 2.5 * math.log10(1 + bgerr / bg)

                    # Need to report to FITSstore in electrons
                    if self.bunit == 'ADU':
                        fak *= ext.gain()
                    bg *= fak
                    bgerr *= fak

                    measurement['mag'] = [bgmag]
                    measurement['mag_std'] = [bgmagerr]
                    measurement['electrons'] = [bg]
                    measurement['electrons_std'] = [bgerr]
            else:
                log.warning(f"Background is less than or equal to 0 for {extid}")
        else:
            log.stdinfo("No nominal photometric zeropoint available for "
                        f"{extid}, filter {self.filter_name}")
        self.measurements.append(measurement)
        return nsamples

    def calculate_metric(self, extensions='last'):
        """
        Produces an average result from measurements made on one or more
        extensions and uses these results to calculate the actual QA band.
        The BG version of this method is unusual because each individual
        measurement is actually an average of many sampled pixel values.

        Parameters
        ----------
        extensions : str/slice
            list of extensions to average together in this calculation

        Returns
        -------
        results : dict
            list of average results (also appended to self.results if the
            calculation is being performed on the last extension)

        Updates
        -------
        self.comments : list
            All relevant comments for the ADCC and/or FITSstore reports
        """
        self.comments = []
        t = self.get_measurements(extensions)
        results = {}
        if t:
            weights = t['nsamples']
            results = {'nsamples': int(weights.sum())}
            for value_key, std_key in (('bg', 'bgerr'), ('mag', 'mag_std'),
                                       ('electrons', 'electrons_std')):
                try:
                    mean = np.ma.average(t[value_key], weights=weights)
                except KeyError:
                    pass
                else:
                    if mean is not np.ma.masked:
                        var1 = np.average((t[value_key] - mean)**2, weights=weights)
                        var2 = (weights * t[std_key] ** 2).sum() / weights.sum()
                        sigma = np.sqrt(var1 + var2)
                        # Coercion to float to ensure JSONable
                        results.update({value_key: float(mean), std_key: float(sigma)})
            self.calculate_qa_band(results.get('mag'), results.get('mag_std'))
            if self.warning:
                self.comments.append(self.warning)

        if extensions == 'last':
            self.results.append(results)
        return results

    def report(self, results, header=None):
        """
        Logs the formatted output of a measureBG report

        Parameters
        ----------
        results : dict
            output from calculate_metrics
        header : str
            header indentification for report
        """
        if header is None:
            header = f"Filename: {self.filename}"

        body = [('Sky level measurement:', '{:.0f} +/- {:.0f} {}'.
                 format(results['bg'], results['bgerr'], self.bunit))]
        if results.get('mag') is not None:
            body.append(('Mag / sq arcsec in {}:'.format(self.filter_name),
                         '{:.2f} +/- {:.2f}'.format(results['mag'], results['mag_std'])))
        if self.band:
            body.append(('BG band:', f'BG{self.bandstr(self.band)} ({self.band_info})'))
        else:
            body.append(('(BG band could not be determined)', ''))

        if self.reqband:
            body.append(('Requested BG:', f'BG{self.bandstr(self.reqband)}'))
            if self.warning:
                body.append((f'WARNING: {self.warning}', ''))
        else:
            body.append(('(Requested BG could not be determined)', ''))

        self.output_log_report([header], body)


class CCReport(QAReport):
    def __init__(self, ad, log=None, limit_dict=None):
        super().__init__(ad, log=log, limit_dict=limit_dict)
        self.reqband = ad.requested_cc()
        self.exptime = ad.exposure_time()
        if 'NODANDSHUFFLE' in ad.tags:
            log.warning("Imaging Nod-And-Shuffle. Photometry may be dubious.")
            # AFAIK the number of nod_cycles isn't actually relevant -
            # there's always 2 nod positions, thus the exposure
            # time for any given star is half the total
            self.exptime *= 0.5
        self.atm_ext = ad.nominal_atmospheric_extinction()
        if self.atm_ext is None:
            log.warning("Cannot get atmospheric extinction: assuming zero.")
            self.atm_ext = 0.0
        self.fitsdict_items.extend(["mag", "mag_std", "cloud", "cloud_std",
                                    "nsamples"])

    def add_measurement(self, ext):
        """
        Get the data from this extension and add them to the
        data list of the BGReport object

        Parameters
        ----------
        ext : single-slice AD object
            extension upon which measurement is being made
        """
        try:
            objcat = ext.OBJCAT
        except AttributeError:
            good_obj = Table()
        else:
            good_obj = objcat[np.logical_and.reduce([objcat['MAG_AUTO'] > -999,
                            objcat['MAG_AUTO'] < 90, objcat['REF_MAG'] > -999,
                            ~np.isnan(objcat['REF_MAG']),
                            ~np.isnan(objcat['REF_MAG_ERR'])])]
            good_obj['ZEROPOINT'] = (good_obj['REF_MAG'] - good_obj['MAG_AUTO'] -
                                     self.atm_ext - 2.5 * math.log10(self.exptime))
            good_obj['ZEROPOINT_ERR'] = np.sqrt(good_obj['REF_MAG_ERR'] ** 2 +
                                                good_obj['MAGERR_AUTO'] ** 2)
            nom_phot_zpt = ext.nominal_photometric_zeropoint()
            if nom_phot_zpt is not None:
                good_obj['CLOUD'] = nom_phot_zpt - good_obj['ZEROPOINT']
        self.measurements.append(good_obj)
        return len(good_obj)

    def calculate_metric(self, extensions='last'):
        """
        Produces an average result from measurements made on one or more
        extensions and uses these results to calculate the actual QA band.

        Parameters
        ----------
        extensions : str/slice
            list of extensions to average together in this calculation

        Returns
        -------
        results : dict
            list of average results (also appended to self.results if the
            calculation is being performed on the last extension)

        Updates
        -------
        self.comments : list
            All relevant comments for the ADCC and/or FITSstore reports
        """
        self.comments = []
        t = self.get_measurements(extensions)
        results = {}

        if t:
            # Sources must be free of SExtractor flags and unsaturated, and
            # <2% of pixels be otherwise flagged (typically bad/non-linear)
            good_obj = t[np.logical_and.reduce([t['FLAGS'] == 0,
                                t['NIMAFLAGS_ISO'] < 0.02 * t['ISOAREA_IMAGE'],
                                t['IMAFLAGS_ISO'] & DQ.saturated == 0])]

            # TODO: weight instead?
            # Trim out where zeropoint error > err_limit
            err_limit = 0.1 if len(good_obj) > 5 else 0.2
            good_obj = good_obj[good_obj["ZEROPOINT_ERR"] < err_limit]

            # TODO: 1-sigma clip is crap for this pre-clipping
            if len(good_obj) > 2:
                zpt_data = good_obj["ZEROPOINT"]
                mean = zpt_data.mean()
                sigma = math.sqrt(zpt_data.std() ** 2 +
                                  (good_obj["ZEROPOINT_ERR"] ** 2).mean())
                good_obj = good_obj[abs(zpt_data - mean) < sigma]

            if good_obj:
                zpt_data = good_obj["ZEROPOINT"]
                weights = 1. / (good_obj["ZEROPOINT_ERR"] ** 2)
                zpt = np.average(zpt_data, weights=weights)
                zpt_std = math.sqrt(np.average((zpt_data - zpt) ** 2, weights=weights) +
                                    1. / weights.mean())
                try:
                    cloud = np.average(good_obj["CLOUD"], weights=weights)
                    cloud_std = zpt_std
                except KeyError:
                    cloud = cloud_std = None
                results = {"mag": float(zpt), "mag_std": float(zpt_std),
                           "cloud": float(cloud), "cloud_std": float(cloud_std),
                           "nsamples": len(good_obj)}
                self.calculate_qa_band(results.get("cloud"), results.get("cloud_std"),
                                       simple=False)
                if self.warning:
                    self.comments.append(self.warning)

        if extensions == 'last':
            self.results.append(results)
        return results

    def report(self, results, all_results=None, header=None):
        """
        Logs the formatted output of a measureCC report. This is unusual in
        that it can be sent a list of results dicts and will report on the
        zeropoints for each.

        Parameters
        ----------
        results : dict
            output from calculate_metrics
        all_results : list/None
            list of outputs from individual extensions
        header : str
            header indentification for report

        Returns
        -------
        list: list of comments to be passed to the FITSstore report
        """
        if header is None:
            header = f"Filename: {self.filename}"

        header = [header]
        single_ext = all_results is None

        if single_ext:
            nsamples = results["nsamples"]
            body = [(f"Zeropoint measurement ({self.filter_name}-band):",
                     '{:.2f} +/- {:.2f}'.format(results["mag"], results["mag_std"]))]
            npz = results["cloud"] + results["mag"]  # Rather than call descriptor again
            body.append(('Nominal zeropoint:', '{:.2f}'.format(npz)))
        else:
            nsamples = 0
            for result in all_results:
                nsamples += result.get("nsamples", 0)
                rstr = ('{:.2f} +/- {:.2f}'.format(result["mag"], result["mag_std"])
                        if result.get("mag") else 'not measured')
                try:
                    body.append(('', rstr))
                except NameError:
                    body = [(f"Zeropoints by detector ({self.filter_name}-band):", rstr)]

        body.append(('Estimated cloud extinction:', '{:.2f} +/- {:.2f} mag'.
                     format(results["cloud"], results["cloud_std"])))

        if self.reqband:
            if isinstance(self.band, int):
                bands = [self.band]
            else:
                bands = self.band
            body.append(('CC bands consistent with this:', ', '.join(['CC{}'.
                   format(self.bandstr(x)) for x in bands])))
            if self.reqband:
                body.append(('Requested CC:', f'CC{self.bandstr(self.reqband)}'))
                if self.warning:
                    body.append((f'WARNING: {self.warning}', ''))
            else:
                body.append(('(Requested CC could not be determined)', ''))

        header.append(f'{nsamples} sources used to measure zeropoint')
        self.output_log_report(header, body)


class IQReport(QAReport):
    def __init__(self, ad, log=None, limit_dict=None):
        super().__init__(ad, log=log, limit_dict=limit_dict)
        self.reqband = ad.requested_iq()
        self.instrument = ad.instrument()
        self.is_ao = ad.is_ao()
        self.image_like = ad.tags.intersection(
            {'IMAGE', 'LS', 'XD', 'MOS'}) == {'IMAGE'}
        self.fitsdict_items.extend(["fwhm", "fwhm_std", "elip", "elip_std",
                                    "nsamples", "adaptive_optics", "ao_seeing",
                                    "strehl", "isofwhm", "isofwhm_std",
                                    "ee50d", "ee50d_std", "pa", "pa_std",
                                    "ao_seeing", "strehl"])
        try:
            self.airmass = ad.airmass()
        except:
            self.airmass = None

        # For AO observations, the AO-estimated seeing is used (the IQ
        # is also calculated from the image if possible)
        self.ao_seeing = None
        if self.is_ao:
            try:
                self.ao_seeing = ad.ao_seeing()
            except:
                log.warning("No AO-estimated seeing found for this AO "
                            "observation")
            else:
                log.warning("This is an AO observation, the AO-estimated "
                            "seeing will be used for the IQ band calculation")

    def add_measurement(self, ext, strehl_fn=None):
        """
        Perform a measurement on this extension and add the results to the
        measurements list of the IQReport object

        Parameters
        ----------
        ext : single-slice AD object
            extension upon which measurement is being made
        strehl_fn : None/callable
            function to calculate Strehl ratio from "sources" catalog
        """
        print("IMAGELIKE", self.image_like)
        sources = (gt.clip_sources(ext) if self.image_like
                   else gt.fit_continuum(ext))
        if (len(sources) > 0 and self.is_ao and self.image_like and
                self.instrument in ('GSAOI', 'NIRI', 'GNIRS') and
                strehl_fn is not None):
            strehl_list = strehl_fn(sources)
            if strehl_list:
                sources["strehl"] = np.ma.masked_where(np.array(strehl_list) > 0.6,
                                                       strehl_list)
        self.measurements.append(sources)
        return len(sources)

    def calculate_metric(self, extensions='last', ao_seeing_fn=None):
        """
        Produces an average result from measurements made on one or more
        extensions and uses these results to calculate the actual QA band.

        Parameters
        ----------
        extensions : str/slice
            list of extensions to average together in this calculation
        ao_seeing_fn : None/callable
            function to estimate natural seeing from "sources" catalog for
            AO images

        Returns
        -------
        results : dict
            list of average results (also appended to self.results if the
            calculation is being performed on the last extension). All
            floats are coerced to standard python float type since np.float64
            is not JSONable

        Updates
        -------
        self.comments : list
            All relevant comments for the ADCC and/or FITSstore reports
        """
        self.comments = []
        t = self.get_measurements(extensions)
        results = {}
        if t:
            fwhm_data = t["fwhm_arcsec"]
            try:
                weights = t["weight"]
            except KeyError:
                fwhm, fwhm_std = float(fwhm_data.mean()), float(fwhm_data.std())
            else:
                fwhm = float(np.average(fwhm_data, weights=weights))
                fwhm_std = float(np.sqrt(np.average((fwhm_data - fwhm) ** 2,
                                         weights=weights)))
            results = {"fwhm": fwhm, "fwhm_std": fwhm_std, "nsamples": len(t)}

            if self.image_like:
                ellip = t["ellipticity"]
                isofwhm = t["isofwhm_arcsec"]
                ee50d = t["ee50d"]
                pa = t["pa"]
                results.update({"elip": float(ellip.mean()), "elip_std": float(ellip.std()),
                                "isofwhm": float(isofwhm.mean()), "isofwhm_std": float(isofwhm.std()),
                                "ee50d": float(ee50d.mean()), "ee50d_std": float(ee50d.std()),
                                "pa": float(pa.mean()), "pa_std": float(pa.std())})
            else:
                results.update({"elip": None, "elip_std": None})

        if t or self.is_ao:
            results["adaptive_optics"] = self.is_ao

            results["ao_seeing"] = self.ao_seeing
            results["strehl"] = None
            results["strehl_std"] = None
            if self.is_ao:
                if "strehl" in t.colnames:
                    # .data here avoids a np.partition warning
                    data = sigma_clip(t["strehl"].data)
                    # Weights are used, with brighter sources being more heavily
                    # weighted. This is not simply because they will have better
                    # measurements, but because there is a bias in SExtractor's
                    # FLUX_AUTO measurement, which underestimates the total flux
                    # for fainter sources (this is due to its extrapolation of
                    # the source profile; the apparent profile varies depending
                    # on how much of the uncorrected psf is detected).
                    weights = t["flux"]
                    strehl = np.ma.average(data, weights=weights)
                    if strehl is not np.ma.masked:
                        results["strehl"] = float(strehl)
                        strehl_std = np.sqrt(np.average((data - strehl) ** 2,
                                                         weights=weights))
                        results["strehl_std"] = float(strehl_std)
                try:
                    fwhm_and_std = ao_seeing_fn(results)
                except TypeError:
                    fwhm_and_std = None
                if fwhm_and_std:
                    fwhm, fwhm_std = fwhm_and_std
                    self.comments.append("AO observation. IQ band from estimated "
                                         "natural seeing.")
                else:
                    fwhm = self.ao_seeing
                    fwhm_std = None
                    self.comments.append("AO observation. IQ band from estimated "
                                         "AO seeing.")

            if self.airmass:
                zcorr = self.airmass ** (-0.6)
                zfwhm = fwhm * zcorr
                zfwhm_std = None if fwhm_std is None else fwhm_std * zcorr
                self.calculate_qa_band(zfwhm, zfwhm_std)
            else:
                self.log.warning("Airmass not found, not correcting to zenith")
                self.calculate_qa_band(fwhm, fwhm_std)
                zfwhm = zfwhm_std = None
            results.update({"zfwhm": zfwhm, "zfwhm_std": zfwhm_std})

        if extensions == 'last':
            self.results.append(results)
        return results

    def report(self, results, header=None):
        """
        Logs the formatted output of a measureIQ report

        Parameters
        ----------
        results : dict
            output from calculate_metrics
        header : str
            header indentification for report

        Returns
        -------
        list: list of comments to be passed to the FITSstore report
        """
        log = self.log
        if header is None:
            header = f"Filename: {self.filename}"

        header = [header]
        nsamples = results.get("nsamples", 0)
        if nsamples > 0:  # AO seeing has no sources
            header.append(f'{nsamples} source(s) used to measure IQ')

        body = [('FWHM measurement:', '{:.3f} +/- {:.3f} arcsec'.
                 format(results["fwhm"], results["fwhm_std"]))] if results.get("fwhm") else []

        if self.image_like:
            if 'NON_SIDEREAL' in self.ad_tags:
                header.append('WARNING: NON SIDEREAL tracking. IQ measurements '
                              'will be unreliable')
            if results.get("elip"):
                body.append(('Ellipticity:', '{:.3f} +/- {:.3f}'.
                             format(results["elip"], results["elip_std"])))
            if self.is_ao:
                if results.get("strehl"):
                    body.append(('Strehl ratio:', '{:.3f} +/- {:.3f}'.
                                 format(results["strehl"], results["strehl_std"])))
                else:
                    body.append(('(Strehl could not be determined)', ''))

        zfwhm, zfwhm_std = results.get("zfwhm"), results.get("zfwhm_std")
        if zfwhm:
            # zfwhm_std=0.0 if there's only one measurement
            stdmsg = ('{:.3f} +/- {:.3f} arcsec'.format(zfwhm, zfwhm_std)
                      if zfwhm_std is not None else '(AO) {:.3f} arcsec'.format(zfwhm))
            body.append(('Zenith-corrected FWHM (AM {:.2f}):'.format(self.airmass),
                         stdmsg))

        if self.band:
            body.append(('IQ range for {}-band:'.format('AO' if self.is_ao
                                                        else self.filter_name),
                         'IQ{} ({} arcsec)'.format(self.bandstr(self.band), self.band_info)))
        else:
            body.append(('(IQ band could not be determined)', ''))

        if self.reqband:
            body.append(('Requested IQ:', f'IQ{self.bandstr(self.reqband)}'))
            if self.warning:
                body.append((f'WARNING: {self.warning}', ''))
        else:
            body.append(('(Requested IQ could not be determined)', ''))

        # allow comparison if "elip" is None
        if result.get("elip") != None and results.get("elip", 0) > 0.1:
            body.append(('', 'WARNING: high ellipticity'))
            self.comments.append('High ellipticity')
            if 'NON_SIDEREAL' in self.ad_tags:
                body.append(('- this is likely due to non-sidereal tracking', ''))

        if {'IMAGE', 'LS'}.issubset(self.ad_tags):
            log.warning('Through-slit IQ may be overestimated due to '
                        'atmospheric dispersion')
            body.append(('', 'WARNING: through-slit IQ measurement - '
                             'may be overestimated'))
            self.comments.append('Through-slit IQ measurement')

        if nsamples == 1:
            log.warning('Only one source found. IQ numbers may not be accurate')
            body.append(('', 'WARNING: single source IQ measurement - '
                             'no error available'))
            self.comments.append('Single source IQ measurement, no error available')
        self.output_log_report(header, body)

        if nsamples > 0:
            if not self.image_like:
                self.comments.append('IQ measured from profile cross-cut')
            if 'NON_SIDEREAL' in self.ad_tags:
                self.comments.append('Observation is NON SIDEREAL, IQ measurements '
                                     'will be unreliable')
