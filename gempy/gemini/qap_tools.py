#
#                                                                        DRAGONS
#
#                                                                   gempy.gemini
#                                                                   qap_tools.py
# ------------------------------------------------------------------------------
import getpass
import json
import os
import socket
import sys
import math
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import TimeoutError

import numpy as np
from astropy.table import Table, vstack
from astropy.stats import sigma_clip

from astrodata import __version__ as ad_version
from recipe_system.adcc.servers.eventsManager import EventsManager

from ..utils import logutils
from . import gemini_tools as gt

# ------------------------------------------------------------------------------
log = logutils.get_logger(__name__)
MURL = "http://localhost:8777/event_report" # metrics
SURL = "http://localhost:8777/spec_report"  # spec data
# ------------------------------------------------------------------------------
def ping_adcc():
    """
    Check that there is an adcc running by requesting its site information.

    Returns
    -------
        <bool>: An adcc is running

    """
    upp = False
    site = None
    url = "http://localhost:8777/rqsite.json"
    try:
        request = urllib.request.Request(url)
        adcc_file = urllib.request.urlopen(request)
        site = adcc_file.read()
        adcc_file.close()
    except (urllib.error.HTTPError, urllib.error.URLError):
        pass

    if site:
        upp = True

    return upp

def adcc_report(ad=None, name=None, metric_report=None, metadata=None):
    """
    Parameters
    ----------
    ad: AstroData
        input image
    name: <str>
        type of metric being reported (IQ, ZP, SB, PE)
    metric_report: <dict>
        the QA info
    metadata: <dict>
        metadata of the 'ad' instance. Usually, this is None and
        EventsManager.append_event() will make this dict.

    """
    if not ping_adcc():
        return

    if name == 'spec':
        URL = SURL
    else:
        URL = MURL

    evman = EventsManager()
    evman.append_event(ad=ad, name=name, mdict=metric_report, metadata=metadata)
    event_pkt = evman.event_list.pop()
    jdata = json.dumps(event_pkt).encode('utf-8')
    postdata = jdata
    try:
        post_request = urllib.request.Request(URL)
        postr = urllib.request.urlopen(post_request, postdata)
        postr.read()
        postr.close()
    except urllib.error.HTTPError:
        log.warning("Attempt to deliver metrics to adcc failed.")

    return

def status_report(status):
    """
    Parameters
    ----------
        status: <dict>
                A 'reduce_status' report.

    A status parameter is <dict> of the form,

        status = {"adinput": ad, "current": <str>, "logfile": <str>}

    The key value of 'current' may be any string, but usually will be one of
    '<primitive_name>', '.ERROR:', 'Finished'. <primitive_name> will be
    the currently executing primitive. 'Finished' indicates execution
    completed successfully. ERROR messages should be accompanied by a non-zero
    exit code.
    E.g.,

        status = {"adinput": ad, "current": ".ERROR: 23", "logfile": log}

    """
    if not ping_adcc():
        return

    ad = status['adinput']
    mdict = {"current": status['current'], "logfile": status['logfile']}
    evman = EventsManager()
    evman.append_event(ad=ad, name='status', mdict=mdict, msgtype='reduce_status')
    event_pkt = evman.event_list.pop()
    postdata = json.dumps(event_pkt).encode('utf-8')
    try:
        post_request = urllib.request.Request(MURL)
        postr = urllib.request.urlopen(post_request, postdata)
        postr.read()
        postr.close()
    except urllib.error.HTTPError:
        log.warning("Attempt to deliver status report to adcc failed.")

    return

def fitsstore_report(ad, metric, info_list, calurl_dict, mode, upload=False):
    """
    Parameters
    ----------
    ad: AstroData
        input image

    metric: str
        type of metric being reported (IQ, ZP, SB, PE)

    info_list: list
        the QA info, one dict item per extension

    calurl_dict: dict
        information about the FITSStore (needed if report gets sent)

    mode: <list>
        A string indicating recipe mode ('qa', 'sq', or 'ql')

    upload: <bool>
        Upload metrics to fitstore. Default is False.

    Returns
    -------
    dict
        the QA report

    """
    if metric not in ["iq", "zp", "sb", "pe"]:
        raise ValueError("Unknown metric {}".format(metric))

    # Empty qareport dictionary to build into
    qareport = {}

    # Compose metadata
    qareport["hostname"]   = socket.gethostname()
    qareport["userid"]     = getpass.getuser()
    qareport["processid"]  = os.getpid()
    qareport["executable"] = os.path.basename(sys.argv[0])

    # These may need revisiting.  There doesn't seem to be a
    # way to access a version name or number for the primitive
    # set generating this metric
    qareport["software"] = "QAP"
    qareport["software_version"] = ad_version

    # context --> mode, a <str>.
    # Here we pass 'mode' to the context key for fitsstore.
    qareport["context"] = mode

    qametric_list = []
    for ext, info in zip(ad, info_list):
        # No report is given for an extension without information
        if info:
            qametric = {"filename": ad.filename}
            try:
                qametric["datalabel"] = ad.data_label()
            except:
                qametric["datalabel"] = None
            try:
                qametric["detector"] = ext.array_name()
            except:
                qametric["detector"] = None

            # Extract catalog name from the table header comment
            if metric in ('zp', 'pe'):
                catalog = ad.REFCAT.meta['header'].get('CATALOG', 'SDSS8')
                info.update({"photref" if metric == 'zp' else "astref":catalog})
            qametric.update({metric: info})
            qametric_list.append(qametric)

    # Add qametric dictionary into qareport
    qareport["qametric"] = qametric_list

    if upload:
        send_fitsstore_report(qareport, calurl_dict)
    return qareport

def send_fitsstore_report(qareport, calurl_dict):
    """
    Send a QA report to the FITSStore for ingestion.

    Parameters
    ----------
    qareport: <dict>
        QA metrics report

    calurl_dict: <dict>
        Provides FITSstore URLs. See the DRAGONS file,
        recipe_system.cal_service.calurl_dict

    Return
    ------
    <void>

    """
    tout_msg = "{} - Could not deliver metrics to fitsstore. "
    tout_msg += "Server failed to respond."
    qalist = [qareport]
    jdata = json.dumps(qalist).encode('utf-8')
    try:
        req = urllib.request.Request(url=calurl_dict["QAMETRICURL"], data=jdata)
        f = urllib.request.urlopen(req)
        f.close()
    except TimeoutError as err:
        log.warning(tout_msg.format('TimeoutError'))
    except urllib.error.HTTPError as xxx_todo_changeme:
        urllib.error.URLError = xxx_todo_changeme
        log.warning("Attempt to deliver metrics to fitsstore failed.")
    except Exception as err:
        log.warning(str(err))
    return


class QAReport:
    result1 = '95% confidence test indicates worse than {}{}'
    result2 = '95% confidence test indicates borderline {}{} or one band worse'
    result3 = '95% confidence test indicates {}{} or better'

    def __init__(self, ad, log=None, limit_dict=None):
        self.log = log or logutils.get_logger(__name__)
        self.filename = ad.filename
        self.filter_name = ad.filter_name(pretty=True)
        self.ad_tags = ad.tags
        self.metric = self.__class__.__name__[:2]
        self.limit_dict = limit_dict
        self.band = None
        self.measurements = []
        self.results = []
        self.comments = []
        self.fitsdict_items = ["percentile_band", "comment"]

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

        self.warning = ''
        if self.limit_dict is None or value is None:
            return
        simple &= unc is not None

        if simple:
            # Straightfoward determination of which band the measured value
            # lies in. The uncertainty on the measurement is ignored.
            bands, limits = list(zip(*sorted(self.limit_dict.items(),
                                             key=lambda k_v: k_v[0], reverse=True)))
            sign = cmp(limits[1], limits[0])
            inequality = '<' if sign > 0 else '>'
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
        return vstack([m for m in self.measurements[extensions] if m])

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
        return info_list

    @staticmethod
    def output_log_report(header, body, llen=32, rlen=26, logtype='stdinfo'):
        """Formats the QA report for the log"""
        logit = getattr(log, logtype)
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

    def measure(self, ext, sampling=100, bias_level=None):
        """
        Get the data from this extension and add them to the
        data list of the BGReport object
        """
        log = self.log
        bg, bgerr, nsamples = gt.measure_bg_from_image(ext, sampling=sampling,
                                                       gaussfit=False)
        extver = ext.hdr['EXTVER']
        extid = f"{self.filename}:{extver}"

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
                        gain = ext.gain()
                        bg *= gain
                        bgerr *= gain

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
        weights = t['nsamples']
        results = {'nsamples': weights.sum()}
        for value_key, std_key in (('bg', 'bgerr'), ('mag', 'mag_std'),
                                   ('electrons', 'electrons_std')):
            mean = np.ma.average(t[value_key], weights=weights)
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
        if results['mag'] is not None:
            body.append(('Mag / sq arcsec in {}:'.format(self.filter_name),
                         '{:.2f} +/- {:.2f}'.format(results['mag'], results['mag_std'])))
        if self.band:
            body.append(('BG band:', f'BG{self.bandstr(self.band)} ({self.band_info})'))
        else:
            body.append(('(BG band could not be determined)', ''))

        if self.reqband:
            body.append(('Requested BG:', f'BG{self.bandstr(self.reqband)}'))
            if self.warning:
                body.append(('WARNING: {}'.format(self.warning), ''))
        else:
            body.append(('(Requested BG could not be determined)', ''))

        self.output_log_report([header], body)


class CCReport(QAReport):
    def __init__(self, ad, log=None, limit_dict=None):
        super().__init__(ad, log=log, limit_dict=limit_dict)
        self.reqband = ad.requested_cc()
        self.fitsdict_items.extend(["mag", "mag_std", "cloud", "cloud_std",
                                    "nsamples"])


class IQReport(QAReport):
    def __init__(self, ad, log=None, limit_dict=None):
        super().__init__(ad, log=log, limit_dict=limit_dict)
        self.measure = True
        self.reqband = ad.requested_iq()
        self.instrument = ad.instrument()
        self.is_ao = ad.is_ao()
        self.image_like = 'IMAGE' in ad.tags and not hasattr(ad, 'MDF')
        self.fitsdict_items.extend(["fwhm", "fwhm_std", "elip", "elip_std",
                                    "nsamples", "adaptive_optics", "ao_seeing",
                                    "strehl", "isofwhm", "isofwhm_std",
                                    "ee50d", "ee50d_std", "pa", "pa_std",
                                    "ao_seeing", "strehl"])
        try:
            self.airmass = ad.airmass()
        except:
            self.airmass = None

        if not {'IMAGE', 'SPECT'} & ad.tags:
            log.warning(f"{ad.filename} is not IMAGE or SPECT; "
                        "no IQ measurement will be performed")
            self.measure = False

        # Check that the data is not an image with non-square binning
        if 'IMAGE' in ad.tags:
            xbin = ad.detector_x_bin()
            ybin = ad.detector_y_bin()
            if xbin != ybin:
                log.warning("No IQ measurement possible, image {} is {} x "
                            "{} binned data".format(ad.filename, xbin, ybin))
                self.measure = False

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
        sources = (gt.clip_sources(ext) if self.image_like
                   else gt.fit_continuum(ext))
        if (len(sources) > 0 and self.is_ao and self.image_like and
                self.instrument in ('GSAOI', 'NIRI', 'GNIRS') and
                strehl_fn is not None):
            strehl_list = np.array(strehl_fn(sources))
            if strehl_list:
                sources["strehl"] = np.ma.masked_where(strehl_list > 0.6,
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
        t = self.get_measurements(extensions)
        self.comments = []
        fwhm_data = t["fwhm_arcsec"]
        try:
            weights = t["weight"]
        except KeyError:
            fwhm, fwhm_std = float(fwhm_data.mean()), float(fwhm_data.std())
        else:
            fwhm = float(np.average(fwhm_data, weights=weights))
            fwhm_std = float(np.sqrt(np.average((fwhm_data - fwhm) ** 2),
                                     weights=weights))
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
        results["adaptive_optics"] = self.is_ao

        # TODO: Handle GSAOI
        results["ao_seeing"] = self.ao_seeing
        results["strehl"] = None
        results["strehl_std"] = None
        if self.is_ao:
            if "strehl" in t.colnames:
                data = sigma_clip(t["strehl"])
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
                    strehl_std = np.sqrt(np.ma.average((data - strehl) ** 2,
                                                       weights=weights))
                    results["strehl_std"] = float(strehl_std)
            try:
                fwhm_and_std = ao_seeing_fn(results)
            except TypeError:
                fwhm_and_std = None
            if fwhm_and_std:
                fwhm, fwhm_std = fwhm_and_std
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
        if header is None:
            header = f"Filename: {self.filename}"

        header = [header]
        nsamples = results.get("samples", 0)
        if nsamples > 0:  # AO seeing has no sources
            header.append(f'{nsamples} sources used to measure IQ')

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
                if results["strehl"]:
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
            body.append(('Requested IQ:', 'IQ{}'.format(self.bandstr(self.reqband))))
            if self.warning:
                body.append(('WARNING: {}'.format(self.warning), ''))
                self.comments.append(self.warning)
        else:
            body.append(('(Requested IQ could not be determined)', ''))

        if results.get("elip", 0) > 0.1:
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
