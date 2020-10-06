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

    def __init__(self, log=None, limit_dict=None):
        self.log = log or logutils.get_logger(__name__)
        self.metric = self.__class__.__name__[:2]
        self.limit_dict = limit_dict
        self.band = None
        self.warning = ''
        self.measurements = []
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
        value: float
            measured value for the metric
        unc: float
            uncertainty on the measurement (used only if simple=True)
        simple: bool
            use the value only? (else perform a Bayesian analysis)
        """
        cmp = lambda x, y: (x > y) - (x < y)

        self.warning = None
        if self.limit_dict is None or value is None:
            return
        if simple:
            # Straightfoward determination of which band the measured value
            # lies in. The uncertainty on the measurement is ignored.
            bands, limits = list(zip(*sorted(self.limit_dict.items(),
                                             key=lambda k_v: k_v[0], reverse=True)))
            sign = cmp(limits[1], limits[0])
            inequality = '<' if sign > 0 else '>'
            self.info = f'{inequality}{limits[0]}'
            for i in range(len(bands)):
                if cmp(float(value), limits[i]) == sign:
                    self.band = bands[i]
                    self.info = ('{}-{}'.format(*sorted(limits[i:i+2])) if
                                 i < len(bands)-1 else '{}{}'.format(
                                 '<>'.replace(inequality,''), limits[i]))
            if self.reqband is not None and self.band > self.reqband:
                self.warning = '{} requirement not met'.format(self.metric)
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

    def info_list(self):
        """Returns a list of dicts suitable for uploading to FITSstore"""
        info_list = []
        for measurement in self.measurements:
            info_list.append({k: measurement.get(k) for k in self.fitsdict_items})
        return info_list

    @staticmethod
    def output_log_report(header, body, llen=32, rlen=26, logtype='stdinfo'):
        """Formats the QA report"""
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
        super().__init__(log=log, limit_dict=limit_dict)
        self.filename = ad.filename
        self.reqband = ad.requested_bg()
        self.bunit = 'ADU' if ad.is_in_adu() else 'electron'
        self.filter_name = ad.filter_name(pretty=True)
        self.fitsdict_items.extend(["mag", "mag_std", "electrons",
                                    "electrons_std", "nsamples"])

    def add_measurement(self, ext, sampling=100, bias_level=None):
        """
        Perform a measurement on this extension and add the results to the
        measurements list of the BGReport object
        """
        log = self.log
        bg, bgerr, nsamples = gt.measure_bg_from_image(ext, sampling=sampling,
                                                       gaussfit=False)
        extver = ext.hdr['EXTVER']
        extid = f"{self.filename}{extver}"

        if bg is not None:
            log.fullinfo("{}: Raw BG level = {:.3f}".format(extid, bg))
            if bias_level is not None:
                bg -= bias_level
                log.fullinfo(" " * len(extid)+"  Bias-subtracted BG level = "
                                              "{:.3f}".format(bg))
        measurement = {"bg": bg, "bgerr": bgerr, "bunit": self.bunit,
                       "nsamples": nsamples, "extver": extver}

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

                    measurement.update({"mag": bgmag, "mag_std": bgmagerr,
                                        "electrons": bg, "electrons_std": bgerr})
                    self.calculate_qa_band(bgmag, bgmagerr)
                    measurement.update({"percentile_band": self.band,
                                        "comment": [self.warning]})
            else:
                log.warning(f"Background is less than or equal to 0 for {extid}")
        else:
            log.stdinfo("No nominal photometric zeropoint available for "
                        f"{extid}, filter {self.filter_name}")
        self.measurements.append(measurement)

    def average_measurements(self, weights='sample'):
        results = {}
        for value_key, std_key in (('bg', 'bgerr'), ('mag', 'mag_std')):
            data = np.array([(m.get(value_key), m.get(std_key), m.get('nsamples'))
                              for m in self.measurements if m.get(value_key) is not None]).T
            if weights == 'variance':
                wt = 1.0 / data[1]
            elif weights == 'sample':
                wt = data[2]
            else:
                wt = np.ones((data.shape[1],))
            #total_samples = data[2].sum()
            mean = np.average(data[0], weights=wt)
            var1 = np.average((data[0] - mean)**2, weights=wt)
            var2 = (wt * data[1] ** 2).sum() / wt.sum()
            sigma = np.sqrt(var1 + var2)
            results.update({value_key: mean, std_key: sigma})
        return results

    def report(self, report_type=None, results=None):
        """
        Logs the formatted output of a measureBG report

        Parameters
        ----------
        bg_count: Measurement
            background measurement, error, and number of samples
        bunit: str
            units of the background measurement
        bg_mag: Measurement
            background measurement and error in magnitudes (and number of samples)
        qastatus: QAstatus namedtuple
            information about the actual band

        Returns
        -------
        list: list of comments to be passed to the FITSstore report
        """
        comments = []
        headstr = f"Filename: {self.filename}"
        if report_type == 'last':
            results = self.measurements[-1]
            headstr += f":{results['extver']}"
        elif results is None:
            raise ValueError("Cannot report without results unless report_type is 'last'")

        body = [('Sky level measurement:', '{:.0f} +/- {:.0f} {}'.
                 format(results['bg'], results['bgerr'], self.bunit))]
        if results['mag'] is not None:
            body.append(('Mag / sq arcsec in {}:'.format(self.filter_name),
                         '{:.2f} +/- {:.2f}'.format(results['mag'], results['mag_std'])))
        if self.band:
            body.append(('BG band:', f'BG{self.bandstr(self.band)} ({self.info})'))
        else:
            body.append(('(BG band could not be determined)', ''))

        if self.reqband:
            body.append(('Requested BG:', f'BG{self.bandstr(self.reqband)}'))
            if self.warning:
                body.append(('WARNING: {}'.format(self.warning), ''))
                comments.append(self.warning)
        else:
            body.append(('(Requested BG could not be determined)', ''))

        self.output_log_report([headstr], body)
        return comments


class CCReport(QAReport):
    def __init__(self, ad, log=None, limit_dict=None):
        super().__init__(log=log, limit_dict=limit_dict)
        self.filename = ad.filename
        self.reqband = ad.requested_cc()
        self.filter_name = ad.filter_name(pretty=True)
        self.fitsdict_items.extend(["mag", "mag_std", "electrons",
                                    "electrons_std", "nsamples"])


class IQReport(QAReport):
    def __init__(self, ad, log=None, limit_dict=None):
        super().__init__(log=log, limit_dict=limit_dict)
        self.filename = ad.filename
        self.reqband = ad.requested_iq()
        self.filter_name = ad.filter_name(pretty=True)
        self.fitsdict_items.extend(["mag", "mag_std", "electrons",
                                    "electrons_std", "nsamples"])
