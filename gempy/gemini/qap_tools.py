#
#                                                                  gemini_python
#
#                                                                   gempy.gemini
#                                                                   qap_tools.py
# ------------------------------------------------------------------------------
__version__ = 'beta (new hope)'
# ------------------------------------------------------------------------------
import os
import sys
import json
import getpass
import socket
import urllib2

from ..utils import logutils

from astrodata import __version__ as ad_version
from recipe_system.adcc.servers.eventsManager import EventsManager

# ------------------------------------------------------------------------------
log = logutils.get_logger(__name__)
URL = "http://localhost:8777/event_report"
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
        request = urllib2.Request(url)
        adcc_file = urllib2.urlopen(request)
        site = adcc_file.read()
        adcc_file.close()
    except (urllib2.HTTPError, urllib2.URLError):
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

    evman = EventsManager()
    evman.append_event(ad=ad, name=name, mdict=metric_report, metadata=metadata)
    event_pkt = evman.event_list.pop()
    postdata = json.dumps(event_pkt)
    try:
        post_request = urllib2.Request(URL)
        postr = urllib2.urlopen(post_request, postdata)
        postr.read()
        postr.close()
    except urllib2.HTTPError:
        log.warn("Attempt to deliver metrics to adcc failed.")

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
    postdata = json.dumps(event_pkt)
    try:
        post_request = urllib2.Request(URL)
        postr = urllib2.urlopen(post_request, postdata)
        postr.read()
        postr.close()
    except urllib2.HTTPError:
        log.warn("Attempt to deliver status report to adcc failed.")

    return

def fitsstore_report(ad, metric, info_list, calurl_dict, context, upload=False):
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
    qareport["context"] = context
    
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
    Sends a QA report to the FITSStore for ingestion

    Parameters
    ----------
    qareport: dict
        the QA report
    calurl_dict: dict
        information about the FITSstore

    """
    qalist = [qareport]
    try:
        req = urllib2.Request(url=calurl_dict["QAMETRICURL"], data=json.dumps(qalist))
        f = urllib2.urlopen(req)
        f.close()
    except urllib2.HTTPError, urllib2.URLError:
        log.warn("Attempt to deliver metrics to fitsstore failed.")

    return
