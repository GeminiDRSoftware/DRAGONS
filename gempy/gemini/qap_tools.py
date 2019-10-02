#
#                                                                        DRAGONS
#
#                                                                   gempy.gemini
#                                                                   qap_tools.py
# ------------------------------------------------------------------------------
from future import standard_library
standard_library.install_aliases()
from builtins import zip
from concurrent.futures import TimeoutError
# ------------------------------------------------------------------------------
import os
import sys
import json
import getpass
import socket
import urllib.request, urllib.error, urllib.parse

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
        post_request = urllib.request.Request(URL)
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
