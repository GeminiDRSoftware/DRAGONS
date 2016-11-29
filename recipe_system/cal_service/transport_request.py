#
#                                                           request_transport.py
# ------------------------------------------------------------------------------
import urllib
import urllib2
import datetime
import traceback

from os.path import join, basename
from xml.dom import minidom
from pprint  import pformat

from gempy.utils import logutils
from . import calurl_dict
# ------------------------------------------------------------------------------
CALURL_DICT   = calurl_dict.calurl_dict
UPLOADPROCCAL = CALURL_DICT["UPLOADPROCCAL"]
UPLOADCOOKIE  = CALURL_DICT["UPLOADCOOKIE"]
_CALMGR       = CALURL_DICT["CALMGR"]
# ------------------------------------------------------------------------------
CALTYPEDICT = { "arc": "arc",
                "bias": "bias",
                "dark": "dark",
                "flat": "flat",
                "processed_arc":   "processed_arc",
                "processed_bias":   "processed_bias",
                "processed_dark":   "processed_dark",
                "processed_flat":   "processed_flat",
                "processed_fringe": "processed_fringe"}
# -----------------------------------------------------------------------------
RESPONSESTR = """########## Request Data BEGIN ##########
%(sequence)s
########## Request Data END ##########

########## Calibration Server Response BEGIN ##########
%(response)s
########## Calibration Server Response END ##########

########## Nones Report (descriptors that returned None):
%(nones)s
########## Note: all descriptors shown above, scroll up.
        """
# -----------------------------------------------------------------------------
log = logutils.get_logger(__name__)

# -----------------------------------------------------------------------------
def upload_calibration(filename):
    """Uploads a calibration file to the FITS Store.

    :parameter filename: file to be uploaded.
    :type filename: <str>

    :return:     <void>
    """
    fn  = basename(filename)
    url = join(UPLOADPROCCAL, fn)
    postdata = open(filename).read()
    try:
        rq = urllib2.Request(url)
        rq.add_header('Content-Length', '%d' % len(postdata))
        rq.add_header('Content-Type', 'application/octet-stream')
        rq.add_header('Cookie', 'gemini_fits_upload_auth=%s' % UPLOADCOOKIE)
        u = urllib2.urlopen(rq, postdata)
        response = u.read()
    except urllib2.HTTPError, error:
        contents = error.read()
        raise
    return


def calibration_search(rq, return_xml=False):
    """
    Recieves a CalibrationRequest object, encodes the data and make the request
    on the appropriate server. Returns a URL, if any, and the MD5 hash checksum.

    :parameter rq: CalibrationRequest obj
    :type rq: <CalibrationRequest object>

    :parameter return_xml: return the xml message to the caller when no URL
                           is returned from the cal server.
    :type return_xml: <bool>

    :return: A tuple of the matching URL and md5 hash checksum
    :rtype: (<str>, <str>)

    """
    rqurl = None
    calserv_msg = None
    CALMGR = _CALMGR
    source = rq.source
    rqurl = join(CALMGR, CALTYPEDICT[rq.caltype])
    log.stdinfo("CENTRAL CAL SEARCH: {}".format(rqurl))
    rqurl = rqurl + "/{}".format(rq.filename)
    # encode and send request
    sequence = [("descriptors", rq.descriptors), ("types", rq.tags)]
    postdata = urllib.urlencode(sequence)
    response = "CALIBRATION_NOT_FOUND"
    try:
        calRQ = urllib2.Request(rqurl)
        u = urllib2.urlopen(calRQ, postdata)
        response = u.read()
    except urllib2.HTTPError, error:
        calserv_msg = error.read()
        traceback.print_exc()
        return (None, calserv_msg)

    if return_xml:
        return (None, response)

    nones = []
    for dname, dval in rq.descriptors.items():
        if dval is None:
            nones.append(dname)

    preerr = RESPONSESTR % { "sequence": pformat(sequence),
                             "response": response.strip(),
                             "nones"   : ", ".join(nones) \
                             if len(nones) > 0 else "No Nones Sent" }

    try:
        dom = minidom.parseString(response)
        calel = dom.getElementsByTagName("calibration")
        calurlel = dom.getElementsByTagName('url')[0].childNodes[0]
        calurlmd5 = dom.getElementsByTagName('md5')[0].childNodes[0]
    except IndexError:
        return (None, preerr)

    log.stdinfo(repr(calurlel.data))

    return (calurlel.data, calurlmd5.data)
