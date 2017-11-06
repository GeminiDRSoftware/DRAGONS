#
#                                                           request_transport.py
# ------------------------------------------------------------------------------
from future import standard_library
standard_library.install_aliases()
from builtins import str
import urllib.request, urllib.parse, urllib.error
import urllib.request, urllib.error, urllib.parse
import traceback

from os.path import join, basename
from xml.dom import minidom
from pprint  import pformat

from gemini_instruments.common import Section

from gempy.utils import logutils
from . import calurl_dict
# ------------------------------------------------------------------------------
CALURL_DICT   = calurl_dict.calurl_dict
_CALMGR       = CALURL_DICT["CALMGR"]
UPLOADPROCCAL = CALURL_DICT["UPLOADPROCCAL"]
UPLOADCOOKIE  = CALURL_DICT["UPLOADCOOKIE"]
# ------------------------------------------------------------------------------
# sourced from fits_storage.gemini_metadata_utils.cal_types
CALTYPES = [
    "arc",
    "bias",
    "dark",
    # flats
    "flat",
    "domeflat",
    "lampoff_flat",
    "lampoff_domeflat",
    "polarization_flat",
    "qh_flat",
    # masks (use caltype='mask' for MDF queries.)
    "mask",
    "pinhole_mask",
    "ronchi_mask",
    # processed cals
    "processed_arc",
    "processed_bias",
    "processed_dark",
    "processed_flat",
    "processed_fringe",
    # other ...
    "specphot",
    "spectwilight",
    "astrometric_standard",
    "photometric_standard",
    "telluric_standard",
    "polarization_standard"
]
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
    fn = basename(filename)
    url = join(UPLOADPROCCAL, fn)
    postdata = open(filename).read()
    try:
        rq = urllib.request.Request(url)
        rq.add_header('Content-Length', '%d' % len(postdata))
        rq.add_header('Content-Type', 'application/octet-stream')
        rq.add_header('Cookie', 'gemini_fits_upload_auth=%s' % UPLOADCOOKIE)
        u = urllib.request.urlopen(rq, postdata)
        response = u.read()
    except urllib.error.HTTPError as error:
        log.error(str(error))
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
    if rq.caltype not in CALTYPES:
        calserv_msg = "Unrecognised caltype '{}'".format(rq.caltype)
        return (None, calserv_msg)

    rqurl = join(CALMGR, rq.caltype)
    log.stdinfo("CENTRAL CALIBRATION SEARCH: {}".format(rqurl))
    rqurl = rqurl + "/{}".format(rq.filename)
    # encode and send request
    sequence = [("descriptors", rq.descriptors), ("types", rq.tags)]
    postdata = urllib.parse.urlencode(sequence)
    response = "CALIBRATION_NOT_FOUND"
    try:
        calRQ = urllib.request.Request(rqurl)
        u = urllib.request.urlopen(calRQ, postdata)
        response = u.read()
    except urllib.error.HTTPError as err:
        log.error(str(err))
        return (None, str(err))
    except urllib.error.URLError as err:
        log.error(str(err))
        return (None, str(err))

    if return_xml:
        return (None, response)

    nones = []
    for dname, dval in list(rq.descriptors.items()):
        if dval is None:
            nones.append(dname)

    preerr = RESPONSESTR % {"sequence": pformat(sequence),
                            "response": response.strip(),
                            "nones"   : ", ".join(nones) \
                            if len(nones) > 0 else "No Nones Sent"}
    
    try:
        dom = minidom.parseString(response)
        calel = dom.getElementsByTagName("calibration")
        calurlel = dom.getElementsByTagName('url')[0].childNodes[0]
        calurlmd5 = dom.getElementsByTagName('md5')[0].childNodes[0]
    except IndexError:
        return (None, preerr)

    log.stdinfo(repr(calurlel.data))

    return (calurlel.data, calurlmd5.data)

def handle_returns(dv):
    # TODO: This sends "old style" request for data section, where the section
    #       is converted to a regular 4-element list. In "new style" requests,
    #       we send the Section as-is. This will need to be revised when
    #       (eventually) FitsStorage upgrades to new AstroData
    if isinstance(dv, list) and isinstance(dv[0], Section):
        return [[el.x1, el.x2, el.y1, el.y2] for el in dv]
    else:
        return dv
