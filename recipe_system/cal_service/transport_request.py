#
#                                                                        DRAGONS
#
#
#                                                           transport_request.py
# ------------------------------------------------------------------------------
from future import standard_library
standard_library.install_aliases()

from builtins import str

from os.path import join
from os.path import basename
from pprint  import pformat
from xml.dom import minidom

import urllib.request
import urllib.parse
import urllib.error

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
def upload_calibration(filename, is_science=False):
    """
    Uploads a calibration file to the FITS Store.

    Parameters
    ----------
    filename: <str>, File to be uploaded

    Returns
    -------
    <void>

    """
    fn = basename(filename)
    url = UPLOADPROCCAL
    if is_science:
        url = url.replace('upload_processed_cal', 'upload_file')
    url = join(url, fn)
    postdata = open(filename, 'rb').read()
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

def calibration_search(rq, howmany=1, return_xml=False):
    """
    Receives a CalibrationRequest object, encodes the data and make the request
    on the appropriate server. Returns a URL, if any, and the MD5 hash checksum.

    Parameters
    ----------
    rq: <instance>, CalibrationRequest obj

    howmany: <int>
        Maxinum number of calibrations to return

    return_xml: <bool>
        Return the xml message to the caller when no URL is returned from the
        calibration server.

    Returns
    -------
    calurlel, calurlmd5: <list>, <list>
        A tuple of the matching URLs and md5 hash checksums

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
    postdata = urllib.parse.urlencode(sequence).encode('utf-8')
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
        calurlel = [d.childNodes[0].data
                    for d in dom.getElementsByTagName('url')[:howmany]]
        calurlmd5 = [d.childNodes[0].data
                     for d in dom.getElementsByTagName('md5')[:howmany]]
    except IndexError:
        return (None, preerr)

    for url in calurlel:
        log.stdinfo(repr(url))

    return (calurlel, calurlmd5)

def handle_returns(dv):
    # TODO: This sends "old style" request for data section, where the section
    #       is converted to a regular 4-element list. In "new style" requests,
    #       we send the Section as-is. This will need to be revised when
    #       (eventually) FitsStorage upgrades to new AstroData
    if isinstance(dv, list) and isinstance(dv[0], Section):
        return [[el.x1, el.x2, el.y1, el.y2] for el in dv]

    return dv
