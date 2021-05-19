#
#                                                                        DRAGONS
#
#                                                               calrequestlib.py
# ------------------------------------------------------------------------------
import hashlib

from gemini_instruments.common import Section

from gempy.utils import logutils

# GetterError is needed by a module that imports this one
from .file_getter import get_file_iterator, GetterError
# ------------------------------------------------------------------------------
log = logutils.get_logger(__name__)
# ------------------------------------------------------------------------------
# Currently delivers transport_request.calibration_search fn.
#calibration_search = cal_search_factory()
# ------------------------------------------------------------------------------
def get_request(url, filename):
    iterator = get_file_iterator(url)
    with open(filename, 'wb') as fd:
        for chunk in iterator:
            fd.write(chunk)
    return filename


def generate_md5_digest(filename):
    md5 = hashlib.md5()
    fdata = open(filename, 'rb').read()
    md5.update(fdata)
    return md5.hexdigest()


class CalibrationRequest:
    """
    Request objects are passed to a calibration_search() function

    """
    def __init__(self, ad, caltype=None, procmode=None):
        self.ad = ad
        self.caltype = caltype
        self.procmode = procmode
        self.datalabel = ad.data_label()
        self.descriptors = None
        self.filename = ad.filename
        self.tags = ad.tags

    def as_dict(self):
        retd = {}
        retd.update(
            {'filename'   : self.filename,
             'caltype'    : self.caltype,
             'procmode'   : self.procmode,
             'datalabel'  : self.datalabel,
             "descriptors": self.descriptors,
             "tags"       : self.tags,
            }
        )

        return retd

    def __str__(self):
        tempStr = "filename: {}\nDescriptors: {}\nTypes: {}"
        tempStr = tempStr.format(self.filename, self.descriptors, self.tags)
        return tempStr


def get_cal_requests(inputs, caltype, procmode=None, is_local=True):
    """
    Builds a list of :class:`.CalibrationRequest` objects, one for each `ad`
    input.

    Parameters
    ----------
    inputs: <list>
        A list of input AstroData instances.

    caltype: <str>
        Calibration type, eg., 'processed_bias', 'flat', etc.

    Returns
    -------
    rq_events: <list>
        A list of CalibrationRequest instances, one for each passed
       'ad' instance in 'inputs'.

    """
    options = {'central_wavelength': {'asMicrometers': True}}

    handle_returns = (lambda x: x) if is_local else _handle_returns

    rq_events = []
    for ad in inputs:
        log.debug("Received calibration request for {}".format(ad.filename))
        rq = CalibrationRequest(ad, caltype, procmode)
        # Check that each descriptor works and returns a sensible value.
        desc_dict = {}
        for desc_name in ad.descriptors:
            try:
                descriptor = getattr(ad, desc_name)
            except AttributeError:
                pass
            else:
                kwargs = options[desc_name] if desc_name in list(options.keys()) else {}
                try:
                    dv = handle_returns(descriptor(**kwargs))
                except:
                    dv = None
                # Munge list to value if all item(s) are the same
                if isinstance(dv, list):
                    dv = dv[0] if all(v == dv[0] for v in dv) else "+".join(
                        [str(v) for v in dv])
                desc_dict[desc_name] = dv
        rq.descriptors = desc_dict
        rq_events.append(rq)
    return rq_events


def _handle_returns(dv):
    # TODO: This sends "old style" request for data section, where the section
    #       is converted to a regular 4-element list. In "new style" requests,
    #       we send the Section as-is. This will need to be revised when
    #       (eventually) FitsStorage upgrades to new AstroData
    if isinstance(dv, list) and isinstance(dv[0], Section):
        return [[el.x1, el.x2, el.y1, el.y2] for el in dv]
    return dv
