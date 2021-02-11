#
#                                                                        DRAGONS
#
#                                                               calrequestlib.py
# ------------------------------------------------------------------------------
import hashlib

from gempy.utils import logutils

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

    _handle_returns = lambda x: x if is_local else _handle_returns

    rq_events = []
    for ad in inputs:
        log.stdinfo("Received calibration request for {}".format(ad.filename))
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
                    dv = _handle_returns(descriptor(**kwargs))
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
