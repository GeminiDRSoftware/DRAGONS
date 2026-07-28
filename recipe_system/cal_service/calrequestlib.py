#
#                                                                        DRAGONS
#
#                                                               calrequestlib.py
# ------------------------------------------------------------------------------
import datetime
import hashlib

import numpy

from gempy.utils import logutils

# ------------------------------------------------------------------------------
log = logutils.get_logger(__name__)
# ------------------------------------------------------------------------------
# Currently delivers transport_request.calibration_search fn.
#calibration_search = cal_search_factory()
# ------------------------------------------------------------------------------


def generate_md5_digest(filename):
    with open(filename, 'rb') as f:
        digest = hashlib.file_digest(f, "md5")
    return digest.hexdigest()


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
        return (f"filename: {self.filename}\n"
                f"Descriptors: {self.descriptors}\n"
                f"Types: {self.tags}")


def get_descriptors_dict(ad):
    # Helper function for get_cal_requests. Builds the descriptors dict that
    # we post to the calmgr. Called elsewhere (eg in the FitsStorage calmgr
    # post tests), can be used to build other calmgr post clients

    options = {'central_wavelength': {'asMicrometers': True}}

    desc_dict = {}
    for desc_name in ad.descriptors:
        # Check that each descriptor works and returns a sensible value.
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



    # Add composite detector_binning to request so we can query Header field of same name in cals
    dvx = desc_dict["detector_x_bin"] if "detector_x_bin" in desc_dict else None
    dvy = desc_dict["detector_y_bin"] if "detector_y_bin" in desc_dict else None

    # Quick check to handle when these are dictionaries, i.e. for GHOST data
    if (dvx is not None) and (dvy is not None):
        if isinstance(dvx, dict) or isinstance(dvy, dict):
            # dict always means multi-arm data
            # We preserve binning if it matches across all arms, else None
            dvxs = set([x for x in dvx.values() if x is not None])
            dvys = set([y for y in dvy.values() if y is not None])
            if len(dvxs) == 1 and len(dvys) == 1:
                dvx = dvxs.pop()
                dvy = dvys.pop()
            else:
                # No "right" answer for what binning is for file as a whole
                dvx = None
                dvy = None

    # By now, we have normalized to single-valued binnings
    if (dvx is not None) and (dvy is not None):
        desc_dict["detector_binning"] = "%dx%d" % (dvx, dvy) if dvx is not None and dvy is not None else None
    else:
        desc_dict["detector_binning"] = None

    return desc_dict

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

    rq_events = []
    for ad in inputs:
        log.debug("Received calibration request for {}".format(ad.filename))
        rq = CalibrationRequest(ad, caltype, procmode)
        rq.descriptors = get_descriptors_dict(ad)
        rq_events.append(rq)
    return rq_events


def _handle_returns(dv):
    # This turns all namedtuple objects into simple tuples so that the str()
    # method drops the name of the class and they can be interpreted without
    # needing that class to be defined at the other end of the transportation
    # TODO: 4/22/2021: Coercing to lists to ensure functionality with
    #  existing FitsStorage code
    if dv is None:
        return dv
    if isinstance(dv, list) and isinstance(dv[0], tuple):
        return [list(el) if el is not None else list() for el in dv]
    elif isinstance(dv, tuple):
        return list(dv)
    return dv
