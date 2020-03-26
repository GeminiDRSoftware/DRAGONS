#
#                                                                        DRAGONS
#
#                                                               calrequestlib.py
# ------------------------------------------------------------------------------
import hashlib

from os import mkdir
from os.path import basename, exists
from os.path import join, split

from urllib.parse import urlparse

from gempy.utils import logutils

from geminidr import set_caches
from recipe_system.cal_service import cal_search_factory, handle_returns_factory
from .file_getter import get_file_iterator, GetterError
# ------------------------------------------------------------------------------
log = logutils.get_logger(__name__)
# ------------------------------------------------------------------------------
# Currently delivers transport_request.calibration_search fn.
calibration_search = cal_search_factory()
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


def _check_cache(cname, ctype):
    cachedir = _makecachedir(ctype)
    cachename = join(cachedir, cname)
    if exists(cachename):
        return cachename, cachedir
    return None, cachedir


def _makecachedir(caltype):
    cache = set_caches()
    cachedir = join(cache["calibrations"], caltype)
    if not exists(cachedir):
        mkdir(cachedir)
    return cachedir


class CalibrationRequest:
    """
    Request objects are passed to a calibration_search() function

    """
    def __init__(self, ad, caltype=None):
        self.ad = ad
        self.caltype = caltype
        self.datalabel = ad.data_label()
        self.descriptors = None
        self.filename = ad.filename
        self.tags = ad.tags

    def as_dict(self):
        retd = {}
        retd.update(
            {'filename'   : self.filename,
             'caltype'    : self.caltype,
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


def get_cal_requests(inputs, caltype):
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

    _handle_returns = handle_returns_factory()

    rq_events = []
    for ad in inputs:
        log.stdinfo("Received calibration request for {}".format(ad.filename))
        rq = CalibrationRequest(ad, caltype)
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


def process_cal_requests(cal_requests, howmany=None):
    """
    Conduct a search for calibration files for the passed list of calibration
    requests. This passes the requests to the calibration_search() function,
    and then examines the search results to see if a matching file, if any,
    is cached. If not, then the calibration file is retrieved from the
    archive.

    If a calibration match is found by the calibration manager, a URL is
    returned. This function will perform a cache inspection to see if the
    matched calibraiton file is already present. If not, the calibration
    will be downloaded and written to the cache. It is this path that is
    returned in the dictionary structure. A path of 'None' indicates that no
    calibration match was found.

    Parameters
    ----------
    cal_requests : <list>
        A list of CalibrationRequest objects

    howmany : <int>, optional
        Maximum number of calibrations to return per request
        (not passed to the server-side request but trims the returned list)

    Returns
    -------
    calibration_records : <dict>
        A set of science frames and matching calibrations.

    Example
    -------
    The returned dictionary has the form::

        {(ad): <filename_of_calibration_including_path>,
            ...
        }

    """
    calibration_records = {}
    warn = "\tNo {} calibration file found for {}\n"
    md5msg = "\tNo {} calibration found for {}\n"

    def _add_cal_record(rq, calfile):
        calibration_records.update({rq.ad: calfile})
        return

    cache = set_caches()
    for rq in cal_requests:
        calname = None
        calmd5 = None
        calurl = None
        calurl, calmd5 = calibration_search(rq, howmany=(howmany if howmany else 1))
        if not calurl:
            log.warning("START CALIBRATION SERVICE REPORT\n")
            if not calmd5:
                log.warning(md5msg.format(rq.caltype, rq.filename))
            else:
                log.warning("\t{}".format(calmd5))
                log.warning(warn.format(rq.caltype, rq.filename))

            log.warning("END CALIBRATION SERVICE REPORT\n")
            continue

        calibs = []
        for url, md5 in zip(calurl, calmd5):
            log.info("Found calibration (url): {}".format(url))
            components = urlparse(url)
            calname = basename(components.path)
            cachename, cachedir = _check_cache(calname, rq.caltype)
            if cachename:
                cached_md5 = generate_md5_digest(cachename)
                if cached_md5 == md5:
                    log.stdinfo("Cached calibration {} matched.".format(cachename))
                    calibs.append(cachename)
                else:
                    log.stdinfo("File {} is cached but".format(calname))
                    log.stdinfo("md5 checksums DO NOT MATCH")
                    log.stdinfo("Making request on calibration service")
                    log.stdinfo("Requesting URL {}".format(url))
                    try:
                        calname = get_request(url, cachename)
                        calibs.append(cachename)
                    except GetterError as err:
                        for message in err.messages:
                            log.error(message)
                continue

            log.status("Making request for {}".format(url))
            fname = split(url)[1]
            calname = join(cachedir, fname)
            try:
                calname = get_request(url, calname)
            except GetterError as err:
                for message in err.messages:
                    log.error(message)
            else:
                # hash compare
                download_mdf5 = generate_md5_digest(calname)
                if download_mdf5 == md5:
                    log.status("MD5 hash match. Download OK.")
                    calibs.append(calname)
                else:
                    err = "MD5 hash of downloaded file does not match expected hash {}"
                    raise IOError(err.format(md5))

        # If howmany=None, append the only file as a string, instead of the list
        if calibs:
            _add_cal_record(rq, calibs if howmany else calibs[0])

    return calibration_records
