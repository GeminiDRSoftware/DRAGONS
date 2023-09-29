# Defines the RemoteDB class for calibration returns. This is a high-level
# interface to FITSstore. It may be subclassed in future

from os import path, makedirs
from io import BytesIO
from pprint  import pformat
from xml.dom import minidom

import urllib.request
import urllib.parse
import urllib.error

from .caldb import CalDB, CalReturn
from .calrequestlib import get_cal_requests, generate_md5_digest, get_request
from .calrequestlib import GetterError

UPLOADCOOKIE = "qap_upload_processed_cal_ok"

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


class RemoteDB(CalDB):
    """
    The class for remote calibration databases. It inherits from CalDB, but
    also has the following attributes:

    Attributes
    ----------
    server : str
        URL of the server
    store_science : bool
        whether processed science images should be uploaded
    _upload_cookie : str
        the cookie to send when uploading files
    _calmgr : str
        the URL for making requests to the remote calibration manager
    _proccal_url, _science_url : str
        the URLs for uploading processed calibrations and processed science
        images, respectively.
    """
    def __init__(self, server, name=None, valid_caltypes=None, get_cal=True,
                 store_cal=False, store_science=False, procmode=None, log=None,
                 upload_cookie=None):
        if name is None:
            name = server
        super().__init__(name=name, get_cal=get_cal, store_cal=store_cal,
                         log=log, valid_caltypes=valid_caltypes,
                         procmode=procmode)
        self.store_science = store_science
        if not server.startswith("http"):  # allow https://
            server = f"http://{server}"
        self.server = server
        self._calmgr = f"{self.server}/calmgr"
        self._proccal_url = f"{self.server}/upload_processed_cal"
        self._science_url = f"{self.server}/upload_file"
        self._upload_cookie = upload_cookie or UPLOADCOOKIE

    def _get_calibrations(self, adinputs, caltype=None, procmode=None,
                          howmany=1):
        log = self.log
        cal_requests = get_cal_requests(adinputs, caltype, procmode=procmode,
                                        is_local=False)
        cals = []
        for rq in cal_requests:
            procstr = "" if procmode is None else f"/{procmode}"
            rqurl = f"{self._calmgr}/{rq.caltype}{procstr}/{rq.filename}"
            log.stdinfo(f"Querying remote database: {rqurl}")
            remote_cals = retrieve_calibration(rqurl, rq, howmany=howmany)
            if not remote_cals[0]:
                log.warning("START CALIBRATION SERVICE REPORT\n")
                if remote_cals[1]:
                    log.warning(f"\t{remote_cals[1]}")
                log.warning(f"No {rq.caltype} found for {rq.filename}")
                log.warning("END CALIBRATION SERVICE REPORT\n")
                cals.append(None)
                continue

            good_cals = []
            caldir = path.join(self.caldir, rq.caltype)
            for calurl, calmd5 in zip(*remote_cals):
                log.stdinfo(f"Found calibration (url): {calurl}")
                calname = path.basename(urllib.parse.urlparse(calurl).path)
                cachefile = path.join(caldir, calname)
                if path.exists(cachefile):
                    cached_md5 = generate_md5_digest(cachefile)
                    if cached_md5 == calmd5:
                        log.stdinfo(f"Cached calibration {cachefile} matched.")
                        good_cals.append(cachefile)
                        continue
                    else:
                        log.stdinfo(f"File {calname} is cached but")
                        log.stdinfo("md5 checksums DO NOT MATCH")

                log.stdinfo(f"Making request for {calurl}")
                if not path.exists(caldir):
                    makedirs(caldir)
                try:
                    get_request(calurl, cachefile)
                except GetterError as err:
                    for message in err.messages:
                        log.error(message)
                        cals.append(None)
                    continue
                download_mdf5 = generate_md5_digest(cachefile)
                if download_mdf5 == calmd5:
                    log.status("MD5 hash match. Download OK.")
                    good_cals.append(cachefile)
                else:
                    raise OSError("MD5 hash of downloaded file does not match "
                                  f"expected hash {calmd5}")
            # Append list if >1 requested, else just the filename string
            if good_cals:
                cals.append(good_cals if howmany != 1 else good_cals[0])
            else:
                cals.append(None)

        return CalReturn([None if cal is None else (cal, self.name)
                          for cal in cals])

    def _store_calibration(self, cal, caltype=None):
        """Store calibration. If this is a processed_science, cal should be
        an AstroData object, otherwise it should be a filename"""
        is_science = caltype is not None and "science" in caltype
        if not ((is_science and self.store_science) or
                (not is_science and self.store_cal)):
            self.log.stdinfo(f"{self.name}: NOT storing {cal} as {caltype}")
            return

        assert isinstance(cal, str) ^ is_science
        self.log.stdinfo(f"{self.name}: Storing {cal} as {caltype}")
        if "science" in caltype:
            # Write to a stream in memory, not to disk
            f = BytesIO()
            cal.write(f)
            postdata = f.getvalue()
            url = f"{self._science_url}/{cal.filename}"
        else:
            postdata = open(cal, "rb").read()
            url = f"{self._proccal_url}/{path.basename(cal)}"

        try:
            rq = urllib.request.Request(url)
            rq.add_header('Content-Length', '%d' % len(postdata))
            rq.add_header('Content-Type', 'application/octet-stream')
            rq.add_header('Cookie', "gemini_fits_upload_auth="
                                    f"{self._upload_cookie}")
            u = urllib.request.urlopen(rq, postdata)
            response = u.read()
            self.log.stdinfo(f"{url} uploaded OK.")
        except urllib.error.HTTPError as error:
            self.log.error(str(error))
            raise


def retrieve_calibration(rqurl, rq, howmany=1):
    sequence = [("descriptors", rq.descriptors), ("types", rq.tags)]
    postdata = urllib.parse.urlencode(sequence).encode('utf-8')
    try:
        calrq = urllib.request.Request(rqurl)
        u = urllib.request.urlopen(calrq, postdata)
        response = u.read()
    except (urllib.error.HTTPError, urllib.error.URLError) as err:
        return None, str(err)

    desc_nones = [k for k, v in rq.descriptors.items() if v is None]
    preerr = RESPONSESTR % {"sequence": pformat(sequence),
                            "response": response.strip(),
                            "nones"   : ", ".join(desc_nones) \
                            if len(desc_nones) > 0 else "No Nones Sent"}
    try:
        dom = minidom.parseString(response)
        calurlel = [d.childNodes[0].data
                    for d in dom.getElementsByTagName('url')[:howmany]]
        calurlmd5 = [d.childNodes[0].data
                     for d in dom.getElementsByTagName('md5')[:howmany]]
    except IndexError:
        return None, preerr

    return calurlel, calurlmd5
