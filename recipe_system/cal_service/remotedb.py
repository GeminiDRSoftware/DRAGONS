# Defines the RemoteDB class for calibration returns. This is a high-level
# interface to FITSstore. It may be subclassed in future

from os import path
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
    def __init__(self, server, name=None, valid_caltypes=None, get=True,
                 store=True, log=None):
        if name is None:
            name = server
        super().__init__(name=name, get=get, store=store, log=log,
                         valid_caltypes=valid_caltypes)
        # TODO: we want to make an MDF a full calibration, but currently
        # we can only retrieve it from a remote location, so this handles that
        if valid_caltypes is None:
            self._valid_caltypes.append("mask")
        if not server.startswith("http"):  # allow https://
            server = f"http://{server}"
        self.server = server
        self._calmgr = f"{self.server}/calmgr"
        self._proccal_url = f"{self.server}/upload_processed_cal"
        self._science_url = f"{self.server}/upload_file"

    def _get_calibrations(self, adinputs, caltype=None, procmode=None):
        log = self.log
        cal_requests = get_cal_requests(adinputs, caltype, procmode=procmode,
                                        is_local=False)
        cals = []
        for rq in cal_requests:
            procstr = "" if procmode is None else f"/{procmode}"
            rqurl = f"{self._calmgr}/{rq.caltype}{procstr}/{rq.filename}"
            log.stdinfo(f"CENTRAL CALIBRATION SEARCH: {rqurl}")
            calurl, calmd5 = retrieve_calibration(rqurl, rq)
            if not calurl:
                log.warning("START CALIBRATION SERVICE REPORT\n")
                log.warning(f"\t{calmd5}")
                log.warning(f"No {rq.caltype} found for {rq.filename}")
                log.warning("END CALIBRATION SERVICE REPORT\n")
                cals.append(None)
                continue
            self.log.info(f"Found calibration (url): {calurl}")
            calname = path.basename(urllib.parse.urlparse(calurl).path)
            cachefile = path.join(self.caldir, rq.caltype, calname)
            if path.exists(cachefile):
                cached_md5 = generate_md5_digest(cachefile)
                if cached_md5 == calmd5:
                    log.stdinfo(f"Cached calibration {cachefile} matched.")
                    cals.append(cachefile)
                    continue
                else:
                    log.stdinfo(f"File {calname} is cached but")
                    log.stdinfo("md5 checksums DO NOT MATCH")
                    log.stdinfo("Making request on calibration service")

            log.stdinfo("Making request for {url}")
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
                cals.append(cachefile)
            else:
                raise OSError("MD5 hash of downloaded file does not match "
                              f"expected hash {calmd5}")

        return CalReturn([None if cal is None else (cal, self.name)
                          for cal in cals])


    def _store_calibration(self, cal, caltype=None):
        """Store calibration. If this is a processed_science, cal should be
        an AstroData object, otherwise it should be a filename"""
        assert isinstance(cal, str) ^ ("science" in caltype)
        if "science" in caltype:
            # Write to a stream in memory, not to disk
            f = BytesIO()
            cal.write(f)
            postdata = f.getvalue()
            url = f"{self._science_url}/{cal.filename}"
        else:
            postdata = open(cal, "rb").read()
            url = f"{self._proccal_url}/{cal}"

        try:
            rq = urllib.request.Request(url)
            rq.add_header('Content-Length', '%d' % len(postdata))
            rq.add_header('Content-Type', 'application/octet-stream')
            rq.add_header('Cookie', f"gemini_fits_upload_auth={UPLOADCOOKIE}")
            u = urllib.request.urlopen(rq, postdata)
            response = u.read()
            self.log.stdinfo(f"{url} uploaded OK.")
        except urllib.error.HTTPError as error:
            self.log.error(str(error))
            raise


def retrieve_calibration(rqurl, rq):
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
        # Simplified for howmany=1 only
        calurlel = dom.getElementsByTagName("url")[0].childNodes[0].data
        calurlmd5 = dom.getElementsByTagName("md5")[0].childNodes[0].data
    except IndexError:
        return None, preerr

    return calurlel, calurlmd5
