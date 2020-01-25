from __future__ import division
from __future__ import print_function
#
#                                                                        DRAGONS
#
#                                                                  http_proxy.py
# ------------------------------------------------------------------------------
from future import standard_library
standard_library.install_aliases()
from builtins import str
from past.utils import old_div

import os
import sys
import json
import time
import select
import datetime

import urllib.error
import urllib.parse
import urllib.request

from socketserver import ThreadingMixIn
from http.server import BaseHTTPRequestHandler, HTTPServer

from recipe_system.cal_service import calurl_dict
# ------------------------------------------------------------------------------
#
# Global bits for logging
REQMSG = "Requesting current OP day events "
RECMSG = "Received {} events."
FAILMSG = "Failed to access Fitsstore. No metrics available."
msg_form  = '"%s" %s %s'
info_code = 203
fail_code = 416
no_access_code = 503
size = "-"

# ------------------------------------------------------------------------------
def parsepath(path):
    """
    parsepath w/ urlparse.

    parameters: <string>
    return:     <dict>

    """
    rparms = {}
    parsed_url = urllib.parse.urlparse(path)
    rparms.update({"path": parsed_url.path})
    rparms.update({"query": parsed_url.query})
    rparms.update(urllib.parse.parse_qs(parsed_url.query))
    return rparms

# ------------------------------------------------------------------------------
#                                Timing functions
def server_time():
    """
    Return a dictionary of server timing quantities related to current time.
    This dict will be returned to a call on the server, /rqsite.json (See
    do_GET() method of ADCCHandler class.

    parameters: <void>
    return:     <dict>, dictionary of time now values.

    """
    lt_now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    utc_now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
    utc_offset = datetime.datetime.utcnow() - datetime.datetime.now()

    if utc_offset.days != 0:
        utc_offset = -utc_offset
        utc_offset = -int(round(old_div(utc_offset.seconds,3600.)))
    else:
        utc_offset = int(round(old_div(utc_offset.seconds,3600.)))

    timezone = old_div(time.timezone, 3600)
    if timezone == 10:
        local_site = 'gemini-north'
    elif timezone in [3, 4]:   # TZ -4 but +1hr DST applied inconsistently
        local_site = 'gemini-south'
    else:
        local_site = 'remote'

    time_dict = {"local_site": local_site,
                 "tzname"    : time.tzname[0],
                 "lt_now"    : lt_now,
                 "utc_now"   : utc_now,
                 "utc_offset": utc_offset}
    return time_dict

def stamp_to_ymd(timestamp):
    """
    Caller sends a timestamp in seconds of epoch. Return string for
    year month day of that time as YYYYMMDD' as used by url requests, as in
    http://<fitsstore_server>/qaforgui/20130616

    parameters: <float>,  seconds of epochs.
    return:     <string>, YYYYMMDD of passed time.

    """
    return time.strftime("%Y%m%d", time.localtime(timestamp))

def stamp_to_opday(timestamp):
    """
    Converts a passed time stamp (sec) into the corresponding operational
    day. I.e. timestamps >= 14.00h are the next operational day.

    parameters: <float>, time in epoch seconds
    return:     <string>, YYYYMMDD

    """
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    if dt_object.hour >= 14:
        timestamp = timestamp + 86400
    return  stamp_to_ymd(timestamp)

def ymd_to_stamp(yy, mm, dd, hh=0):
    """
    Caller passes integers for year, month, and day. Return is
    the epoch time (sec). Year is 4 digit, eg., 2013

    parameters: <int>, <int>, <int> [, <int>] Year, Month, Day [,Hour]
    return:     <float>, epoch time in seconds.

    """
    ymd = "{} {} {} {}".format(yy, mm, dd, hh)
    return time.mktime(time.strptime(ymd, "%Y %m %d %H"))

def current_op_timestamp():
    """
    Return the epoch time (sec) of the start of current operational day,
    where turnover occurs @ 14.00h localtime. I.e. if the hour >= 14.00,
    then the current operational day is tomorrow.

    Eg., 2013-08-02 17.00h is 20130803

    parameters: <void>
    return:     <float>

    """
    hh = 14
    tnow = datetime.datetime.now()
    t_epoch = time.time()

    if tnow.hour >= 14.0:
        op_day = stamp_to_ymd(t_epoch)
    else:
        op_day = stamp_to_ymd(t_epoch - 86400)

    yy, mm, dd = op_day[:4], op_day[4:6], op_day[6:]
    timestamp = ymd_to_stamp(yy, mm, dd, hh)
    return timestamp

#                             END Timing functions
# ------------------------------------------------------------------------------
#   FITS Store query.
def fstore_get(timestamp):
    """
    Open a url on fitsstore/qaforgui/ with the passed timestamp.
    timestamp is in epoch seconds, which is converted here to a
    YMD string for the URL.  Return a list of dicts of qa metrics data.

    Exceptions on urlopen()
    -----------------------
    Any number of exceptions may be thrown on URL access: URLError, HTTPError,
    TimeoutError, ... . We don't really care which specific failure occurred,
    only that QA metrics are not acessible. Here, we catch all failures and
    simply pass, returning a empty list.

    N.B. -- A timestamp that evaluates to False will request everything
    from fitsstore. This could be huge. Be careful passing no timestamp!

    Parameters
    ----------
    timestamp : <float>, time in epoch seconds

    Return
    ------
    qa_data : <list>, list of dicts (json) of qametrics

    """
    qa_data = list()

    # Get the fitsstore query url from calurl_dict
    fitsstore_qa = calurl_dict.calurl_dict['QAQUERYURL']
    if not timestamp:
        try:
            store_handle = urllib.request.urlopen(furl)
            qa_data  = json.loads(store_handle.read())
        except Exception:
            self.log_message(msg_form, FAILMSG, no_access_code, size)
            pass
    else:
        date_query = stamp_to_opday(timestamp)
        furl = os.path.join(fitsstore_qa, date_query)
        try:
            store_handle = urllib.request.urlopen(furl)
            qa_data = json.loads(store_handle.read())
        except Exception:
            self.log_message(msg_form, FAILMSG, no_access_code, size)
            pass

    return qa_data

# ------------------------------------------------------------------------------
class ADCCHandler(BaseHTTPRequestHandler):
    """
    ADCC services request handler.

    """
    events = None
    informers  = None

    def address_string(self):
        host, port = self.client_address[:2]
        return host

    def log_request(self, code='-', size='-'):
        """Log an accepted request.

        This is called by send_response().

        This overrides BaseHTTPRequestHandler.log_request.
        See that class for what the method does normally.

        """
        msg_form  = '"%s" %s %s'
        try:
            assert self.informers["verbose"]
            self.log_message(msg_form, repr(self.requestline), code, size)
        except AssertionError:
            if "cmdqueue.json" in self.requestline:
                pass
            else:
                self.log_message(msg_form, repr(self.requestline), code, size)
        return

    def do_GET(self):
        """
        Defined services on HTTP GET requests.

        """
        events = self.informers["events"]
        self.informers["verbose"] = True
        dark_theme = self.informers['dark']
        parms = parsepath(self.path)
        try:
            # First test for an html request on the QAP nighttime_metrics page.
            # I.e.  <localhost>:<port>/qap/nighttime_metrics.html
            if self.path.startswith("/qap"):
                if ".." in self.path:
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    data = "<b>bad path error</b>"
                    self.wfile.write(bytes(data.encode('utf-8')))

                dirname = os.path.dirname(__file__)
                if dark_theme:
                    joinlist = [dirname, "../client/adcc_faceplate_dark/"]
                else:
                    joinlist = [dirname, "../client/adcc_faceplate/"]

                # Split out any parameters in the URL
                self.path = self.path.split("?")[0]

                #append any further directory info.
                joinlist.append(self.path[5:])
                fname = os.path.join(*joinlist)
                self.log_message('{} {} {}'.format("Loading "+joinlist[1]+
                                            os.path.basename(fname), 203, '-'))
                try:
                    with open(fname, 'rb') as f:
                        data = f.read()
                except IOError:
                    data = bytes("<b>NO SUCH RESOURCE AVAILABLE</b>".encode('utf-8'))

                self.send_response(200)
                if  self.path.endswith(".js"):
                    self.send_header('Content-type', 'text/javascript')
                elif self.path.endswith(".css"):
                    self.send_header("Content-type", "text/css")
                elif fname.endswith(".png"):
                    self.send_header('Content-type', "image/png")
                else:
                    self.send_header('Content-type', 'text/html')

                self.end_headers()
                self.wfile.write(data)
                return

            # ------------------------------------------------------------------
            # The vast majority of HTTP client GET requests will be on the
            # cmdqueue service. Handle first.
            if parms["path"].startswith("/cmdqueue.json"):
                self._handle_cmdqueue_json(events, parms)

            # ------------------------------------------------------------------
            # Server time
            # Queried by metrics client
            elif parms["path"].startswith("/rqsite.json"):
                self.send_response(200)
                self.send_header('Content-type', "application/json")
                self.end_headers()
                tdic = server_time()
                self.wfile.write(
                    bytes(json.dumps(tdic, sort_keys=True, indent=4).encode('utf-8'))
                )

           # ------------------------------------------------------------------
           # Queried by metrics client
            elif parms["path"].startswith("/rqlog.json"):
                self.send_response(200)
                self.send_header('Content-type', "application/json")
                self.end_headers()
                if "file" in parms:
                    logfile = parms["file"][0]
                    if not os.path.exists(logfile):
                        msg = "Log file not available"
                    else:
                        f = open(logfile, "r")
                        msg = f.read()
                        f.close()
                else:
                    msg = "No log file available"

                tdic = {"log": msg}
                self.wfile.write(
                    bytes(json.dumps(tdic, sort_keys=True, indent=4).encode('utf-8'))
                    )
        except IOError:
            self.send_error(404,'File Not Found: %s' % self.path)
            raise
        return

    def do_POST(self):
        info_code = 203
        size = "-"
        events = self.informers["events"]
        parms  = parsepath(self.path)
        vlen   = int(self.headers["Content-Length"])
        pdict  = self.rfile.read(vlen)

        if parms["path"].startswith("/event_report"):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            aevent = json.loads(pdict)
            events.append_event(aevent)
            self.log_message('"%s" %s %s', "Appended event", info_code, size)
            self.log_message('"%s" %s %s', repr(aevent), info_code, size)

        return
    # -------------------------------------------------------------------------
    # privitized handling cmdqueue.json requests

    def _handle_cmdqueue_json(self, events, parms):
        """
        Handle HTTP client GET requests on service: cmdqueue.json

        """
        verbosity = self.informers["verbose"]

        self.send_response(200)
        self.send_header('Content-type', "application/json")
        self.end_headers()

        # N.B. A timestamp of zero will request *everything* from fitsstore
        # This could be huge. Be careful passing GET request on cmdqueue.json
        # with no timestamp.

        if "timestamp" in parms:
            fromtime = float(parms["timestamp"][0])
        else:
            fromtime = 0

        # event_list = [] implies a new adcc. Request current op day
        # metrics from fitsstore.

        if not events.event_list:
            self.log_message(msg_form, "No extant events.", info_code, size)
            self.log_message(msg_form, REQMSG+"@fitsstore", info_code, size)

            events.event_list = fstore_get(current_op_timestamp())
            tdic = events.get_list()
            self.log_message(msg_form,RECMSG.format(len(tdic)),info_code, size)
            tdic.insert(0, {"msgtype": "cmdqueue.request","timestamp": time.time()})
            tdic.append({"msgtype": "cmdqueue.request", "timestamp": time.time()})
            self.wfile.write(
                bytes(json.dumps(tdic, sort_keys=True, indent=4).encode('utf-8'))
            )

        # Handle current nighttime requests ...
        elif stamp_to_opday(fromtime) == stamp_to_opday(current_op_timestamp()):
            if verbosity:
                self.log_message(msg_form, REQMSG+stamp_to_opday(fromtime),info_code,size)

            tdic = events.get_list(fromtime=fromtime)
            tdic.insert(0, {"msgtype":"cmdqueue.request","timestamp": time.time()})
            self.wfile.write(
                bytes(json.dumps(tdic, sort_keys=True, indent=4).encode('utf-8'))
                )

        # Handle previous day requests
        elif fromtime < current_op_timestamp():
            if verbosity:
                self.log_message(msg_form, "Requested metrics on ... " +
                                 stamp_to_opday(fromtime), info_code, size)

            tdic = fstore_get(fromtime)
            if verbosity:
                self.log_message(msg_form, "Received " + str(len(tdic)) +
                                 " events from fitsstore.", info_code, size)

            # Append the last timestamp from the event_list. This is done
            # to trigger the client to pinging the adcc from the last
            # recorded event.
            tdic.insert(0, {"msgtype": "cmdqueue.request", "timestamp": time.time()})
            tdic.append({"msgtype": "cmdqueue.request", "timestamp": time.time()})
            self.wfile.write(
                bytes(json.dumps(tdic, sort_keys=True, indent=4).encode('utf-8'))
            )

        # Cannot handle the future ...
        else:
            self.log_message(msg_form, "Invalid timestamp received.",fail_code, size)
            self.log_message(msg_form, "Future events not known.", fail_code, size)

        return


class MTHTTPServer(ThreadingMixIn, HTTPServer):
    """Handles requests using threads"""


def startInterfaceServer(*args, **informers):
    import socket
    run_event = args[0]
    port = informers['port']
    ADCCHandler.informers = informers
    findingPort = True
    while findingPort:
        try:
            print("Starting HTTP server on port %s ... " % str(port))
            server = MTHTTPServer(('', port), ADCCHandler)
            findingPort = False
        except socket.error:
            print("failed, port taken")
            port += 1

    print("Started  HTTP server on port %s" % str(port))
    while run_event.is_set():
        r, w, x = select.select([server.socket], [], [], .5)
        if r:
            server.handle_request()

    print("http_proxy: received signal 'clear'. Shutting down proxy server ...")
    server.socket.close()
    return

main = startInterfaceServer
