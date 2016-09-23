#
#                                                                     QAP Gemini
#
#                                                                  http_proxy.py
#                                                                        07-2013
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
#
# This has been modified to make queries on fitstore qaforgui urls.
# 
# Updated parsepath() w/ urlparse.

import os
import json
import time
import select
import urllib2
import urlparse
import datetime
import subprocess

from SocketServer import ThreadingMixIn
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

from astrodata import AstroData
from astrodata.utils.Lookups import get_lookup_table

# ------------------------------------------------------------------------------
def parsepath(path):
    """A better parsepath w/ urlparse.

    parameters: <string>
    return:     <dict>
    """
    rparms = {}
    parsed_url = urlparse.urlparse(path)
    rparms.update({"path": parsed_url.path})
    rparms.update({"query": parsed_url.query})
    rparms.update(urlparse.parse_qs(parsed_url.query))
    return rparms

# ------------------------------------------------------------------------------
#                                Timing functions
def server_time():
    """Return a dictionary of server timing quantities related to current time.
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
        utc_offset = -int(round(utc_offset.seconds/3600.))
    else:
        utc_offset = int(round(utc_offset.seconds/3600.))
        
    timezone = time.timezone / 3600
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
    """Caller sends a timestamp in seconds of epoch. Return string for
    year month day of that time as YYYYMMDD' as used by url requests, as in
    http://<fitsstore_server>/qaforgui/20130616

    parameters: <float>,  seconds of epochs.
    return:     <string>, YYYYMMDD of passed time.
    """
    return time.strftime("%Y%m%d", time.localtime(timestamp))


def stamp_to_opday(timestamp):
    """Converts a passed time stamp (sec) into the corresponding operational
    day. I.e. timestamps >= 14.00h are the next operational day.

    parameters: <float>, time in epoch seconds
    return:     <string>, YYYYMMDD
    """
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    if dt_object.hour >= 14:
        timestamp = timestamp + 86400
    return  stamp_to_ymd(timestamp)


def ymd_to_stamp(yy, mm, dd, hh=0):
    """Caller passes integers for year, month, and day. Return is
    the epoch time (sec). Year is 4 digit, eg., 2013

    parameters: <int>, <int>, <int> [, <int>] Year, Month, Day [,Hour]
    return:     <float>, epoch time in seconds.
    """
    return time.mktime(time.strptime("%s %s %s %s" % (yy, mm, dd, hh), 
                                     "%Y %m %d %H"))


def current_op_timestamp():
    """Return the epoch time (sec) of the start of current operational day,
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


#                            End Timing functions
# ------------------------------------------------------------------------------
#   FITS Store query.
def fstore_get(timestamp):
    """Open a url on fitsstore/qaforgui/ with the passed timestamp.
    timestamp is in epoch seconds, which is converted here to a 
    YMD string for the URL.  Return a list of dicts of qa metrics data.

    N.B. A timestamp that evaluates to False (0, None) will request everything 
    from fitsstore. This could be huge. Be careful passing no timestamp!

    parameters: <float>, time in epoch seconds
    return:     <list>,  list of dicts (json) of qametrics
    """
    # Get the fitsstore query url from calurl_dict
    qurlPath     = "Gemini/calurl_dict"
    fitsstore_qa = get_lookup_table(qurlPath, "calurl_dict")['QAQUERYURL']

    if not timestamp:
        furl         = os.path.join(fitsstore_qa)
        store_handle = urllib2.urlopen(furl)
        qa_data      = json.loads(store_handle.read())   
    else:
        date_query    = stamp_to_opday(timestamp)
        furl          = os.path.join(fitsstore_qa, date_query)
        store_handle  = urllib2.urlopen(furl)
        qa_data       = json.loads(store_handle.read())
    return qa_data

# ------------------------------------------------------------------------------
webserverdone = False
# ------------------------------------------------------------------------------
class ADCCHandler(BaseHTTPRequestHandler):
    """ADCC services request handler.
    """
    informers  = None
    dataSpider = None
    dirdict    = None
    state      = None
    counter    = 0
    stamp_register = []

    def address_string(self):
        host, port = self.client_address[:2]
        return host

    def log_request(self, code='-', size='-'):
        """Log an accepted request.

        This is called by send_response().

        This is an override of BaseHTTPRequestHandler.log_request method.
        See that class for what the method does normally.
        """
        try:
            assert (self.informers["verbose"])
            self.log_message('"%s" %s %s', self.requestline, 
                             str(code), str(size))
        except AssertionError:
            if "cmdqueue.json" in self.requestline:
                pass
            else:
                self.log_message('"%s" %s %s', self.requestline, 
                                 str(code), str(size))
        return

    def do_GET(self):
        """Defined ADCC services on GET requests."""
        global webserverdone
        rim = self.informers["rim"]
        parms = parsepath(self.path)

        # Older revisions of adcc may not supply 'verbose' key
        try: 
            self.informers["verbose"]
        except KeyError: 
            self.informers["verbose"] = True

        try:
            # First test for an html request on the QAP nighttime_metrics 
            # page. 
            # I.e.  <localhost>:<port>/qap/nighttime_metrics.html
            if self.path.startswith("/qap"):
                if ".." in self.path:
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    data = "<b>bad path error</b>"
                    self.wfile.write(data)
                dirname = os.path.dirname(__file__)
                joinlist = [dirname, "../client/adcc_faceplate/"]
                
                # Split out any parameters in the URL
                self.path = self.path.split("?")[0]

                #append any further directory info.
                joinlist.append(self.path[5:])
                fname = os.path.join(*joinlist)
                self.log_message('"%s" %s %s', "Loading " + \
                                 joinlist[1] + os.path.basename(fname), 
                                 203, '-')
                try:
                    f = open(fname, "r")
                    data = f.read()
                    f.close()
                except IOError:
                    data = "<b>NO SUCH RESOURCE AVAILABLE</b>"

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
                self._handle_cmdqueue_json(rim, parms)

            # ------------------------------------------------------------------
            # Server time
            # Queried by metrics client
            elif parms["path"].startswith("/rqsite.json"):
                self.send_response(200)
                self.send_header('Content-type', "application/json")
                self.end_headers()
                tdic = server_time()
                self.wfile.write(json.dumps(tdic, sort_keys=True, indent=4))

           # ------------------------------------------------------------------
           # Queried by metrics client
            elif parms["path"].startswith("/rqlog.json"):
                self.send_response(200)
                self.send_header('Content-type', "application/json")
                self.end_headers()
                
                if "file" in parms:
                    logfile = parms["file"][0]
                    print logfile
                    if not os.path.exists(logfile):
                        msg = "Log file not available"
                    else:
                        f = open(logfile, "r")      
                        msg = f.read()
                        f.close()
                else:
                    msg = "No log file available"

                tdic = {"log":msg}

                self.wfile.write(json.dumps(tdic, sort_keys=True, indent=4))

            # ------------------------------------------------------------------
            elif parms["path"] == "/":
                page = """
                <html>
                <head>
                </head>
                <body>
                <h4>prsproxy engineering interface</h4>
                <ul>
                <li><a href="/engineering">Engeering Interface</a></li>
                <li><a href="qap/engineering.html">Engeering AJAX App</a></li>
                <li><a href="datadir.xml">Data Directory View</a></li>
                <li><a href="killprs">Kill this server</a> (%(numinsts)d """ +\
                    """copies of reduce registered)</li>
                </ul>
                <body>
                </html>"""
                self.send_response(200)
                self.send_header("content-type", "text-html")
                self.end_headers()
                self.wfile.write(page % {"numinsts":rim.numinsts})
 
            # ------------------------------------------------------------------
            elif self.path.startswith("/engineering"):
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                if "rim" in ADCCHandler.informers:
                    rim = ADCCHandler.informers["rim"]
                    evman = rim.events_manager
                    import pprint
                    data = "<u>Events</u><br/><pre>"
                    data += "num events: %d\n" % len(evman.event_list)
                    for mevent in evman.event_list:
                        data += pprint.pformat(mevent)
                        data += "\n------------------------------\n"
                    data += "</pre>"
                    self.wfile.write(data)
        except IOError:
            print "Caught IOError"
            self.send_error(404,'File Not Found: %s' % self.path)
            raise
        return


    def do_POST(self):
        """Defined ADCC services on POST requests."""
        global webserverdone
        parms = parsepath(self.path)
        vlen = int(self.headers["Content-Length"])
        head = self.rfile.read(vlen)
        pdict = head
        
        if parms["path"].startswith("/runreduce"):
            # Get events manager
            evman = None
            if "rim" in ADCCHandler.informers:
                rim = ADCCHandler.informers["rim"]
                evman = rim.events_manager

            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            
            reduce_params = json.loads(pdict)

            # get() returns None if no key
            fp  = reduce_params.get("filepath")
            opt = reduce_params.get("options")
            prm = reduce_params.get("parameters")
            cmdlist = ["reduce", "--invoked"]

            # reduce_beta is the old reduce, deprecated @Rev4949
            #cmdlist = ["reduce_beta", "--invoked"]

            # cmdlist built for reduce2 parameters (-p) field, i.e. no commas
            if prm is not None:
                cmdlist.extend(["-p"])
                cmdlist.extend([str(key)+"="+str(val) for key,val in prm.items()])

            if opt is not None:
                for key,val in opt.items():
                    cmdlist.extend(["--"+str(key), str(val)])

            # build for reduce_beat parameters (-p) field, i.e. w/ commas
            # if prm is not None:
            #     prm_str = ""
            #     for key in prm:
            #         prm_str += str(key) + "=" + str(prm[key]) + ","
            #     if prm_str:
            #         prm_str = prm_str.rstrip(",")
            #     cmdlist.extend(["-p", prm_str])

            if fp is not None:
                cmdlist.append(str(fp))
                # Check that file can be opened
                try:
                    ad = AstroData(fp)
                except:
                    self.wfile.write("Can't use AstroData to open %s"% fp)
                    return

                # Report reduction status
                self.wfile.write("Reducing %s\n" % fp)
                self.wfile.write("Command: %s\n" % " ".join(cmdlist))
                evman.append_event(ad, "status", 
                                   {"current":"reducing" , "logfile":None},
                                   msgtype="reduce_status")

            # Send reduce log to hidden directory
            logdir = ".autologs"
            if not os.path.exists(logdir):
                os.mkdir(logdir)

            reducelog = os.path.join(logdir, "reduce-addcinvokedlog-%d%s" % \
                                     (os.getpid(), str(time.time())))

            f = open(reducelog, "w")
            loglink = "reducelog-latest"
            if os.path.exists(loglink):
                os.remove(loglink)

            os.symlink(reducelog, loglink)

            # Call reduce
            pid = subprocess.call(cmdlist, stdout=f, stderr=f)

            f.close()
            # Report finished status
            if fp is not None:
                if pid == 0:
                    evman.append_event(ad, "status",
                                       {"current":"reduction finished",
                                        "logfile":reducelog},
                                       msgtype="reduce_status")
                else:
                    evman.append_event(ad, "status",
                                       {"current":"reduction ERROR",
                                        "logfile":reducelog},
                                       msgtype="reduce_status")

            # Get text from log
            f = open(reducelog, "r")      
            txt = f.read()
            f.close()
            self.wfile.write(txt)
            self.wfile.flush()
            return

    # -------------------------------------------------------------------------
    # privitized handling cmdqueue.json requests
    
    def _handle_cmdqueue_json(self, rim, parms):
        """Handle HTTP client GET requests on service: cmdqueue.json
        """
        msg_form  = '"%s" %s %s'
        info_code = 203
        fail_code = 416
        size = "-"
        verbosity = self.informers["verbose"]

        self.send_response(200)
        self.send_header('Content-type', "application/json")
        self.end_headers()

        # N.B. A timestamp of zero will request everything from fitsstore
        # This could be huge. Be careful passing GET request on cmdqueue.json
        # with no timestamp.

        if "timestamp" in parms:
            fromtime = float(parms["timestamp"][0])
        else:
            fromtime = 0

        # event_list = [] implies a new adcc. Request current op day 
        # metrics from fitsstore.

        if not rim.events_manager.event_list:
            self.log_message(msg_form, "No extant RIM events.",
                             info_code, size)
            self.log_message(msg_form, 
                             "Requesting current OP day events @FITS store",
                             info_code, size)

            rim.events_manager.event_list = fstore_get(current_op_timestamp())

            self.log_message(msg_form, "Received " + 
                             str(len(rim.events_manager.event_list)) + 
                             " events.", info_code, size)
            self.log_message(msg_form, "On QA metrics request: " +
                             stamp_to_opday(current_op_timestamp()),
                             info_code, size)

            tdic = rim.events_manager.get_list()

            tdic.insert(0, {"msgtype"  : "cmdqueue.request",
                            "timestamp": time.time()})
            tdic.append({"msgtype"  : "cmdqueue.request",
                         "timestamp": time.time()})

            self.wfile.write(json.dumps(tdic, sort_keys=True, indent=4))

        # Handle current nighttime requests ...
        elif stamp_to_opday(fromtime) == stamp_to_opday(current_op_timestamp()):
            if verbosity:
                self.log_message(msg_form, "Request metrics on current OP day: "+
                                 stamp_to_opday(fromtime), 
                                 info_code, size)

            tdic = rim.events_manager.get_list(fromtime=fromtime)
            tdic.insert(0, {"msgtype"  : "cmdqueue.request",
                            "timestamp": time.time()})

            self.wfile.write(json.dumps(tdic, sort_keys=True, indent=4))

        # Handle previous day requests
        elif fromtime < current_op_timestamp():
            if verbosity:
                self.log_message(msg_form, 
                                 "Requested metrics on ... " +
                                 stamp_to_opday(fromtime), info_code, size)
                                
            tdic = fstore_get(fromtime)

            if verbosity:
                self.log_message(msg_form, "Received " + str(len(tdic)) + 
                                 " events from fitsstore.", info_code, size)
            
            # Append the last timestamp from the event_list. This is done
            # to trigger the client to pinging the adcc from the last 
            # recorded event.

            tdic.insert(0, {"msgtype"  : "cmdqueue.request",
                            "timestamp": time.time()})
            tdic.append({"msgtype"  : "cmdqueue.request",
                         "timestamp": time.time()})

            self.wfile.write(json.dumps(tdic, sort_keys=True, indent=4))

        # Cannot handle the future ...
        else:
            self.log_message(msg_form, "Invalid timestamp received.", 
                             fail_code, size)
            self.log_message(msg_form, "Future events not known.",
                             fail_code, size)
        return
            


class MTHTTPServer(ThreadingMixIn, HTTPServer):
    """Handles requests using threads"""


def startInterfaceServer(port=8777, **informers):
    import socket
    try:
        # important to set prior to any instantiations of ADCCHandler
        ADCCHandler.informers = informers
        # e.g. below by the HTTPServer class
        findingPort = True
        while findingPort:
            try:
                print "Starting HTTP server on port %s ... " % str(port)
                server = MTHTTPServer(('', port), ADCCHandler)
                findingPort = False
            except socket.error:
                print "failed, port taken"
                port += 1
                
        print "Started  HTTP server on port %s" % str(port)
        while True:
            r, w, x = select.select([server.socket], [], [], .5)
            if r:
                server.handle_request()

            if webserverdone == True:
                print "shutting down HTTP interface"
                break
    except KeyboardInterrupt:
        print '^C received, shutting down server'
        server.socket.close()

main = startInterfaceServer
