#
#                                                                  gemini_python
#
#                                                     recipe_system.adcc.servers
#                                                                xmlrpc_proxy.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
import os
import time
import select
import socket
import xmlrpclib
import subprocess

from sys import stdout
from time import sleep

from SimpleXMLRPCServer import SimpleXMLRPCServer

from astrodata.utils import logutils
from astrodata.utils.Errors  import ADCCCommunicationError
# -----------------------------------------------------------------------------
log = logutils.get_logger(__name__)
reduceServer = None

# -----------------------------------------------------------------------------
def start_adcc(callerlockfile=None):
    import tempfile
    
    if callerlockfile is None:
        clf = tempfile.NamedTemporaryFile("w", prefix="clf4pid" + str(os.getpid()))
        clfn = clf.name
        clf.close()
    else:
        clfn = callerlockfile

    logdir = ".autologs"
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logname = os.path.join(logdir, "adcc-reducelog-%d-%s" % 
                           (os.getpid(), str(time.time())))
                          
    prsout = open(logname, "w")
    loglink = "adcclog-latest"

    if os.path.exists(loglink):
        os.remove(loglink)
    # print "creating %s -> %s" % (logname, loglink)
    os.symlink(logname, loglink)
    
    prsargs = ["adcc",
                "--invoked",
                "--reduce-pid", "%d" % os.getpid(),
                "--startup-report", clfn
               ]

    aproc = subprocess.Popen(prsargs, 
                           stdout=prsout, 
                           stderr=subprocess.STDOUT
                           )

    # wait for adccinfo, unless there is a lockfile, we'll trust it,
    # deal later with a dead host.
    if True:
        while not os.path.exists(clfn):
            # print "P69:waiting for", clfn, aproc.pid
            sleep(1)

    if os.path.exists(clfn):
        os.remove(clfn)

    return aproc

# -----------------------------------------------------------------------------
class ReduceCommands(object):
    prsready = False
    
    def __init__(self, reduce_server=None):
        self.reduce_server = reduce_server

    def get_version(self):
        return [("ReduceXMLRPC", "0.1")]

    def prs_ready(self):
        self.prsready = True
        reduceServer.prsready = True

# -----------------------------------------------------------------------------
class ReduceServer(object):
    finished = False
    prsready = False
    reducecmds = None
    listenport = 54530    
    xmlrpcthread = None

    def __init__(self):
        global reduceServer
        from threading import Thread
        self.xmlrpcthread = Thread(None, self.start_listening, "reducexmlrpc")
        self.xmlrpcthread.start() 
        reduceServer = self
        
    def start_listening(self):
        adcc_report  = 'adcclog-latest'
        findingport  = True

        while(findingport):
            try:
                server = SimpleXMLRPCServer(("localhost", 
                                             self.listenport), 
                                            allow_none=True, 
                                            logRequests=False)
                findingport = False
            except socket.error:
                self.listenport += 1

        self.reducecmds = ReduceCommands(self)
        server.register_instance(self.reducecmds)
        try:
            while True:
                r,w,x = select.select([server.socket], [],[],.5)
                if r:
                    server.handle_request()

                if self.finished == True:
                    break
        except KeyboardInterrupt:
            print '^C received, shutting down server'
            server.socket.close()
        return

# -----------------------------------------------------------------------------
class PRSProxy(object):
    # the xmlrpc interface is saved
    _class_prs = None

    def __init__(self, reduce_server=None, port=53530):
        self.prs = None
        self.log = None

        self.found = False
        self.prsport = port
        self.version  = None
        self.finished  = False
        self.registered = False
        
        self.httpport = None
        self.reducecmds = None
        self.xmlrpcthread = None
        self.reduce_server = reduce_server

        try:
            self.prs = xmlrpclib.ServerProxy("http://localhost:%d" % self.prsport, 
                                            allow_none=True,
                                            use_datetime=True)
        except socket.error:
            raise ADCCCommunicationError("Socket Error")

        self.found = True
        PRSProxy._class_prs = self
 
    def __del__(self):
        if hasattr(self.prs, "found"):
            self.prs.unregister(os.getpid())

    @classmethod
    def get_adcc(cls, reduce_server=None, check_once=False):
        # note: the correct ADCC will store it's info in .adcc/adccinfo.py
        racefile = ".adcc/adccinfo.py"
        try:
            with open(racefile) as rfile:
                info = eval(rfile.read())
        except IOError:
            if check_once is True:
                return None            
            comm_err = "\n\tSYSTEM ERROR:: "
            comm_err += "ADCC not found or the configuration has been corrupted."
            raise ADCCCommunicationError(comm_err)

        if (cls._class_prs) is not None:
            return cls._class_prs

        found    = False
        newProxy = None
        newProxy = PRSProxy(reduce_server=reduce_server, port=info["xmlrpc_port"])
        newProxy.httpport = info["http_port"]
        newProxy.localCalUrl = "http://localhost:%s/calsearch.xml" % newProxy.httpport

        if not found:
            log.info("reduce-->adcc?")

        while(not found):
            try:
                newProxy.version = newProxy.get_version()
                newProxy.found = True
                found = True
                if reduce_server:
                    details =  {"port":reduce_server.listenport}
                else:
                    details = {}

                newProxy.register(details)
            except socket.error:
                newProxy.found = False
                stdout.write(".")
                sleep(.1)
                if check_once:
                    newProxy = None
                    break

        log.info("reduce--><--adcc") 
        return newProxy

    def register(self, details=None):
        log.info("XMLRPC_proxy: Registering with adcc.")
        self.prs.register(os.getpid(), details)
        self.registered = True
        return

    def unregister(self):
        if self.registered:
            log.info("XMLRPC_proxy: Unregistering with adcc.")
            self.prs.unregister(os.getpid())
            self.registered=False
        else:
            log.warning("XMLRPC_proxy: Cannot unregister; Not registered.")
        return

    def calibration_search(self, cal_rq):
        if self.found == False:
            return None
        else:
            calrqdict = cal_rq.as_dict()
            try:
                cal = self.prs.calibration_search(calrqdict)
            except:
                print "XMLRPC_proxy: Calibration search fault"
                import traceback
                traceback.print_exc()
                log.error("XMLRPC_proxy: ADCC EXCEPTION: No calibration.")
                return None
            return cal

    def get_version(self):
        self.version = self.prs.get_version()
        return self.version

    def display_request(self, rq):
        self.prs.display_request(rq)
        return 

    def report_qametrics(self, event_list):
        self.prs.report_qametrics_2adcc(event_list)
        return

