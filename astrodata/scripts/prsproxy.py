#!/usr/bin/env python

import xmlrpclib
from urllib import urlopen
from SimpleXMLRPCServer import SimpleXMLRPCServer
import os, signal
import re
import exceptions
import urllib
from xml.dom import minidom
from threading import Thread
import prsproxyweb
import socket
import select

CALMGR = "http://hbffits1.hi.gemini.edu/calmgr"
CALTYPEDICT = { "bias": "processed_bias",
                "flat": "processed_flat"}


from optparse import OptionParser

parser = OptionParser()

parser.set_description("This is the proxy to PRS functionality, also invoked locally, e.g. for calibration requests.")
parser.add_option("-i", "--invoked", dest = "invoked", action = "store_true",
            default = False,
            help = """Used by processes that invoke prsproxy, so that PRS proxy knows
            when to exit. If not present, the prsproxy registers itself and will
            only exit by user control (or by os-level signal).""")
parser.add_option("-r", "--reduce-port", dest = "reduceport", default=53531, type="int",
            help="""When invoked by reduce, this is used to inform the prsproxy of the 
            port on which reduce listens for xmlrpc commands.""")
parser.add_option("-p", "--reduce-pid", dest ="reducepid", default=None, type="int",
            help = """When invoked by reduce, this option is used to inform the
            prsproxy of the reduce application's PID.""")
parser.add_option("-l", "--listen-port", dest = "listenport", default=53530, type="int",
            help="""This is the port that the prsproxy listens at for the xmlrps server.""")
parser.add_option("-w", "--http-port", dest = "httpport", default=8777, type="int",
            help="""This is the port the web interface will respond to. 
            http://localhost:<http-port>/""")
options, args = parser.parse_args()
# ----- UTILITY FUNCS

def urljoin(*args):
    for arg in args:
        if arg[-1] == '/':
            arg = arg[-1]
    ret = "/".join(args)
    print "prs31:", repr(args), ret
    return ret

# -------------------
class ReduceInstanceManager(object):
    numinsts = 0
    finished = False
    reducecmds = None
    reducedict = None
    def __init__(self):
        # get my client for the reduce commands
        print "starting xmlrpc client to port %d..." %options.reduceport,
        self.reducecmds = xmlrpclib.ServerProxy("http://localhost:%d/" % options.reduceport, allow_none=True)
        print "started"
        try:
            self.reducecmds.prsReady()
        except socket.error:
            print "prs50: no reduce instances running"
        self.reducedict = {}
        
    def register(self, pid, details):
        """This function is exposed to the xmlrpc interface, and is used
        by reduce instances to register their details so the prsproxy
        can manage it's own and their processes. 
        """
        self.numinsts +=1
        print "prs73 reg:",self.numinsts 
        self.finished = False
        print "prs75:",repr(details)
        self.reducedict.update({pid:details})
        # self.reducecmds.prsready()
        
    def unregister(self, pid):
        self.numinsts -= 1
        if pid in self.reducedict:
            del self.reducedict[pid] 
        print "prs17 unreg:",self.numinsts 
        if self.numinsts< 0:
            self.numinsts = 0
        if self.numinsts == 0:
            self.finished = True
            
            #quitServer()s

def get_version():
    version = [("PRSProxy","0.1")]
    print "prsproxy version:", repr(version)
    return version
    
def calibrationSearch(rq):
    print "prs38: the request",repr(rq)
    
    if 'caltype' not in rq or 'datalabel' not in rq:
        return None
        
    rqurl = urljoin(CALMGR, CALTYPEDICT[rq['caltype']],rq['datalabel'] )
    print "prs52:", rqurl
    response = urllib.urlopen(rqurl).read()
    #print "prs66:", response
    dom = minidom.parseString(response)
    calel = dom.getElementsByTagName("calibration")
    try:
        calurlel = dom.getElementsByTagName('url')[0].childNodes[0]
    except exceptions.IndexError:
        return None
    #print "prs70:", calurlel.data
    
    #@@TODO: test only 
    return calurlel.data
    
server = SimpleXMLRPCServer(("localhost", options.listenport), allow_none=True)
print "PRS Proxy listening on port %d..." % options.listenport
server.register_function(get_version, "get_version")
server.register_function(calibrationSearch, "calibrationSearch")

rim = ReduceInstanceManager()
server.register_instance(rim)


# server.serve_forever(
# start webinterface
webinterface = True #False
if (webinterface):
    #import multiprocessing
    web = Thread(None, prsproxyweb.main, "webface", 
                    kwargs = {"port":8777,
                              "rim":rim})
    web.start()
    
outerloopdone = False
while True:
    if outerloopdone:
        break;
    try:
        while True: #not finished:
            # print "prs53:", rim.finished
            r,w,x = select.select([server.socket], [],[],.5)
            if r:
                server.handle_request() 
            # print "P146:", repr(rim.reducedict)
            # print "prs55:", rim.finished
            #print "prs104:", prsproxyweb.webserverdone
            if prsproxyweb.webserverdone:
                print "prsproxy exiting due to command vie http interface"
                print "number of reduce instances abandoned:", rim.numinsts
                outerloopdone = True
                break
            if (options.invoked and rim.finished):
                print "prsproxy exiting, no reduce instances to serve."
                outerloopdone = True
                prsproxyweb.webserverdone = True
                break
    except KeyboardInterrupt:
        if rim.numinsts>0:
            # note: save reduce pide (pass in register) and 
            #       and check if pids are running!
            print "\nprsproxy: %d instances of reduce running" % rim.numinsts
            break
        else:
            print "\nprsproxy: exiting due to Ctrl-C"
            # this directly breaks from the outer loop but outerloopdone for clarity
            outerloopdone = True
            prsproxyweb.webserverdone = True
            # not needed os.kill(os.getpid(), signal.SIGTERM)
            break

