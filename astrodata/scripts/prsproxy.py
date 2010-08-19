#!/usr/bin/env python

import xmlrpclib
from urllib import urlopen
from SimpleXMLRPCServer import SimpleXMLRPCServer
import os, signal
import re
import exceptions
import urllib
from xml.dom import minidom

CALMGR = "http://hbffits1.hi.gemini.edu/calmgr"
CALTYPEDICT = { "bias": "processed_bias"}

from optparse import OptionParser

parser = OptionParser()

parser.set_description("This is the proxy to PRS functionality, also invoked locally, e.g. for calibration requests.")
parser.add_option("-i", "--invoked", dest = "invoked", action = "store_true",
            default = False,
            help = """Used by processes that invoke prsproxy, so that PRS proxy knows
            when to exit. If not present, the prsproxy registers itself and will
            only exit by user control (or by os-level signal).""")

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
    
    def register(self):
        self.numinsts +=1
        self.finished = False
        print "prs12 reg:",self.numinsts 
        
    def unregister(self):
        self.numinsts -= 1
        print "prs17 unreg:",self.numinsts 
        if self.numinsts == 0:
            self.finished = True
            #quitServer()s

def version():
    return [("PRSProxy","0.1")]
    
def calibrationSearch(rq):
    print "prs38: the request",repr(rq)
    
    if 'caltype' not in rq or 'datalabel' not in rq:
        return None
        
    rqurl = urljoin(CALMGR, CALTYPEDICT[rq['caltype']],rq['datalabel'] )
    print "prs52:", rqurl
    response = urllib.urlopen(rqurl).read()
    print "prs66:", response
    dom = minidom.parseString(response)
    calel = dom.getElementsByTagName("calibration")
    try:
        calurlel = dom.getElementsByTagName('url')[0].childNodes[0]
    except exceptions.IndexError:
        return None
    print "prs70:", calurlel.data
    
    return calurlel.data
    
server = SimpleXMLRPCServer(("localhost", 8777), allow_none=True)
print "PRS Proxy listening on port 8777..."
server.register_function(version, "version")
server.register_function(calibrationSearch, "calibrationSearch")

rim = ReduceInstanceManager()
server.register_instance(rim)

# server.serve_forever(

outerloopdone = False
while True:
    if outerloopdone:
        break;
    try:
        while True: #not finished:
            # print "prs53:", rim.finished
            server.handle_request() 
            # print "prs55:", rim.finished

            if options.invoked and rim.finished:
                print "prsproxy exiting, no reduce instances to serve."
                outerloopdone = True
                break
    except KeyboardInterrupt:
        if rim.numinsts>0:
            # note: save reduce pide (pass in register) and 
            #       and check if pids are running!
            print "prsproxy: Can't exit, %d instances of reduce running", rim.numinsts
        else:
            print "prsproxy: exiting due to Ctrl-C"
            # this directly breaks from the outer loop but outerloopdone for clarity
            outerloopdone = True
            break

