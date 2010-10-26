#!/usr/bin/env python
import exceptions
from optparse import OptionParser
import os
import pickle
import re
import select
import signal
from SimpleXMLRPCServer import SimpleXMLRPCServer
import socket
from threading import Thread
import urllib
from urllib import urlopen
from xml.dom import minidom
import xmlrpclib
from copy import copy

from CmdQueue import TSCmdQueue
from astrodata.StackKeeper import StackKeeper

# @@DEPEND: PIL
try:
    import PIL
    from PIL import Image
except:
    print "Cannot import PIL"
# @@ main dependencies
import numpy
import numpy as np
from numpy import where
from datetime import datetime
import prsproxyweb
from prsproxyutil import calibrationSearch, CALMGR, CALTYPEDICT,urljoin

from astrodata import AstroData


parser = OptionParser()

parser.set_description("This is the proxy to PRS functionality, also invoked locally, e.g. for calibration requests.")
parser.add_option("-i", "--invoked", dest = "invoked", action = "store_true",
            default = False,
            help = """Used by processes that invoke prsproxy, so that PRS proxy knows
            when to exit. If not present, the prsproxy registers itself and will
            only exit by user control (or by os-level signal).""")
parser.add_option("--preload", dest = "preload", action = "store_true",
            default = False,
            help = """Useful in proxy mode, where some information otherwise produced
            during the first relevant request is prepared prior to starting the 
            HTTPServer.""")
parser.add_option("--reload", dest = "reload", action = "store_true",
            default = False,
            help = """Just like --preload, but uses last, cached (pickled)
            directory scan.""")
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
def getPersistDir(dirtitle = "adcc"):
    dirs = {"adcc":".adcc"}
    
    for dirt in dirs.keys():
        if not os.path.exists(dirs[dirt]):
            os.mkdir(dirs[dirt])
    
    return dirs[dirtitle]
    
    # move to other module or into a class!

def numpy2im(ad):
    if isinstance(ad, AstroData):
        data = ad.hdulist[1].data
    else:
        data = ad        
    #newdata = nump    .float(ad.data)
    newdata = numpy.uint32(data)
    im = Image.fromarray(newdata, mode="I")
    return im
    
stackKeeper = StackKeeper(local=True)
# -------------------
class ReduceInstanceManager(object):
    numinsts = 0
    finished = False
    reducecmds = None
    reducedict = None
    displayCmdHistory = None
    cmdNum = 0
    stackKeeper = None
    
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
        # these members save command history so that tools have access, e.g.
        #   a display tool
        self.stackKeeper = stackKeeper
        self.displayCmdHistory = TSCmdQueue()
        
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
    def stackPut(self, ID, filelist, cachefile = None):
        print "ADCC136:", ID, repr(filelist), repr(cachefile)
        self.stackKeeper.add(ID, filelist, cachefile)
        print "ADCC138: element added to stacklist:" , ID
        self.stackKeeper.persist(cachefile)
    
    def stackGet(self, ID, cachefile = None):
        retval = self.stackKeeper.get(ID, cachefile)
        return retval
        
    def displayRequest(self, rq):
        
        print "adcc99:", repr(rq)
        if "display" in rq:
            dispcmd = rq["display"]
            dispcmd.update({"timestamp":datetime.now(),
                            "cmdNum":self.cmdNum})
            self.cmdNum += 1
            rqcopy = copy(rq)

            print "adcc108:", repr(rqcopy)
            if "files" in dispcmd:
                files = dispcmd["files"]
                print "adcc110:", repr(files)
                for basename in files:
                    fileitem = files[basename]
                    ad = AstroData(fileitem["filename"])
                    print "adcc115: loaded ",ad.filename
                    from copy import deepcopy

                    numsci = ad.countExts("SCI")
                    if numsci > 2:
                        sci = ad[("SCI",2)]                    
                    else:
                        sci = ad[("SCI",1)]
                    data = sci.data
                    mean = data.mean()
                    bottom = data[where(data<mean)].mean()*.80
                    print "adcc140: bottom", bottom
                    top = data[where(data>(1.25*mean))].mean()
                    print "adcc142: top =",top
                    for sci in ad["SCI"]:
                        data = numpy.uint32(deepcopy(sci.data))
                        #data = numpy.uint32(sci.data)
                        if False:
                            mean = data.mean()
                            bottom = data[where(data<mean)].mean()
                            extver = sci.extver()
                            if extver == 1 or extver ==3:
                                bottom = bottom*1.33
                            print "adcc140: bottom", bottom
                            top = data[where(data>(1.25*mean))].mean()
                            print "adcc142: top =",top
                        bottom = int(bottom)
                        top = int(top)
                        #data [where(data<bottom)] = bottom
                        #data[where(data>top)] = top
                        print "adcc164, extver -= %d top,bottom = %d,%d " %(sci.extver(),top, bottom)
                        abstop = 65535
                        factor = abstop/(top-bottom)
                        data = data - bottom
                        data = data*(factor)
                        # data[where(data<0)] = 0
                        # data[where(data>abstop)] = abstop

                        im = numpy2im(data)
                        im = im.transpose(Image.FLIP_TOP_BOTTOM)
                        from CacheManager import get_cache_dir,put_cache_file
                        tdir = get_cache_dir("adcc.display")
                        dispname = "sci%d-%s_%d.png" % (sci.extver(), 
                                                     sci.data_label(),
                                                     dispcmd["cmdNum"])
                        nam = os.path.join(tdir, dispname)
                        put_cache_file(dispname, nam)

                        url = "/displaycache/"+dispname
                        baserq = rqcopy["display"]["files"][basename]
                        
                        if "extdict" not in baserq:
                            baserq.update({"extdict":{}})
                        baserq["extdict"].update(
                                    {"SCI%d"%sci.extver(): url }
                                )
                        rqcopy["display"]["files"][basename].update({"url": None})
                        
                        if os.path.exists(nam):
                            os.remove(nam)
                        im.save(nam, "PNG")
                        # print "adcc186:", repr(rqcopy)
        print "adcc197: about to addcmd"
        self.displayCmdHistory.addCmd(rqcopy)
        
def get_version():
    version = [("PRSProxy","0.1")]
    print "prsproxy version:", repr(version)
    return version
    

    
server = SimpleXMLRPCServer(("localhost", options.listenport), allow_none=True)
print "PRS Proxy listening on port %d..." % options.listenport
server.register_function(get_version, "get_version")
server.register_function(calibrationSearch, "calibrationSearch")

rim = ReduceInstanceManager()
server.register_instance(rim)

if options.preload == True:
    print "adcc: scanning current directory tree"
    from astrodata.DataSpider import DataSpider
    ds = DataSpider(".")
    dirdict = ds.datasetwalk()
    persistpath = os.path.join(getPersistDir(), "dataspider_cache.pkl")
    ddf = open(persistpath, "w")
    pickle.dump(dirdict, ddf)
    ddf.close()
    print "adcc: done scanning current directory tree"
else:
    ds = None
    dirdict = None
    
if options.reload == True:
    from astrodata.DataSpider import DataSpider
    print "prsproxy: reloading result of previous scan of directory tree"
    ds = DataSpider(".")
    persistpath = os.path.join(getPersistDir(), "dataspider_cache.pkl")
    ddf = open(persistpath)
    dirdict = pickle.load(ddf)
    ddf.close()


# server.serve_forever(
# start webinterface
webinterface = True #False
if (webinterface):
    #import multiprocessing
    if ds and dirdict:
        web = Thread(None, prsproxyweb.main, "webface", 
                    kwargs = {"port":8777,
                              "rim":rim,
                              "dirdict":dirdict,
                              "dataSpider":ds})
    else:
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
            #below allows exit anyway
            outerloopdone = True
            prsproxyweb.webserverdone = True
            break
        else:
            print "\nprsproxy: exiting due to Ctrl-C"
            # this directly breaks from the outer loop but outerloopdone for clarity
            outerloopdone = True
            prsproxyweb.webserverdone = True
            # not needed os.kill(os.getpid(), signal.SIGTERM)
            break

