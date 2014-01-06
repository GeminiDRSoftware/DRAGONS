#
#                                                                     QAP Gemini
#
#                                                       reduceInstanceManager.py
#                                                                        07-2013
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
# Exported ReduceInstanceManager from adcc.
import xmlrpclib

from copy   import copy
from socket import error as socketError

try:
    from PIL import Image
except ImportError:
    print "Cannot import PIL"

from numpy  import uint32

from astrodata import AstroData
from astrodata.StackKeeper import StackKeeper
from astrodata.eventsmanagers import EventsManager
from astrodata.adutils.reduceutils.CmdQueue import TSCmdQueue
from astrodata.adutils.reduceutils.CacheManager import get_cache_dir, put_cache_file
# ------------------------------------------------------------------------------
def numpy2im(ad):
    if isinstance(ad, AstroData):
        data = ad.hdulist[1].data
    else:
        data = ad        
    newdata = uint32(data)
    im = Image.fromarray(newdata, mode="I")
    return im


class ReduceInstanceManager(object):
    numinsts = 0
    finished = False
    reducecmds = None
    reducedict = None
    displayCmdHistory = None
    cmdNum = 0
    #stackKeeper = None
    events_manager = None
    
    def __init__(self, reduceport):
        # get my client for the reduce commands
        print "starting xmlrpc client to port %d..." % reduceport,
        self.reducecmds = xmlrpclib.ServerProxy("http://localhost:%d/" \
                                                % reduceport, allow_none=True)
        print "started"
        try:
            self.reducecmds.prs_ready()
        except socketError:
            print "prs50: no reduce instances running"
        self.reducedict = {}
        # these members save command history so that tools have access, e.g.
        #   a display tool
        self.stackKeeper = StackKeeper(local=True)
        self.displayCmdHistory = TSCmdQueue()
        self.events_manager = EventsManager(persist="adcc_events.jsa")
        
    def register(self, pid, details):
        """This function is exposed to the xmlrpc interface, and is used
        by reduce instances to register their details so the prsproxy
        can manage it's own and their processes. 
        """
        self.numinsts +=1
        print "registering client %d, number currently registered: %s" \
            % (pid, self.numinsts )
        self.finished = False
        print "registering client details:",repr(details)
        self.reducedict.update({pid:details})
        return
        
    def unregister(self, pid):
        self.numinsts -= 1
        if pid in self.reducedict:
            del self.reducedict[pid] 
        print "ADCC: unregistering client %d, number remaining registered %d" \
            % (pid, self.numinsts)
        if self.numinsts< 0:
            self.numinsts = 0
        if self.numinsts == 0:
            self.finished = True
        return
            
    def stackPut(self, ID, filelist, cachefile = None):
        self.stackKeeper.add(ID, filelist, cachefile)
        self.stackKeeper.persist(cachefile)
        return

    def stackGet(self, ID, cachefile = None):
        retval = self.stackKeeper.get(ID, cachefile)
        #print "adcc147:", repr(retval)
        return retval
        
    def stackIDsGet(self, cachefile = None):
        # print "adcc153:"
        retval = self.stackKeeper.get_stack_ids(cachefile)
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

                    numsci = ad.count_exts("SCI")
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
                        data = uint32(deepcopy(sci.data))
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
                        print "adcc164, extver -= %d top,bottom = %d,%d " \
                            %(sci.extver(),top, bottom)
                        abstop = 65535
                        factor = abstop/(top-bottom)
                        data = data - bottom
                        data = data*(factor)
                        im = numpy2im(data)
                        im = im.transpose(Image.FLIP_TOP_BOTTOM)
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

        self.displayCmdHistory.addCmd(rqcopy)
        return
        
    def report_qametrics_2adcc(self, qd):
        self.events_manager.append_event(qd)
        return
