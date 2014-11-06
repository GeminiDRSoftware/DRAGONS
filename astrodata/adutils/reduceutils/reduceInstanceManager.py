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
# Exported and trimmed ReduceInstanceManager from adcc.
import xmlrpclib
from socket import error as socketError

from astrodata.StackKeeper import StackKeeper
from astrodata.eventsmanagers import EventsManager

# ------------------------------------------------------------------------------
class ReduceInstanceManager(object):
    numinsts = 0
    finished = False
    reducecmds = None
    reducedict = None
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
            print "Found no current reduce instances."
        self.reducedict = {}
        # these members save command history so that tools have access, e.g.
        #   a display tool
        self.stackKeeper = StackKeeper(local=True)
        self.events_manager = EventsManager(persist="adcc_events.jsa")
        
    def register(self, pid, details):
        """This function is exposed to the xmlrpc interface, and is used
        by reduce instances to register their details so the prsproxy
        can manage it's own and their processes. 
        """
        self.numinsts += 1
        print "registering client %d, number currently registered: %s" \
            % (pid, self.numinsts )
        self.finished = False
        print "registering client details:", repr(details)
        self.reducedict.update({pid:details})
        return
        
    def unregister(self, pid):
        self.numinsts -= 1
        if pid in self.reducedict:
            del self.reducedict[pid] 
        print "ADCC: unregistering client %d, number remaining registered %d" \
            % (pid, self.numinsts)
        if self.numinsts < 0:
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
        return retval
        
    def stackIDsGet(self, cachefile = None):
        retval = self.stackKeeper.get_stack_ids(cachefile)
        return retval
        
    def report_qametrics_2adcc(self, qd):
        self.events_manager.append_event(qd)
        return
