#
#                                                                  gemini_python
#
#                                                  astrodata.adutils.reduceutils
#                                                                    CmdQueue.py
#                                                                   -- DPD Group
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-3]
__version_date__ = '$Date$'[7:-3]
# ------------------------------------------------------------------------------
import os
import cPickle

from threading import RLock
# ------------------------------------------------------------------------------

class CmdQueueError(Exception):
    """Exception for Command Queue module"""
    pass
        
class TSCmdQueue(object):
    """Thread Safe Command Queue"""
    def __init__(self):
        self.q = []
        self.lock = RLock()
        self.load();
        
    def dump(self):
        self.lock.acquire()
        cqfile = open("cqfile.pkl", "w")
        cPickle.dump(self.q, cqfile)
        cqfile.close()
        self.lock.release()
        return

    def load(self):
        self.lock.acquire()
        if os.path.exists("cqfile.pkl"):
            cqfile = open("cqfile.pkl", "r")
            self.q = cPickle.load(cqfile)
            cqfile.close()
        self.lock.release()
        return

    def addCmd(self, cmd, **kwargs):
        self.lock.acquire()
        if type(cmd) == dict:
            cmddict = cmd
        elif type(cmd) == str:
            cmddict = {cmd:kwargs}
        else:
            self.lock.release()
            raise CmdQueueError("cmd argument must be a string or dict.")
        self.q.append(cmddict)
        self.dump()
        self.lock.release()
        return

    def clearOld(self, cmdNum=None, date=None):
        """
        clearOld(...) deletes old commands by date or cmdNum
        """
        if cmdNum and date:
            raise CmdQueueError("Call with one of cmdNum OR data.")
        if not cmdNum and not date:
            raise CmdQueueError("either cmdNum or date must be set.")
        return

    def peekSince(self, cmdNum = None, date = None):
        """
        peekSince(...) grabs a copy of the queue from a given cmdNum or date.
        The command must possess one of these qualities
        """
        if cmdNum and date:
            raise CmdQueueError("Call with one of cmdNum OR data.")
        if not cmdNum and not date:
            raise CmdQueueError("Either cmdNum or date must be set.")
            
        self.lock.acquire()
        retary = []
        for cmddict in reversed(self.q):
            cmds = cmddict.keys()
            if len(cmds)>1:
                raise CmdQueueError("got multiple commands in single cmddict")
            else:
                cmd = cmds[0]
                cmdbody = cmddict[cmd]

            if cmdNum != None:
                print "CQ64:\npeek inclusive of cmdNum =", cmdNum, cmdbody["cmdNum"]
                if "cmdNum" in cmdbody:
                    if cmdbody["cmdNum"] >= cmdNum:
                        retary.append(cmddict)
                    else:
                        self.lock.release()
                        return retary

            if date:
                if "timestamp" in cmddict:
                    if cmddict["timestamp"] >= date:
                        retary.append(cmdict)
                    else:
                        self.lock.release()
                        return retary

        # note this means all the commands were sent
        # lock has to be released in each of the possible return paths
        self.lock.release()
        return retary
