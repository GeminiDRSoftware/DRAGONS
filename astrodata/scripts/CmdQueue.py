from threading import RLock

class CQExcept:
    """This class is an exception class for the Thread Safe Commands Queue module"""
    
    def __init__(self, msg="Exception Raised in AstroData system"):
        """This constructor accepts a string C{msg} argument
        which will be printed out by the default exception 
        handling system, or which is otherwise available to whatever code
        does catch the exception raised.
        
        :param: msg: a string description about why this exception was thrown
        
        :type: msg: string
        """
        self.message = msg
    def __str__(self):
        """This string operator allows the default exception handling to
        print the message associated with this exception.
        :returns: string representation of this exception, the self.message member
        :rtype: string"""
        return self.message
        
class TSCmdQueue:
    """Thread Safe Command Queue"""
    q = None
    lock = None
    def __init__(self):
        self.q = []
        self.lock = RLock()
        
    def addCmd(self, cmd, **kwargs):
        self.lock.acquire()
        print "CQ34L:", repr(cmd), repr(kwargs)
        if type(cmd) == dict:
            cmddict = cmd
        elif type(cmd) == str:
            cmddict = {cmd:kwargs}
        else:
            lock.release()
            raise CQExcept("cmd argument must be given as string or dict.")
        self.q.append(cmddict)
        self.lock.release()  
    
    def clearOld(self, cmdNum = None, date = None):
        """
        clearOld(...) deletes old commands by date or cmdNum
        """
        if cmdNum and date:
            CQExcept("cmdNum and date are mutually exclusive options, pick one.")
        if not cmdNum and not date:
            raise CQExcept("either cmdNum or date must be set.")
        
      
    def peekSince(self, cmdNum = None, date = None):
        """
        peekSince(...) grabs a copy of the queue from a given cmdNum or date.
        The command must possess one of these qualities
        """
        if cmdNum and date:
            CQExcept("cmdNum and date are mutually exclusive options, pick one.")
        if not cmdNum and not date:
            CQExcept("Either cmdNum or date must be set.")
            
        self.lock.acquire()
        retary = []
        print "CQ67:\n\n", repr(self.q)
        for cmddict in reversed(self.q):
            print "CQ69:", repr(cmddict)
            cmds = cmddict.keys()
            if len(cmds)>1:
                raise CQExcept("got multiple commands in single cmddict")
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
        # note also the lock has to be released in each of the possible return paths
        self.lock.release()

        return retary        
            
        
