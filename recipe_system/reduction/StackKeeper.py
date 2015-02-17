#
#                                                                  gemini_python
#
#                                                        recipe_system.reduction
#                                                                 StackKeeper.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
import os
import pickle

from copy import copy
from threading import RLock

from ..adcc.servers import xmlrpc_proxy
from .reductionContextRecords import StackableRecord, FringeRecord

class SKExcept:
    """ This is the general exception the classes and functions in the
    StackKeeper.py module raise.
    """
    def __init__(self, msg="Exception Raised in StackKeeper", **argv):
        """This constructor takes a message to print to the user."""
        self.message = msg
        for arg in argv.keys():
            exec("self."+arg+"="+repr(argv[arg]))
            
            
    def __str__(self):
        """
        This str conversion member returns the message given by the user 
        (or the default message) when the exception is not caught.

        """
        return self.message
        
class StackKeeper(object):
    """
    A data structure for accessing stackable lists.
    It keeps a dictionary indexed by the cachefile name, which
    contain a dict keyed by stack id, with a list of filenames as the value.
    """
    # stack dirs
    adcc = None
    lock = None
    shared = False
    stackLists = None
    cacheIndex = None
    def __init__(self, local = False):
        # print "SK34:", repr(local)
        shared = (not local)
        self.lock = RLock()
        self.stackLists = {}
        self.cacheIndex = {}
        
        if local != True:
            self.local = local
            self.adcc = xmlrpc_proxy.PRSProxy.get_adcc()
            # print "SK43:", repr(self.adcc)

        self.local = local
        self.shared = shared
    
    def add(self, id, addtostack, cachefile = None):
        """
        Add a list of stackables for a given id. If the id does not exist, 
        make a new stackable list.
        
        @param id: An id based off that derived from IDFactory.getStackableID. 
        @type id: <str>
        
        @param addtostack: A list of stackable file or StackableRecord instance.
        @type addtostack: <list> or <StackableRecord>
        """
        if hasattr(self, "_cc"):
            cc = self._cc
        else:
            self._cc = 1
            cc = 1

        cachefile = os.path.abspath(cachefile)
        self.lock.acquire()
        if cachefile == None:
            raise SKExcept("""
                            cachedir not specified, 
                            will use local cachedir in future but must 
                            be set at current time.""")
        if type(addtostack) != list:
            addtostack = [addtostack]
            
            
        if self.local == False:
            abscf = os.path.abspath(cachefile)
            self.adcc.prs.stackPut(id, addtostack, abscf)
            self.lock.release()
            return 
            
        # this is the local storage, in general use this is the instance
        # used by the adcc
        if cachefile not in self.cacheIndex:

            self.load(cachefile)
            
        stacksDict = self.cacheIndex[cachefile]
        if id not in stacksDict:
            stacksDict.update( {id:StackableRecord(id,[])} )

        flist = stacksDict[id].filelist
        for ftoadd in addtostack:
            if ftoadd not in flist:
                flist.append(ftoadd)
        
        self.persist(cachefile = cachefile)
        self.lock.release()

    def get_stack_ids(self, cachefile = None):
        cachefile = os.path.abspath(cachefile)
        if self.local == False:
            retval = self.adcc.prs.stackIDsGet(cachefile)
            return retval
        else:
            if cachefile not in self.cacheIndex:
                return []
            # MT locking
            self.lock.acquire()
            stacksDict = self.cacheIndex[cachefile]
            sids = stacksDict.keys()
            self.lock.release()     
            return sids
        
        return []           
        

    def get(self, id, cachefile = None):
        '''
        Get the stackable list for a given id.
        
        @param id: An id based off that derived from IDFactory.getStackableID.
        @type id: str
        
        @return: List of files for stacking.
        @rtype: list of str
        '''
        
        cachefile = os.path.abspath(cachefile)
        if self.local == False:
            #print "SK136: remote stack request"
            retval = self.adcc.prs.stackGet(id, cachefile)
            #print "SK138:", repr(retval)
            return retval
        else:
            #print "SK139: local stack load"
            pass
            
        print "SK143",repr(self.cacheIndex)
        
        if (cachefile not in self.cacheIndex 
            or len(self.cacheIndex[cachefile]) == 0):
            #@@NOTE: use memory version first, one adcc per machine makes this
            # reasonable
            self.load(cachefile = cachefile)
            if cachefile not in self.cacheIndex:
                return []

        self.lock.acquire()

        stacksDict = self.cacheIndex[cachefile]
                    
        if id not in stacksDict:
            self.lock.release()
            return []
        else:
            scopy = copy(stacksDict[id].filelist)
            self.lock.release() 
            return scopy
            
    def load(self, cachefile = None):
        """
        This member loads the persistent stack for the given cachefile
        name. NOTE: the contents of the cachefile will stomp any in-memory
        copy of the cachefile. Process and thread safety 
        (say if the list should
        be made a union) must take place in the calling function.
        """
        if cachefile == None:
            raise SKExcept("Cannot load stack list, cachefile == None")
        self.lock.acquire()
        try:
            if os.path.exists(cachefile):
                pfile = open(cachefile, "r")
                # print "SK131:", cachefile
                stacksDict = pickle.load(pfile)
                pfile.close()
            else:
                stacksDict = {}
            # @@NOTE: consider doing union between in memory stack and loaded one
            # @@NOTE: this would only be an issue if there are either
            # @@NOTE:  two or more adcc instances running (which has to be forced)
            # @@NOTE:  or adcc instances running on different machine while two
            # @@NOTE:  or more reduces run on different machine, in a shared directory
            # @@NOTE:  i.e. via network mount (i.e. NFS)
            self.cacheIndex.update({cachefile: stacksDict})
        except:
            self.lock.release()
            raise
        self.lock.release()
                
    def persist(self, cachefile = None):
        if (self.local == False):
            #do nothing, persistence is done on local side
            return
            
        if cachefile == None:
            raise SKExcept("Cannot persist, cachefile == None")
        self.lock.acquire()
        if cachefile not in self.cacheIndex:
            print "cachefile not in Index"
            raise
            return # nothing to persist
        pfile = open(cachefile, "w")
        stacksDict = self.cacheIndex[cachefile]
        pickle.dump(stacksDict, pfile)
        pfile.close()
        self.lock.release()
        
    def __str__(self):
        self.lock.acquire()
        tempstr = ""
        for cachefile in self.cacheIndex:
            for item in self.cacheIndex[cachefile].values():
                tempstr += str(item) + "\n"
        self.lock.release()
        return tempstr

class FringeKeeper:
    """
    A data structure for accessing stackable lists.
    """
    stack_lists = None
    
    def __init__(self):
        self.stack_lists = {}
    
    def add(self, list_id, astro_id, addtostack):
        """
        Add a list of stackables for a given id. If the id does not exist, 
        make a new stackable list.
        
        @param id: An id based off that derived from IDFactory.getStackableID. 
        @type id: <str>
        
        @param addtostack: A list of stackable files or a StackableRecord instance.
        @type addtostack: <list> or <StackableRecord>  
        """
        if type(addtostack) != list:
            addtostack = [addtostack]
            
        if list_id not in self.stack_lists:
            self.stack_lists.update( {list_id:FringeRecord(list_id, astro_id, [])} )
        
        ##@@FIXME: This code seems pointless if the code above sets it to a list.
        if type(addtostack) == list:
            # A quick way to perform diff on a list.
            addtostack = list(set(addtostack) - set(self.stack_lists[id].filelist))
        else:
            # Assumed it is StackableRecord 
            # This will also convert the addtostack to a list
            addtostack = list(set(addtostack.filelist) - set(self.stack_lists[id].filelist))
        self.stack_lists[id].filelist.extend(addtostack)

    def get(self, id):
        """
        Get the stackable list for a given id.
        
        @param id: An id based off that derived from IDFactory.getStackableID.
        @type id: str
        
        @return: List of files for stacking.
        @rtype: list of str
        """
        if id not in self.stack_lists:
            return None
        else:
            return self.stack_lists[id]

    def __str__(self):
        tempstr = ""
        for item in self.stack_lists.values():
            tempstr += str(item) + "\n"
        return tempstr
    
