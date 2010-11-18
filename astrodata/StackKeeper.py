import os
from copy import copy
from threading import RLock
import pickle

from ReductionContextRecords import StackableRecord, FringeRecord
import Proxies

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
        """This str conversion member returns the message given by the user (or the default message)
        when the exception is not caught."""
        return self.message
        
class StackKeeper(object):
    '''
    A data structure for accessing stackable lists.
    It keeps a dictionary indexed by the cachefile name, which
    contain a dict keyed by stack ID, with a list of filenames as the value.
    '''
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
            self.adcc = Proxies.PRSProxy.getADCC()
            # print "SK43:", repr(self.adcc)

        self.local = local
        self.shared = shared
    
    def add(self, ID, addtostack, cachefile = None):
        '''
        Add a list of stackables for a given ID. If the ID does not exist, make a new stackable list.
        
        @param ID: An id based off that derived from IDFactory.getStackableID. 
        @type ID: str
        
        @param addtostack: A list of files for stacking or a StackableRecord instance.
        @type addtostack: list or StackableRecord  
        '''
        #print "SK62: entering add"
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
            self.adcc.prs.stackPut(ID, addtostack, abscf)
            return 
            
        # this is the local storage, in general use this is the instance
        # used by the adcc
        # print "79: about to add to:", cachefile, repr(self.cacheIndex)
        if cachefile not in self.cacheIndex:
            stacksDict  = {}
            self.cacheIndex.update({cachefile:stacksDict})
        #print "SK83:"
        stacksDict = self.cacheIndex[cachefile]

        if ID not in stacksDict:
            stacksDict.update( {ID:StackableRecord(ID,[])} )

        # print "SK89:"   
        # @@REVIEW
        # could use set to handle duplicates but I am not sure
        # I want to lose the ordered nature of this list
        #stacksDict[ID].filelist.extend(addtostack)
        flist = stacksDict[ID].filelist
        for ftoadd in addtostack:
            if ftoadd not in flist:
                flist.append(ftoadd)
        
        self.lock.release()
        

    def get(self, ID, cachefile = None):
        '''
        Get the stackable list for a given ID.
        
        @param ID: An id based off that derived from IDFactory.getStackableID.
        @type ID: str
        
        @return: List of files for stacking.
        @rtype: list of str
        '''
        
        cachefile = os.path.abspath(cachefile)
        if self.local == False:
            retval = self.adcc.prs.stackGet(ID, cachefile)
            return retval
            
        if cachefile not in self.cacheIndex:
            return []

        self.lock.acquire()

        stacksDict = self.cacheIndex[cachefile]

        self.lock.release() 
                    
        if ID not in stacksDict:
            return []
        else:
            return copy(stacksDict[ID].filelist)
            
    def persist(self, cachefile = None):
        if cachefile == None:
            raise SKExcept("Cannot persist, cachefile == None")
        self.lock.acquire()
        if cachefile not in self.cacheIndex:
            return # nothing to persist
        pfile = open(cachefile, "w")
        # print "SK131:", cachefile
        stacksDict = self.cacheIndex[cachefile]
        # print "SK134:", repr(stacksDict)
        
        pickle.dump(stacksDict, pfile)
        pfile.close()
        
        # print "SK137: about to release SK lock in persist(..)"
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
    '''
    A data structure for accessing stackable lists.
    '''
    stackLists = None
    
    def __init__(self):
        self.stackLists = {}
    
    def add(self, listID, astroID, addtostack):
        '''
        Add a list of stackables for a given ID. If the ID does not exist, make a new stackable list.
        
        @param ID: An id based off that derived from IDFactory.getStackableID. 
        @type ID: str
        
        @param addtostack: A list of files for stacking or a StackableRecord instance.
        @type addtostack: list or StackableRecord  
        '''
        if type(addtostack) != list:
            addtostack = [addtostack]
            
        if listID not in self.stackLists:
            self.stackLists.update( {listID:FringeRecord(listID, astroID, [])} )
        
        ##@@FIXME: This code seems pointless if the code above sets it to a list. Check into it.
        if type(addtostack) == list:
            # A quick way to perform diff on a list.
            # This code may not be necessary, but it is nice for testing, so you
            # do not have the same file being added to stackables.
            addtostack = list( set(addtostack) - set(self.stackLists[ID].filelist) )
        else:
            # Assumed it is StackableRecord [Although this does not happen at the time I am
            # writing this, I have a feeling it will].
            # This will also convert the addtostack to a list
            addtostack = list( set(addtostack.filelist) - set(self.stackLists[ID].filelist) )
        self.stackLists[ID].filelist.extend(addtostack)
        # print "SK40: STACKLIST AFTER ADD:", self.stackLists[ID]
        

    def get(self, ID):
        '''
        Get the stackable list for a given ID.
        
        @param ID: An id based off that derived from IDFactory.getStackableID.
        @type ID: str
        
        @return: List of files for stacking.
        @rtype: list of str
        '''
        if ID not in self.stackLists:
            return None
        else:
            return self.stackLists[ID]

    def __str__(self):
        tempstr = ""
        for item in self.stackLists.values():
            tempstr += str(item) + "\n"
        return tempstr
    
