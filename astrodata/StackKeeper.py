from ReductionContextRecords import StackableRecord, FringeRecord

class StackKeeper(object):
    '''
    A data structure for accessing stackable lists.
    '''
    stackLists = None
    
    def __init__(self):
        self.stackLists = {}
    
    def add(self, ID, addtostack):
        '''
        Add a list of stackables for a given ID. If the ID does not exist, make a new stackable list.
        
        @param ID: An id based off that derived from IDFactory.getStackableID. 
        @type ID: str
        
        @param addtostack: A list of files for stacking or a StackableRecord instance.
        @type addtostack: list or StackableRecord  
        '''
        if type(addtostack) != list:
            addtostack = [addtostack]
            
        if ID not in self.stackLists:
            self.stackLists.update( {ID:StackableRecord(ID,[])} )
        
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
    
