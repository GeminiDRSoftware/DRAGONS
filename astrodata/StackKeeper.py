from ReductionContextRecord import StackableRecord

class StackKeeper(object):
    stackLists = None
    
    def __init__(self):
        self.stackLists = {}
    
    def add(self, ID, addtostack):
        if type(addtostack) != list:
            addtostack = [addtostack]
            
        if ID not in self.stackLists:
            self.stackLists.update( {ID:StackableRecord(ID,[])} )
        
        
        
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
        print "STACKLIST AFTER ADD:", self.stackLists[ID]
        

    def get(self, ID):
        if ID not in self.stackLists:
            return None
        else:
            return self.stackLists[ID]

    def __str__(self):
        tempstr = ""
        for item in self.stackLists.values():
            tempstr += str(item) + "\n"
        return tempstr

    
