class StackKeeper(object):
    stackLists = None
    
    def __init__(self):
        self.stackLists = {}
    
    def append(self, ID, addtostack):
        if type(addtostack) != list:
            addtostack = [addtostack]
            
        if ID not in self.stackLists:
            self.stackLists.update({ID:[]})
            
        self.stackLists[ID].extend(addtostack)

    def get(self, ID):
        if ID not in self.stackLists:
            return None
        else:
            return self.stackLists[ID]
