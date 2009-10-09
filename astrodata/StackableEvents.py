from ReductionObjectEvent import ReductionObjectEvent


class UpdateStackableEvent( ReductionObjectEvent ):
    '''
    
    '''
    def __init__(self):
        super( UpdateStackableEvent, self ).__init__()
        self.stkID = None
        self.stkList = []
        
        
    def __str__(self):
        '''
        
        
        '''
        
        tempStr = super( UpdateStackableEvent, self ).__str__()
        tempStr = tempStr + "ID: " + str( self.stkID ) + "\n" + \
                    "STACKABLE LIST: " + str( self.stkList )
        
        return tempStr



class GetStackableEvent( ReductionObjectEvent ):
    '''
    
    '''
    def __init__(self):
        super( GetStackableEvent, self ).__init__()
        self.stkID = None
        
        
    def __str__(self):
        '''
        
        
        '''
        
        tempStr = super( GetStackableEvent, self ).__str__()
        tempStr = tempStr + "ID: " + str( self.stkID )
        
        return tempStr
    
    