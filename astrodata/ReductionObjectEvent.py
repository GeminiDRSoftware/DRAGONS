from datetime import datetime

class ReductionObjectEvent( object ):
    '''
    
    '''
    
    def __init__(self):
        self.ver = None
        self.datetime = datetime.now()

    def __str__(self):
        '''
        
        '''
        tempStr = "VERSION: " + str( self.ver ) + \
                    "\nDATETIME: " + str( self.datetime ) + "\n"
                    
        return tempStr