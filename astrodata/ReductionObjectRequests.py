from datetime import datetime

class ReductionObjectRequest( object ):
    '''
    Current Requests:
        CalibrationRequest
        UpdateStackableRequest
        GetStackableRequest
        DisplayRequest
    '''
    
    def __init__(self):
        self.ver = None
        self.timestamp = datetime.now()

    def __str__(self):
        '''
        
        '''
        tempStr = "VERSION: " + str( self.ver ) + \
                    "\nTIMESTAMP: " + str( self.timestamp ) + "\n"
                    
        return tempStr
    
    
    
class CalibrationRequest( ReductionObjectRequest ):
    '''
    The structure that stores the calibration parameters from the xml 
    calibration file.
    It is used by the control loop to be added to the request queue.
    '''
    def __init__( self,  filename=None, identifiers={}, criteria={}, priorities={}, caltype=None ):
        super( CalibrationRequest, self ).__init__()
        self.filename = filename
        self.identifiers = identifiers
        self.criteria = criteria
        self.priorities = priorities
        self.caltype = caltype
        
    
    def __str__(self):
        tempStr = super( CalibrationRequest, self ).__str__()
        tempStr = tempStr + """filename: %(name)s
Identifiers: %(id)s
Criteria: %(crit)s
Priorities: %(pri)s
"""% {'name':str(self.filename),'id':str(self.identifiers), \
              'crit':str(self.criteria),'pri':str(self.priorities)}
        return tempStr


class UpdateStackableRequest( ReductionObjectRequest ):
    '''
    
    '''
    def __init__( self, stkID=None, stkList=[] ):
        super( UpdateStackableRequest, self ).__init__()
        self.stkID = stkID
        self.stkList = stkList
        
        
    def __str__(self):
        '''
        
        
        '''
        
        tempStr = super( UpdateStackableRequest, self ).__str__()
        tempStr = tempStr + "ID: " + str( self.stkID ) + "\n" + \
                    "STACKABLE LIST: " + str( self.stkList )
        
        return tempStr


class GetStackableRequest( ReductionObjectRequest ):
    '''
    
    '''
    def __init__( self, stkID=None ):
        super( GetStackableRequest, self ).__init__()
        self.stkID = stkID
        
        
    def __str__(self):
        '''
        
        
        '''
        
        tempStr = super( GetStackableRequest, self ).__str__()
        tempStr = tempStr + "ID: " + str( self.stkID )
        
        return tempStr

class DisplayRequest( ReductionObjectRequest ):
    '''
    
    '''
    def __init__( self, disID=None, disList=[] ):
        super( DisplayRequest, self ).__init__()
        self.disID = disID
        self.disList = disList
        
        
    def __str__(self):
        '''
        
        
        '''
                
        tempStr = super( DisplayRequest, self ).__str__()
        tempStr = tempStr + "ID: " + str( self.disID ) + "\n" + \
                    "DISPLAY LIST: " + str( self.disList )
        
        return tempStr