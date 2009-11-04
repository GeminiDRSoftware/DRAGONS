from datetime import datetime
'''
    Current Requests:
        CalibrationRequest
        UpdateStackableRequest
        GetStackableRequest
        DisplayRequest
'''
class ReductionObjectRequest( object ):
    '''
    The parent of all Requests which contains members prevalent to all requests
    (i.e. timestamp).
    '''
    def __init__(self):
        self.ver = None
        self.timestamp = datetime.now()

    def __str__(self):
        '''
        
        '''
        tempStr = "\n\nVERSION: " + str( self.ver ) + \
                    "\nTIMESTAMP: " + str( self.timestamp ) + "\n"
                    
        return tempStr


class CalibrationRequest( ReductionObjectRequest ):
    '''
    The structure that stores the calibration parameters from the xml 
    calibration file.
    It is used by the control loop to be added to the request queue.
    '''
    filename = None
    identifiers = {}
    criteria = {}
    priorities = {}
    caltype = None
    
    def __init__( self,  filename=None, identifiers={}, criteria={}, priorities={}, caltype=None ):
        super( CalibrationRequest, self ).__init__()
        self.filename = None#filename
        #print "ASDSADSA:", identifiers
        self.identifiers = {}#identifiers
        self.criteria = {}#criteria
        self.priorities = {}#priorities
        self.caltype = None#caltype
        
    
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
    Contains all relevant information to request updating the stackable index.
    '''
    def __init__( self, stkID=None, stkList=[] ):
        super( UpdateStackableRequest, self ).__init__()
        self.stkID = stkID
        self.stkList = stkList
        
        
    def __str__(self):
        tempStr = super( UpdateStackableRequest, self ).__str__()
        tempStr = tempStr + "ID: " + str( self.stkID ) + "\n" + \
                    "STACKABLE LIST: " + str( self.stkList )
        
        return tempStr


class GetStackableRequest( ReductionObjectRequest ):
    '''
    The request to get the stackable list. (More of a PRS issue as updating the stack 
    already does this.)
    '''
    def __init__( self, stkID=None ):
        super( GetStackableRequest, self ).__init__()
        self.stkID = stkID
        
    def __str__(self):
        tempStr = super( GetStackableRequest, self ).__str__()
        tempStr = tempStr + "ID: " + str( self.stkID )
        
        return tempStr


class DisplayRequest( ReductionObjectRequest ):
    '''
    The request to display a list of fits files.
    '''
    def __init__( self, disID=None, disList=[] ):
        super( DisplayRequest, self ).__init__()
        self.disID = disID
        self.disList = disList
        
        
    def __str__(self):
        tempStr = super( DisplayRequest, self ).__str__()
        tempStr = tempStr + "ID: " + str( self.disID ) + "\n" + \
                    "DISPLAY LIST: " + str( self.disList )
        
        return tempStr