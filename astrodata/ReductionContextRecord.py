from datetime import datetime
from AstroData import AstroData

class ReductionContextRecord( object ):
    '''
    
    
    '''
    timestamp = None
    
    def __init__( self, timestamp ):
        if timestamp == None:
            timestamp = datetime.now()
        self.timestamp = timestamp
    
    
class CalibrationRecord( ReductionContextRecord ):
    '''
    
    
    '''
    sciFilename = None
    caltype = None
    filename = None
    
    def __init__(self, sciFilename, filename, caltype, timestamp = None):
        super( CalibrationRecord, self ).__init__( timestamp )
        self.sciFilename = sciFilename
        self.filename = filename
        self.caltype = caltype
        
    def __str__(self):
        rets = """
    sciFilename = %s
    caltype     = %s
    filename    = %s
    timestamp   = %s \n""" % (self.sciFilename, self.caltype, self.filename, self.timestamp)
        return rets
    
    
class StackableRecord( ReductionContextRecord ):
    '''
    
    
    '''
    stkid = None
    filelist = []
    
    def __init__( self, stkid, filelist, timestamp=None ):
        super( StackableRecord, self ).__init__( timestamp )
        self.stkid = stkid
        self.filelist = filelist
    
    def __str__(self):
        rets = """
    stkid     = %s
    filelist  = %s
    timestamp = %s \n""" % ( str(self.stkid), str(self.filelist), self.timestamp )
        return rets  
    
class OutputRecord( ReductionContextRecord ):
    '''
    
    '''
    displayID = None
    filename = None
    ad = None
    
    def __init__(self, filename, displayID= None, timestamp = None, ad=None):
        super( OutputRecord, self ).__init__( timestamp )
        self.filename = filename
        #displayID may be obsolete
        self.displayID = displayID
        self.ad = Astrodata( filename )
    
    def __str__(self):
        rets = """
    displayID     = %s
    filename  = %s
    timestamp = %s
    astrodata = %s \n""" % ( str(self.displayID), str(self.filename), self.timestamp, str(self.ad) )
        return rets  
