from datetime import datetime
from astrodata.AstroData import AstroData

class ReductionContextRecord( object ):
    '''
    The parent record. Contains all members global to all records. (i.e. timestamp)
    '''
    timestamp = None
    
    def __init__( self, timestamp ):
        if timestamp == None:
            timestamp = datetime.now()
        self.timestamp = timestamp
    
    
class CalibrationRecord( ReductionContextRecord ):
    '''
    Record for storing all relevant members related to calibration data.
    This is used specifically by the ReductionContext in its calibrations
    member.
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
    Contains the local cache information for a particular set of stackable data.
    Used in the ReductionContext records stackeep.
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
    
##@@FIXME: Because of the nature of how Output -> Input, the name of this record may need to change
## at some point.
class OutputRecord( ReductionContextRecord ):
    '''
    Contains any metadata related to output/input within the ReductionContext.
    This is used specifically in the ReductionContext records inputs and outputs.
    '''
    displayID = None
    filename = None
    ad = None
    
    def __init__(self, filename, displayID=None, timestamp=None, ad=None):
        super( OutputRecord, self ).__init__( timestamp )
        if type( filename ) == AstroData:
            self.filename = filename.filename
            self.ad = filename
        elif type( filename ) == str:
            self.filename = filename
            if ad is None:
                self.ad = AstroData( filename )
            else:
                self.ad = ad
        else:
            raise "BAD ARGUMENT"
        ##@@TODO: displayID may be obsolete
        self.displayID = displayID
        
        
    def __str__(self):
        rets = """
    displayID     = %s
    filename  = %s
    timestamp = %s
    astrodata = %s \n""" % ( str(self.displayID), str(self.filename), self.timestamp, str(self.ad) )
        return rets  
