from datetime import datetime

from astrodata.AstroData import AstroData
import os

#------------------------------------------------------------------------------ 
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
    timestamp   = %s""" % (  os.path.basename(self.sciFilename), 
                                self.caltype, 
                                os.path.basename(self.filename), 
                                self.timestamp)
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

class FringeRecord( ReductionContextRecord ):
    '''
    Contains the cache information for a set of fringeable data.
    '''
    fringeid = None
    listid = None
    filelist = []
    
    def __init__(self, fringeid, listid, filelist, timestamp=None):
        super( FringeRecord, self ).__init__( timestamp )
        self.fringeid = fringeid
        self.listid = listid
        self.filelist = filelist
        
        
    def __str__(self):
        rets = '''
fringeID   = %s
listID     = %s
filelist   = %s
timestamp = %s
''' % ( str(self.fringeid), str(self.listid), str(self.filelist), self.timestamp )

    def add(self, image):
        if type(image) == list:
            self.filelist.extend( image )
        else:
            self.filelist.append( image )

##@@FIXME: Because of the nature of how Output -> Input, the name of this record may need to change
## at some point.
class AstroDataRecord( ReductionContextRecord ):
    '''
    Contains any metadata related to output/input within the ReductionContext.
    This is used specifically in the ReductionContext records inputs and outputs.
    '''
    displayID = None
    filename = None
    ad = None
    parent = None
    
    def __init__(self, filename, displayID=None, timestamp=None, ad=None, parent=None, load = True):
        super( AstroDataRecord, self ).__init__( timestamp )
        print "RCR110:", type(filename), isinstance(filename, AstroData)
        if isinstance(filename, AstroData):
            self.filename = filename.filename
            self.ad = filename
            self.parent = filename.filename
        elif type( filename ) == str:
            self.filename = filename
            if load == True:
                self.ad = AstroData( filename )
            else:
                self.ad = None
            self.parent = parent
        elif type( filename ) == AstroDataRecord:
            self = filename
            return
        else:
            raise "BAD ARGUMENT"
        ##@@TODO: displayID may be obsolete
        self.displayID = displayID
    def load(self):
        self.ad = AstroData(self.filename)
        
    def __str__(self):
        rets = """
    displayID = %s
    filename  = %s
    timestamp = %s
    parent    = %s
    astrodata = %s \n""" % ( str(self.displayID), str(self.filename), self.timestamp, self.parent, str(self.ad) )
        return rets  
