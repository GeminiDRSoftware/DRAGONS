from datetime import datetime


class ReductionContextRecord( object ):
    '''
    
    
    '''
    timestamp = None
    
    def __init__(self, timestamp):
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
    def __init__(self):
        pass
    
    def __str__(self):
        pass    
