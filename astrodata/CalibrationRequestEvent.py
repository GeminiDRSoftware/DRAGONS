from ReductionObjectEvent import ReductionObjectEvent

class CalibrationRequestEvent( ReductionObjectEvent ):
    '''
    The structure that stores the calibration parameters from the xml 
    calibration file.
    It is used by the control loop to be added to the request queue.
    '''
    def __init__(self):
        self.inputFilename = None
        self.identifiers = {}
        self.criteria = {}
        self.priorities = {}
    
    def __str__(self):
        tempStr = """inputFilename: %(name)s
Identifiers: %(id)s
Criteria: %(crit)s
Priorities: %(pri)s
"""% {'name':str(self.inputFilename),'id':str(self.identifiers), \
              'crit':str(self.criteria),'pri':str(self.priorities)}
        return tempStr