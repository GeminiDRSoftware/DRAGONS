from ReductionObjectEvent import ReductionObjectEvent

class CalibrationRequestEvent( ReductionObjectEvent ):
    '''
    The structure that stores the calibration parameters from the xml 
    calibration file.
    It is used by the control loop to be added to the request queue.
    '''
    def __init__(self):
        super( CalibrationRequestEvent, self ).__init__()
        self.filename = None
        self.identifiers = {}
        self.criteria = {}
        self.priorities = {}
        self.caltype = None
        
    
    def __str__(self):
        tempStr = super( CalibrationRequestEvent, self ).__str__()
        tempStr = tempStr + """filename: %(name)s
Identifiers: %(id)s
Criteria: %(crit)s
Priorities: %(pri)s
"""% {'name':str(self.filename),'id':str(self.identifiers), \
              'crit':str(self.criteria),'pri':str(self.priorities)}
        return tempStr
