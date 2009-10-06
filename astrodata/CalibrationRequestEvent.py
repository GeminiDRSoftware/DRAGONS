from ReductionObjectEvent import ReductionObjectEvent
from datetime import datetime

class CalibrationRequestEvent( ReductionObjectEvent ):
    '''
    The structure that stores the calibration parameters from the xml 
    calibration file.
    It is used by the control loop to be added to the request queue.
    '''
    def __init__(self):
        self.filename = None
        self.identifiers = {}
        self.criteria = {}
        self.priorities = {}
        self.caltype = None
        self.datetime = datetime.now()
    
    def __str__(self):
        tempStr = """filename: %(name)s
Identifiers: %(id)s
Criteria: %(crit)s
Priorities: %(pri)s
"""% {'name':str(self.filename),'id':str(self.identifiers), \
              'crit':str(self.criteria),'pri':str(self.priorities)}
        return tempStr
