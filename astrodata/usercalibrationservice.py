class UserCalibrationService(object):
    user_cal_dict = None
    
    def __init__(self):
        self.user_cal_dict = {}
    
    def add_calibration(self, caltype = None, cal_file = None):
        self.user_cal_dict.update({caltype:cal_file})
        
    def get_calibration(self, caltype = None):
        # print "ucs11:", self.user_cal_dict
        if caltype in self.user_cal_dict:
            calfilename = self.user_cal_dict[caltype]
        else:
            calfilename = None
        return calfilename
    
user_cal_service = UserCalibrationService()

