#
#                                                                  gemini_python
#
#
#                                                      usercalibrationservice.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]  # Changed by swapper, 22 May 2014
__version_date__ = '$Date$'[7:-3]
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class UserCalibrationService(object):
    
    def __init__(self):
        self.user_cal_dict = {}
    
    def add_calibration(self, caltype=None, cal_file=None):
        self.user_cal_dict.update({caltype:cal_file})
        return
        
    def get_calibration(self, caltype=None):
        return self.user_cal_dict.get(caltype)

user_cal_service = UserCalibrationService()
