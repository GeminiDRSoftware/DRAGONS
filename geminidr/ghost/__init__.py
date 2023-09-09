
# For development, let's patch in our calibration_ghost implementation to make life easier.
# Once it is finalized, we can migrate it back into the Gemini CalMgr code directly, or
# implement some more formal override option
print("OVERRIDING GHOST CALIBRATIONS")

from gemini_calmgr.cal import inst_class
from .calibration_ghost import CalibrationGHOST
inst_class["GHOST"] = CalibrationGHOST
