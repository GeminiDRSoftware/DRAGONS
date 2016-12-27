# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_spect.py file, in alphabetical order.

from .parameters_gmos import ParametersGMOS

class ParametersGMOSSpect(ParametersGMOS):
    findAcquisitionSlits = {
        "suffix"                : "_acqSlitsAdded",
    }
