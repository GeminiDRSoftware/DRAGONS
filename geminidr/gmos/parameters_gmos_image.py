# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_image.py file, in alphabetical order.

from parameters_gmos import ParametersGMOS
from geminidr.core.parameters_image import ParametersImage
from geminidr.core.parameters_photometry import ParametersPhotometry

class ParametersGMOSImage(ParametersGMOS, ParametersImage, ParametersPhotometry):
    # Override attach_mdf=True
    standardizeStructure = {
        "suffix"                : "_structureStandardized",
        "attach_mdf"            : False,
        "mdf"                   : None,
    }
