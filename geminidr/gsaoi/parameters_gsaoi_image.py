# This parameter file contains the parameters related to the primitives located
# in the primitives_gsaoi_image.py file, in alphabetical order.
from .parameters_gsaoi import ParametersGSAOI
from ..core.parameters_image import ParametersImage
from ..core.parameters_photometry import ParametersPhotometry

class ParametersGSAOIImage(ParametersGSAOI, ParametersImage, ParametersPhotometry):
    pass