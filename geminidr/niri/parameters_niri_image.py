# This parameter file contains the parameters related to the primitives located
# in the primitives_niri_image.py file, in alphabetical order.
from .parameters_niri import ParametersNIRI
from ..core.parameters_image import ParametersImage
from ..core.parameters_photometry import ParametersPhotometry

class ParametersNIRIImage(ParametersNIRI, ParametersImage, ParametersPhotometry):
    associateSky = {
        "distance"              : 1.,
        "time"                  : 900.,
    }