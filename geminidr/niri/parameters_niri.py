# This parameter file contains the parameters related to the primitives located
# in the primitives_niri.py file, in alphabetical order.
from ..gemini.parameters_gemini import ParametersGemini
from ..core.parameters_nearIR import ParametersNearIR

class ParametersNIRI(ParametersGemini, ParametersNearIR):
    addReferenceCatalog = {
        "suffix"                : "_refcatAdded",
        "radius"                : 0.033,
        "source"                : "2mass",
    }
    associateSky = {
        "suffix"                : "_skyAssociated",
        "distance"              : 1.,
        "time"                  : 900.,
    }
    detectSources = {
        "suffix"                : "_sourcesDetected",
        "mask"                  : False,
        "replace_flags"         : 249,
        "set_saturation"        : True,
    }
