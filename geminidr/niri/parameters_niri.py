# This parameter file contains the parameters related to the primitives located
# in the primitives_niri.py file, in alphabetical order.
from ..gemini.parameters_gemini import ParametersGemini
from ..core.parameters_nearIR import ParametersNearIR

class ParametersNIRI(ParametersGemini, ParametersNearIR):
    addDQ = {
        "suffix"            : "_dqAdded",
        "bpm"               : None,
        "illum_mask"        : False,
        "latency"           : True,
    }
    addReferenceCatalog = {
        "suffix"                : "_refcatAdded",
        "radius"                : 0.033,
        "source"                : "2mass",
    }
    associateSky = {
        "suffix"                : "_skyAssociated",
        "distance"              : 1.,
        "time"                  : 900.,
        "max_skies"             : None,
        "use_all"               : False,
    }
    detectSources = {
        "suffix"                : "_sourcesDetected",
        "mask"                  : False,
        "replace_flags"         : 249,
        "set_saturation"        : True,
    }
