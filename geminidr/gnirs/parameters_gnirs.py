# This parameter file contains the parameters related to the primitives located
# in the primitives_gnirs.py file, in alphabetical order.
from ..gemini.parameters_gemini import ParametersGemini
from ..core.parameters_nearIR import ParametersNearIR

class ParametersGNIRS(ParametersGemini, ParametersNearIR):
    addDQ = {
        "suffix"                : "_dqAdded",
        "bpm"                   : None,
        "illum_mask"            : True,
    }
    addReferenceCatalog = {
        "suffix"                : "_refcatAdded",
        "radius"                : 0.033,
        "source"                : "2mass",
    }
    associateSky = {
        "suffix"                : "_skyAssociated",
        "distance"              : 1.,
        "time"                  : 600.,
    }