# This parameter file contains the parameters related to the primitives located
# in the primitives_gsaoi.py file, in alphabetical order.
from ..gemini.parameters_gemini import ParametersGemini
from ..core.parameters_nearIR import ParametersNearIR

class ParametersGSAOI(ParametersGemini, ParametersNearIR):
    addReferenceCatalog = {
        "suffix"                : "_refcatAdded",
        "radius"                : 0.033,
        "source"                : "2mass",
    }
    associateSky = {
        "suffix"                : "_skyAssociated",
        "time"                  : 900.,
        "distance"              : 1.,
        "max_skies"             : None,
        "use_all"               : False,
    }
    tileArrays = {
        "suffix"                : "_tiled",
        "tile_all"              : True,
    }