# This parameter file contains the parameters related to the primitives located
# in the primitives_gnirs.py file, in alphabetical order.
from ..gemini.parameters_gemini import ParametersGemini
from ..core.parameters_nearIR import ParametersNearIR

class ParametersGNIRS(ParametersGemini, ParametersNearIR):
    associateSky = {
        "suffix"                : "_skyAssociated",
        "distance"              : 1.,
        "time"                  : 600.,
        "max_skies"             : None,
        "use_all"               : False,
    }