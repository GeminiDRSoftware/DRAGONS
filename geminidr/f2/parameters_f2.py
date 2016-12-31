# This parameter file contains the parameters related to the primitives located
# in the primitives_f2.py file, in alphabetical order.
from ..gemini.parameters_gemini import ParametersGemini
from ..core.parameters_nearIR import ParametersNearIR

class ParametersF2(ParametersGemini, ParametersNearIR):
    addReferenceCatalog = {
        "suffix"                : "_refcatAdded",
        "radius"                : 0.033,
        "source"                : "2mass",
    }

