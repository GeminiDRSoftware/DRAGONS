# This parameter file contains the parameters related to the primitives located
# in the primitives_gemini.py file, in alphabetical order.

from geminidr.core.parameters_bookkeeping import ParametersBookkeeping
from geminidr.core.parameters_preprocess import ParametersPreprocess
from geminidr.core.parameters_standardize import ParametersStandardize
from geminidr.core.parameters_visualize import ParametersVisualize

class ParametersGemini(ParametersBookkeeping, ParametersPreprocess,
                       ParametersStandardize, ParametersVisualize):
    standardizeObservatoryHeaders = {
        "suffix"            : "_observatoryHeadersStandardized",
    }