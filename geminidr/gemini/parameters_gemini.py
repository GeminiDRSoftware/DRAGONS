# This parameter file contains the parameters related to the primitives located
# in the primitives_gemini.py file, in alphabetical order.
from gempy.library import config

class addMDFConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_mdfAdded", optional=True)
    mdf = config.Field("Name of MDF", str, None, optional=True)

class standardizeStructureConfig(addMDFConfig):
    attach_mdf = config.Field("Attach MDF to spectroscopic data?", bool, True)

    def setDefaults(self):
        self.suffix = "_structureStandardized"
