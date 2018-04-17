# This parameter file contains the parameters related to the primitives located
# in the primitives_standardize.py file, in alphabetical order.
from gempy.library import config
from astrodata import AstroData

class addIllumMaskToDQConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_illumMaskAdded")
    illum_mask = config.Field("Name of illumination mask", str, None, optional=True)

class addDQConfig(addIllumMaskToDQConfig):
    static_bpm = config.Field("Static bad pixel mask", (str, AstroData), "default", optional=True)
    user_bpm = config.Field("User bad pixel mask", (str, AstroData), None, optional=True)
    add_illum_mask = config.Field("Apply illumination mask?", bool, False)

    def setDefaults(self):
        self.suffix = "_dqAdded"

class addMDFConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_mdfAdded")
    mdf = config.Field("Name of MDF", str, None, optional=True)

class addVARConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_varAdded")
    read_noise = config.Field("Add read noise?", bool, False)
    poisson_noise = config.Field("Add Poisson noise?", bool, False)

class standardizeInstrumentHeadersConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_instrumentHeadersStandardized")

class standardizeObservatoryHeadersConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_observatoryHeadersStandardized")

class standardizeHeadersConfig(standardizeObservatoryHeadersConfig, standardizeInstrumentHeadersConfig):
    def setDefaults(self):
        self.suffix = "_headersStandardized"

class standardizeStructureConfig(addMDFConfig):
    attach_mdf = config.Field("Attach MDF?", bool, True)

    def setDefaults(self):
        self.suffix = "_structureStandardized"

class validateDataConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_dataValidated")
    num_exts = config.ListField("Allowed number of extensions", int, 1, optional=True, single=True)

class prepareConfig(standardizeHeadersConfig, standardizeStructureConfig, validateDataConfig):
    def setDefaults(self):
        self.suffix = "_prepared"
