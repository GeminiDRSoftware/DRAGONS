# This parameter file contains the parameters related to the primitives located
# in the primitives_standardize.py file, in alphabetical order.
from gempy.library import config
from astrodata import AstroData

class addIllumMaskToDQConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_illumMaskAdded", optional=True)
    illum_mask = config.Field("Name of illumination mask", str, None, optional=True)

class addDQConfig(addIllumMaskToDQConfig):
    static_bpm = config.Field("Static bad pixel mask", (str, AstroData), "default", optional=True)
    user_bpm = config.Field("User bad pixel mask", (str, AstroData), None, optional=True)
    add_illum_mask = config.Field("Apply illumination mask?", bool, False)

    def setDefaults(self):
        self.suffix = "_dqAdded"

class addMDFConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_mdfAdded", optional=True)
    mdf = config.Field("Name of MDF", str, None, optional=True)

class addVARConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_varAdded", optional=True)
    read_noise = config.Field("Add read noise?", bool, False)
    poisson_noise = config.Field("Add Poisson noise?", bool, False)

class makeIRAFCompatibleConfig(config.Config):
    pass

class standardizeInstrumentHeadersConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_instrumentHeadersStandardized", optional=True)

class standardizeObservatoryHeadersConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_observatoryHeadersStandardized", optional=True)

class standardizeHeadersConfig(standardizeObservatoryHeadersConfig, standardizeInstrumentHeadersConfig):
    def setDefaults(self):
        self.suffix = "_headersStandardized"

class standardizeStructureConfig(addMDFConfig):
    attach_mdf = config.Field("Attach MDF to spectroscopic data?", bool, True)

    def setDefaults(self):
        self.suffix = "_structureStandardized"

class validateDataConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_dataValidated", optional=True)

class prepareConfig(standardizeHeadersConfig, standardizeStructureConfig, validateDataConfig):
    def setDefaults(self):
        self.suffix = "_prepared"
