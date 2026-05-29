from gempy.library import config
from geminidr.core import parameters_spect


class correctFlexureConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_flexureCorrected",
                          optional=True)


class createDataCubeConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_dataCubeCreated",
                          optional=True)
    height_2dspec = config.Field("height of strip. 0 will set to median.", int, 0)
    wavelength_increasing_order =  config.Field("rearrange the data so that wavelengths are"
                                                "in increasing order. default is False",
                                                bool, False)


class determineSlitEdgesNewConfig(parameters_spect.determineSlitEdgesConfig):
    def setDefaults(self):
        self.nsum = 10
        self.min_snr = 1.5
        self.spectral_order = 4
        self.debug_max_shift = 0.2


class distortionCorrectConfig(parameters_spect.distortionCorrectConfig):
    use_dragons = config.Field("Use DRAGONS code for interpolation?", bool, False)



class extractSpectrumUsingProfileConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_myspec",
                          optional=True)
    extraction_mode = config.Field("Extraction mode", str, "optimal")
    pixel_per_res_element = config.Field("number of pixel per res. element", float, 0.)
    cr_rejection_thresh = config.RangeField("Sigma threshold for cosmic ray rejection", float, 30.,
                                            min=0)


class flagDiscrepantPixelsConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_discrepantPixelsFlagged",
                          optional=True)
    discrepant_pixel_threshold = config.RangeField("Sigma threshold for flagging discrepant pixels", float, 30.,
                                                   min=1)


class makeABNewConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_AB",  optional=True)
    remove_level = config.Field("readoutpattern remove level", int, 2)
    remove_amp_wise_var = config.Field("remove amp-wise variation if True", bool, False)


class makeSyntheticImageConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_syntheticImageMade",
                          optional=True)


class normalizeFlatNewConfig(parameters_spect.normalizeFlatConfig):
    pass
