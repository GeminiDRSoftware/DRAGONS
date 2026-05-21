# This parameter file contains the parameters related to the primitives
# define in the primitives_igrins.py file

from gempy.library import config
from geminidr.core import parameters_nearIR, parameters_spect, parameters_standardize

class prepareConfig(parameters_standardize.prepareConfig):
    def setDefaults(self):
        self.require_wcs = False


class addDQConfig(parameters_nearIR.addDQConfig):
    pass


class addNoiseTableConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_noiseTableAdded",
                          optional=True)


class attachWatTableConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_watTableAttached",
                          optional=True)


class checkCALDBConfig(config.Config):
    caltypes = config.ListField("list of caltypes to check", str, [], single=False)


class determineSlitEdgesConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_slitEdgesDetermined",
                          optional=True)


class distortionCorrectConfig(parameters_spect.distortionCorrectConfig):
    use_dragons = config.Field("Use DRAGONS code for interpolation?", bool, False)


class estimateNoiseConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_noiseEstimated",
                          optional=True)


class estimateSlitProfileConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_slitProfileEstimated",
                          optional=True)
    do_ab = config.Field("if ABBA is used", bool, True)
    frac_slit = config.ListField("slit fraction to extract", tuple, [(0, 1)])
    slit_profile_method = config.ChoiceField("slit profile method: column or full", str,
                                             allowed = {"full": "old method",
                                                        "column": "new method"},
                                             default="column", optional=False)
    slit_profile_range = config.Field("x-range of detectors where slit profiles are estimated",
                                      tuple, (800, 2048-800))


class extractSimpleSpecConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_extracted",
                          optional=True)


class extractSpectraMultiConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_extracted",
                          optional=True)
    nslices = config.RangeField("Number of slices (must be odd) to extract "
                                "along the length of the slit", int, 5,
                                min=1, max=9, inclusiveMax=True, optional=False)

    def validate(self):
        super().validate()
        if self.nslices % 2 == 0:
            raise config.ValidationError("nslices must be an odd number to have a central slice.")


class extractSpectraConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_spec1d",
                          optional=True)
    extraction_mode = config.Field("Extraction mode", str, "optimal")
    pixel_per_res_element = config.Field("number of pixel per res. element", float, 0.)


class fixHeaderConfig(config.Config):
    tags = config.ListField("tags to force", str, [], single=False)
    suffix = config.Field("suffix for files with fixed header", str, "_fixed")


class fixIgrinsHeaderConfig(config.Config):
    pass


class getInitialWvlsolConfig(config.Config):
    suffix = config.Field("Initial Wavelength solution attached", str, "_wvl0")


class identifyMultilineConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_multilineIdentified",
                          optional=True)


class identifyOrdersConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_ordersIdentified",
                          optional=True)


class identifyLinesConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_linesIdentified",
                          optional=True)


class makeABConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_AB",  optional=True)
    remove_level = config.Field("readoutpattern remove level", int, 2)
    remove_amp_wise_var = config.Field("remove amp-wise variation if True", bool, False)


class makeBPMConfig(parameters_nearIR.makeBPMConfig):
    def setDefaults(self):
        # We need to revisit these parameters.
        # self.dark_lo_thresh = -150.
        # self.dark_hi_thresh = 650.
        self.flat_lo_thresh = 0.1
        # self.flat_hi_thresh = 1.28


class makeIgrinsBPMConfig(config.Config):
    hotpix_sigma_clip1 = config.Field("Sigma criterion to mask in + shape", float, 100)
    hotpix_sigma_clip2 = config.Field("Sigma criterion to mask in . shape", float, 10)
    deadpix_thresh = config.Field("Threshold to flat deadpixels", float, 0.6)
    deadpix_smooth_size = config.Field("Kernel size of the median filter", int, 9)


class makeSpectralMapsConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_spectralMapsMade",
                          optional=True)


class maskBeyondSlitConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_maskedBeyondSlit",
                          optional=True)


class readoutPatternCorrectFlatOffConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_readoutPatterncorrected",
                          optional=True)
    flat_off_1st_pattern_removal_mode = config.Field("initial correction method for flat off", str,
                                                     "global_median")
    flat_off_2nd_pattern_removal_mode = config.Field("2nd stage correction method for flat off", str,
                                                     "auto")


class readoutPatternCorrectFlatOnConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_readoutPatterncorrected",
                          optional=True)


class readoutPatternCorrectSkyConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_readoutPatterncorrected",
                          optional=True)


class referencePixelsCorrectConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_referencePixelsCorrected",
                          optional=True)
    apply_correction = config.Field("Apply reference pixel correction?", bool, False)


class saveDebugImageConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_spec_debug", optional=True)
    save_debug =  config.Field("Save the debug image?",  bool, True)


class saveTwodspecConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_spec2d", optional=True)
    height_2dspec = config.Field("height of strip. 0 will set to median.", int, 0)
    wavelength_increasing_order =  config.Field("rearrange the data so that wavelengths are"
                                                "in increasing order. default is False",
                                                bool, False)


class selectFrameConfig(config.Config):
    frmtype = config.Field("frametype to filter", str)


class selectStreamConfig(config.Config):
    stream_name = config.Field("stream name for the output", str)


class setReferenceFrameConfig(config.Config):
    pass


class streamFirstFrameConfig(config.Config):
    pass


class streamPatternCorrectedConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_streamPatternCorrected",
                          optional=True)
    # rpc_mode = config.Field("RP Correction mode", str, "guard")
    rpc_mode = config.ChoiceField("method to correct the pattern", str,
                                  allowed={"full": "only option"}, default="full",
                                  optional=False)


class volumeFitConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_volumeFitted",
                          optional=True)
