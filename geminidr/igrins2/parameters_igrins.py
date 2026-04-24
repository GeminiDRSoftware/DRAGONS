# This parameter file contains the parameters related to the primitives
# define in the primitives_igrins.py file

from gempy.library import config
from geminidr.core import parameters_nearIR, parameters_standardize

class prepareConfig(parameters_standardize.prepareConfig):
    def setDefaults(self):
        self.require_wcs = False

class addDQConfig(parameters_nearIR.addDQConfig):
    pass

class selectFrameConfig(config.Config):
    frmtype = config.Field("frametype to filter", str)

class streamPatternCorrectedConfig(config.Config):
    # rpc_mode = config.Field("RP Correction mode", str, "guard")
    rpc_mode = config.Field("method to correct the pattern", str)
    suffix = config.Field("Readout pattern corrected", str, "_rpc")

class estimateNoiseConfig(config.Config):
    pass

class selectStreamConfig(config.Config):
    stream_name = config.Field("stream name for the output", str)

class addNoiseTableConfig(config.Config):
    pass

class setSuffixConfig(config.Config):
    suffix = config.Field("output suffix", str)

class somePrimitiveConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_suffix")
    param1 = config.Field("Param1", str, "default")
    param2 = config.Field("do param2?", bool, False)

class someStuffConfig(config.Config):
    suffix = config.Field("Output suffix", str, "_somestuff")

class determineSlitEdgesConfig(config.Config):
    pass

class maskBeyondSlitConfig(config.Config):
    pass

class makeBPMConfig(parameters_nearIR.makeBPMConfig):
    def setDefaults(self):
        # We need to revisit these parameters.
        # self.dark_lo_thresh = -150.
        # self.dark_hi_thresh = 650.
        self.flat_lo_thresh = 0.1
        # self.flat_hi_thresh = 1.28

class fixIgrinsHeaderConfig(config.Config):
    pass

class referencePixelsCorrectConfig(config.Config):
    apply_reference_pixels_correction = config.Field("Whether to apply reference pixel corrections", bool, False)

class extractSimpleSpecConfig(config.Config):
    pass

class identifyOrdersConfig(config.Config):
    pass

class identifyLinesConfig(config.Config):
    pass

class getInitialWvlsolConfig(config.Config):
    suffix = config.Field("Initial Wavelength solution attached", str, "_wvl0")

class extractSpectraMultiConfig(config.Config):
    pass

class identifyMultilineConfig(config.Config):
    pass

class volumeFitConfig(config.Config):
    pass

class makeSpectralMapsConfig(config.Config):
    pass

class attachWatTableConfig(config.Config):
    pass

class streamFirstFrameConfig(config.Config):
    pass

class setReferenceFrameConfig(config.Config):
    pass

class makeABConfig(config.Config):
    remove_level = config.Field("readoutpattern remove level", int, 2)
    remove_amp_wise_var = config.Field("remove amp-wise variation if True", bool, False)

class estimateSlitProfileConfig(config.Config):
    do_ab = config.Field("if ABBA is used", bool, True)
    frac_slit = config.ListField("slit fraction to extract", tuple, [(0, 1)])
    slit_profile_method = config.Field("slit profile method: column or full", str, "column")
    slit_profile_range = config.Field("x-range of detectors where slit profiles are estimated",
                                      tuple, (800, 2048-800))
class extractStellarSpecConfig(config.Config):
    extraction_mode = config.Field("Extraction mode", str, "optimal")
    pixel_per_res_element = config.Field("number of pixel per res. element", float, 0.)

class checkCALDBConfig(config.Config):
    caltypes = config.ListField("list of caltypes to check", str, [], single=False)

class fixHeaderConfig(config.Config):
    tags = config.ListField("tags to force", str, [], single=False)
    suffix = config.Field("suffix for files with fixed header", str, "_fixed")

class saveTwodspecConfig(config.Config):
    height_2dspec = config.Field("height of strip. 0 will set to median.", int, 0)
    wavelength_increasing_order =  config.Field("rearrange the data so that wavelengths are"
                                                "in increasing order. default is False",
                                                bool, False)

class saveDebugImageConfig(config.Config):
    save_debug =  config.Field("save the debug image if True. Default is True",
                                                bool, True)
class makeIgrinsBPMConfig(config.Config):
    hotpix_sigma_clip1 = config.Field("Sigma criterion to mask in + shape", float, 100)
    hotpix_sigma_clip2 = config.Field("Sigma criterion to mask in . shape", float, 10)
    deadpix_thresh = config.Field("Threshold to flat deadpixels", float, 0.6)
    deadpix_smooth_size = config.Field("Kernel size of the median filter", int, 9)

class readoutPatternCorrectFlatOffConfig(config.Config):
    flat_off_1st_pattern_removal_mode = config.Field("initial correction method for flat off", str,
                                                     "global_median")
    flat_off_2nd_pattern_removal_mode = config.Field("2nd stage correction method for flat off", str,
                                                     "auto")

class readoutPatternCorrectFlatOnConfig(config.Config):
    pass

class readoutPatternCorrectSkyConfig(config.Config):
    pass
