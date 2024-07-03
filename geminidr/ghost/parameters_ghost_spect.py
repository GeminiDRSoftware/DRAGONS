# This parameter file contains the parameters related to the primitives located
# in the primitives_ghost_spect.py file, in alphabetical order.

from gempy.library import config

from geminidr.core import (
    parameters_ccd, parameters_visualize, parameters_preprocess,
    parameters_spect, parameters_stack)

from astrodata import AstroData as ad


class attachWavelengthSolutionConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_wavelengthSolutionAttached",
                          optional=True)
    arc_before = config.ListField("Before arc to use for wavelength solution",
                            (str, ad), None, optional=True, single=True)
    arc_after = config.ListField("After arc to use for wavelength solution",
                            (str, ad), None, optional=True, single=True)


class barycentricCorrectConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_barycentricCorrected",
                          optional=True)
    velocity = config.Field("Radial velocity correction (km/s)", float,
                            None, optional=True)


class calculateSensitivityConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_sensitivityCalculated", optional=True)
    filename = config.Field("Name of spectrophotometric data file", str, None, optional=True)
    order = config.RangeField("Order of polynomial fit to each echelle order", int,
                              2, min=1, max=5)
    in_vacuo = config.Field("Are spectrophotometric data wavelengths measured "
                            "in vacuo?", bool, None, optional=True)
    debug_airmass0 = config.Field("Calculate sensitivity curve at zero airmass?",
                                  bool, False)
    debug_plots = config.Field("Show response-fitting plots for each order?",
                               bool, False)


class darkCorrectConfig(parameters_preprocess.darkCorrectConfig):
    def setDefaults(self):
        self.suffix = "_darkCorrected"
        self.dark = None
        self.do_cal = "skip"


class determineWavelengthSolutionConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_wavelengthSolutionDetermined",
                          optional=True)
    flat = config.ListField("Flat field", (str, ad), None,
                            optional=True, single=True)
    min_snr = config.RangeField("Minimum S/N for peak detection",
                                float, 20, min=10)
    sigma = config.RangeField("Number of standard deviations for rejecting lines",
                              float, 3, min=1)
    max_iters = config.RangeField("Maximum number of iterations", int, 1, min=1)
    radius = config.RangeField("Matching distance for lines", int, 12, min=2)
    plot1d = config.Field("Produce 1D plots of each order to inspect fit?",
                          bool, False)
    plotrms = config.Field("Produce rms scattergram to inspect fit?",
                           bool, False)
    debug_plot2d = config.Field("Produce 2D plot to inspect fit?", bool, False)


class extractSpectraConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_extracted",
                          optional=True)
    slit = config.ListField("Slit viewer exposure", (str, ad), None,
                            optional=True, single=True)
    slitflat = config.ListField("Slit viewer flat field", (str, ad), None,
                                optional=True, single=True)
    flat = config.ListField("Flat field", (str, ad), None,
                            optional=True, single=True)
    ifu1 = config.ChoiceField("Status of IFU1", str,
                              allowed={"object": "Pointed at an object",
                                       "sky": "Pointed at sky",
                                       "stowed": "Stowed"},
                              default=None, optional=True)
    ifu2 = config.ChoiceField("Status of IFU1", str,
                              allowed={"object": "Pointed at an object",
                                       "sky": "Pointed at sky",
                                       "stowed": "Stowed"},
                              default=None, optional=True)
    sky_subtract = config.Field("Sky-subtract object spectra?", bool, True)
    flat_correct = config.Field("Flatfield the data?", bool, True)
    snoise = config.RangeField("Fraction of signal to be added to noise estimate for CR flagging",
                               float, 0.1, min=0, max=1)
    sigma = config.RangeField("Number of standard deviations at which to flag pixels",
                              float, 6, min=3)
    weighting = config.ChoiceField("Pixel weighting scheme for extraction", str,
                                   allowed={"uniform": "uniform weighting",
                                            "optimal": "optimal extraction"},
                                   default="optimal")
    min_flux_frac = config.RangeField("Minimum fraction of object flux to not flag extracted pixel",
                                      float, 0., min=0, max=1, inclusiveMax=True)
    ftol = config.RangeField("Fractional tolerance for convergence",
                                  float, 0.001, min=1e-8, max=0.05)
    apply_centroids = config.Field("Apply slit center-of-light offsets?", bool, False)
    seeing = config.RangeField("FWHM of seeing disc if no processed_slit is "
                               "available", float, None, min=0.2, optional=True)
    debug_cr_map = config.Field("Add CR map to output?", bool, False)
    debug_order = config.RangeField("Order for CR debugging plot", int, None,
                                       min=33, max=97, optional=True)
    debug_pixel = config.RangeField("Pixel for CR debugging plot", int, None,
                                       min=0, max=6144, optional=True)
    debug_timing = config.Field("Output time per order?", bool, False)


class combineOrdersConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_ordersCombined",
                          optional=True)
    scale = config.ChoiceField("Output wavelength scale", str, {
        #'linear': 'Linear wavelength scale',
        'loglinear': 'Log-linear wavelength scale'
    }, default='loglinear')
    oversample = config.RangeField("Oversampling of output wavelength scale",
                                   float, 1.0, min=0.5, max=10, inclusiveMax=True)
    stacking_mode = config.ChoiceField("Method of stacking spectra", str,
                                       allowed={"none": "No stacking",
                                                "scaled": "Scale spectra before stacking",
                                                "unscaled": "Stack spectra without scaling"},
                                                default="scaled", optional=True)


fluxCalibrateConfig = parameters_spect.fluxCalibrateConfig


class makeIRAFCompatibleConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_irafCompatible",
                              optional=False)


class measureBlazeConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_blazeMeasured",
                          optional=True)
    slitflat = config.ListField("Slit viewer flat field", (str, ad), None,
                                optional=True, single=True)


class overscanCorrectConfig(parameters_ccd.overscanCorrectConfig):
    def setDefaults(self):
        self.suffix = '_overscanCorrect'
        self.niter = 2
        self.lsigma = 3.0
        self.hsigma = 3.0
        self.function = 'chebyshev'
        self.nbiascontam = 4
        self.order = 0


class removeScatteredLightConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_scatteredLightRemoved",
                          optional=True)
    skip = config.Field("Skip removal of scattered light?", bool, True)
    xsampling = config.RangeField("Sampling in x direction (unbinned)", int, 64, min=32)
    debug_spline_smoothness = config.RangeField(
        "Scaling factor for spline smoothness", float, default=5, min=1)
    debug_percentile = config.RangeField("Percentile for statistics of inter-order light",
                                         float, default=40, min=1, max=50, inclusiveMax=True)
    debug_avoidance = config.RangeField(
        "Number of (unbinned) pixels to avoid at edges of orders", int, 2, min=1)
    debug_save_model = config.Field("Attach scattered light model to output?",
                                    bool, False)


class scaleCountsToReference(config.Config):
    suffix = config.Field("Filename suffix", str, "_countsScaled", optional=True)
    tolerance = config.RangeField("Tolerance for scaling compared to exposure time",
                                  float, 0, min=0, max=1, inclusiveMax=True)


class stackArcsConfig(parameters_stack.core_stacking_config):
    time_delta = config.RangeField("Max. time separating bracketed arcs (seconds)",
                              float, 1200, min=0, optional=True)

    def setDefaults(self):
        self.operation = "lmedian"


class stackFramesConfig(parameters_stack.core_stacking_config):
    def setDefaults(self):
        self.reject_method = "none"


class traceFibersConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_fibersTraced",
                          optional=True)
    slitflat = config.Field("Slit viewer flat field",
                            (str, ad),
                            None, optional=True)
    make_pixel_model = config.Field('Add a pixel model to the flat field?',
                                    bool, False)
    smoothing = config.RangeField("Gaussian smoothing of slit profile (unbinned pixels)",
                                  float, 0, min=0, optional=False)


write1DSpectraConfig = parameters_spect.write1DSpectraConfig


class tileArraysConfig(parameters_visualize.tileArraysConfig):
    def setDefaults(self):
        self.suffix = "_arraysTiled"
