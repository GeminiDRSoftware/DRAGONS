# This parameter file contains the parameters related to the primitives located
# in the primitives_spect.py file, in alphabetical order.
import warnings

from astropy import table, units as u
from astropy.io import registry

from astrodata import AstroData
from geminidr.core import parameters_generic
from gempy.library import config, astrotools as at
from gempy.utils import logutils

from . import parameters_preprocess


log = logutils.get_logger(__name__)


def list_of_ints_check(value):
    [int(x) for x in str(value).split(',')]
    return True

def table_writing_formats():
    t = registry.get_formats(table.Table, readwrite="Write")
    return {fmt: "" for fmt, dep in t["Format", "Deprecated"] if dep != "Yes"}


def validate_regions_float(value):
    at.parse_user_regions(value, dtype=float, allow_step=False)
    return True

def validate_regions_int(value, multiple=False):
    ranges = at.parse_user_regions(value, dtype=int, allow_step=True)
    return multiple or len(ranges) == 1


class adjustWavelengthZeroPointConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_wavelengthZeroPointAdjusted",
                          optional=True)
    center = config.RangeField("Central row/column to extract", int, None,
                                min=1, optional=True)
    shift = config.RangeField("Shift to apply in pixels (None: determine automatically)",
                              float, 0, min=-2048, max=2048, optional=True)
    verbose = config.Field("Print extra information", bool, False,
                           optional=True)
    debug_max_shift = config.RangeField("Maximum shift to allow (in pixels)",
                                        float, 5, min=0)

    def validate(self):
        config.Config.validate(self)
        if self.shift is not None and abs(self.shift) > self.debug_max_shift:
            raise ValueError(f"Requested shift ({self.shift}) is larger than "
                             f"maximum allowed shift ({self.debug_max_shift}).\n"
                             "Set debug_max_shift to a larger value.")


class adjustWCSToReferenceConfig(config.Config):
    suffix = config.Field("Filename suffix",
                          str, "_wcsCorrected", optional=True)
    method = config.ChoiceField("Alignment method", str,
                                allowed={"sources_wcs": "Cross-correlate slit profile using WCS",
                                         "sources_offsets": "Cross-correlate slit profile using telescope offsets",
                                         "wcs": "Use WCS (no adjustment)",
                                         "offsets": "Use telescope offsets only"},
                                default="sources_wcs", optional=False)
    fallback = config.ChoiceField("Fallback method", str,
                                  allowed={"sources_offsets": "Cross-correlate slit profile using telescope offsets",
                                           "wcs": "Use WCS (no adjustment)",
                                           "offsets": "Use telescope offsets only"},
                                  default="wcs", optional=True)
    region = config.Field("Pixel section for measuring the spatial profile",
                           str, None, optional=True, check=validate_regions_int)
    tolerance = config.RangeField("Maximum distance from the header offset, "
                                  "for the correlation method (arcsec)",
                                  float, 1, min=0., optional=True)
    debug_block_resampling = config.Field("Block resampling in the spatial direction?", bool, False)
    debug_plots = config.Field("Plot the cross-correlation function?", bool, False)


class attachPinholeModelConfig(parameters_generic.calRequirementConfig):
    suffix = config.Field("Filename suffix", str, "_pinholeModelAttached", optional=True)
    pinhole = config.Field("Pinhole frame", (str, AstroData), None, optional=True)


class attachWavelengthSolutionConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_wavelengthSolutionAttached", optional=True)
    arc = config.ListField("Arc(s) with distortion map", (AstroData, str), None,
                           optional=True, single=True)


class calculateSensitivityConfig(config.core_1Dfitting_config):
    suffix = config.Field("Filename suffix", str, "_sensitivityCalculated", optional=True)
    filename = config.Field("Name of spectrophotometric data file", str, None, optional=True)
    in_vacuo = config.Field("Are spectrophotometric data wavelengths measured "
                            "in vacuo?", bool, None, optional=True)
    bandpass = config.RangeField("Bandpass width (nm) if not supplied",
                                 float, 0.001, min=0.001, max=10.)
    resampling = config.RangeField("Resampling interval (nm) for spectrophotometric data file",
                                   float, None, min=0, inclusiveMin=False, optional=True)
    debug_airmass0 = config.Field("Calculate sensitivity curve at zero airmass?",
                                  bool, False)
    regions = config.Field("Wavelength sample regions (nm)", str, None, optional=True,
                           check=validate_regions_float)
    debug_plot = config.Field("Plot sensitivity curve?", bool, False)
    interactive = config.Field("Display interactive fitter?", bool, False)

    def setDefaults(self):
        del self.grow


class createNewApertureConfig(config.Config):
    aperture = config.Field("Base aperture to offset from", int, None, optional=False)
    shift = config.Field("Shift (in pixels) to new aperture", float, None, optional=False)
    aper_upper = config.RangeField("Offset to new upper edge", float, None,
                                   optional=True, min=0., inclusiveMin=False)
    aper_lower = config.RangeField("Offset to new lower edge", float, None,
                                   optional=True, max=0., inclusiveMax=False)
    suffix = config.Field("Filename suffix", str, "_newApertureCreated", optional=True)

    def validate(self):
        config.Config.validate(self)
        if (self.aper_lower and self.aper_upper) or\
           (not self.aper_lower and not self.aper_upper):
            pass
        else:
            raise ValueError("Both aper_lower and aper_upper must either be "
                             "specified, or left as None.")


class cutSlitsConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_slitsCut", optional=True)


class determineDistortionConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_distortionDetermined", optional=True)
    spatial_order = config.RangeField("Fitting order in spatial direction", int, 3, min=1)
    spectral_order = config.RangeField("Fitting order in spectral direction", int, 4, min=0)
    id_only = config.Field("Use only lines identified for wavelength calibration?", bool, False)
    min_snr = config.RangeField("Minimum SNR for peak detection", float, 5., min=3.)
    fwidth = config.RangeField("Feature width in pixels if reidentifying",
                               float, None, min=1., optional=True)
    nsum = config.RangeField("Number of lines to sum", int, 10, min=1)
    step = config.RangeField("Step in rows/columns for tracing", int, 10, min=1)
    max_shift = config.RangeField("Maximum shift per pixel in line position",
                                  float, 0.05, min=0.001, max=0.1)
    max_missed = config.RangeField("Maximum number of steps to miss before a line is lost", int, 5, min=0)
    min_line_length = config.RangeField("Exclude line traces shorter than this fraction of spatial dimension",
                                        float, 0., min=0., max=1.)
    debug_reject_bad = config.Field("Reject lines with suspiciously high SNR (e.g. bad columns)?", bool, True)
    debug = config.Field("Display line traces on image display?", bool, False)


class determineSlitEdgesConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_slitEdgesDetermined", optional=True)
    spectral_order = config.RangeField("Fitting order in spectral direction",
                                       int, 3, min=1)
    edge1 = config.RangeField("Left/lower edge of illuminated region",
                              float, None, min=1)
    edge2 = config.RangeField("Right/upper edge of illuminated region",
                              float, None, min=1)
    search_radius = config.RangeField("Radius (in pixels) to search for edges",
                                      float, 30, min=5)
    debug_plots = config.Field("Plot fits of edges and print extra information",
                               bool, False)
    debug_max_missed = config.RangeField("Maximum missed steps when tracing edges",
                                         int, 8, min=1)
    debug_max_shift = config.RangeField("Maximum perpendicular shift (in pixels) per pixel",
                                        float, 0.08, min=0.)
    debug_step = config.RangeField("Step size (in pixels) for fitting edges",
                                   int, 20, min=5)
    debug_nsum = config.RangeField("Columns/rows to sum each step when fitting edges",
                                   int, 10, min=5)

    def validate(self):
        if hasattr(self, 'edge1'):
            if [self.edge1, self.edge2].count(None) == 1:
                raise ValueError("Both edges or neither edges must be specified")
            if self.edge1 is not self.edge2 <= self.edge1:
                raise ValueError("Right/upper edge must be greater than left/lower edge")


class determineWavelengthSolutionConfig(config.core_1Dfitting_config):
    suffix = config.Field("Filename suffix", str, "_wavelengthSolutionDetermined", optional=True)
    center = config.RangeField("Central row/column to extract", int, None, min=1, optional=True)
    nsum = config.RangeField("Number of lines to sum", int, 10, min=1)
    min_snr = config.RangeField("Minimum SNR for peak detection", float, 10., min=1.)
    min_sep = config.RangeField("Minimum feature separation (pixels)", float, 2., min=1.)
    weighting = config.ChoiceField("Weighting of identified peaks", str,
                                   allowed={"uniform": "uniform weighting",
                                            "global": "weighted by strength",
                                            "local": "weighted by strength relative to local peaks"},
                                   default="global")
    fwidth = config.RangeField("Feature width in pixels", float, None, min=2., optional=True)
    central_wavelength = config.RangeField("Estimated central wavelength (nm)", float, None,
                                           min=300., max=6000., optional=True)
    dispersion = config.RangeField("Estimated dispersion (nm/pixel)", float, None,
                                   min=-2, max=2, inclusiveMax=True, optional=True)
    linelist = config.Field("Filename of arc line list", str, None, optional=True)
    in_vacuo = config.Field("Use vacuum wavelength scale (rather than air)?", bool, False)
    absorption = config.Field("Is feature type absorption?", bool, False)
    debug_min_lines = config.Field("Minimum number of lines to fit each segment", (str, int), '15,20',
                                   check=list_of_ints_check)
    debug_alternative_centers = config.Field("Try alternative wavelength centers?", bool, False)
    interactive = config.Field("Display interactive fitter?", bool, False)
    num_atran_lines = config.RangeField("Number of lines in ATRAN line list", int, 50.,
                                              min=10, max=300, inclusiveMax=True)
    wv_band = config.ChoiceField("Water Vapor constraint", str,
                                   allowed={"20": "20%-ile",
                                            "50": "50%-ile",
                                            "80": "80%-ile",
                                            "100": "Any",
                                            "header": "header value"},
                                   default="header", optional=False)
    resolution = config.RangeField("Resolution of the observation", int, None, min=10, max=100000,
                                         optional=True)
    combine_method = config.ChoiceField("Combine method to use in 1D spectrum extraction", str,
                                   allowed={"mean": "mean",
                                            "median": "median"},
                                   default="mean", optional=False)
    verbose = config.Field("Print additional fitting information?", bool, False)

    def setDefaults(self):
        del self.function
        del self.grow
        self.niter = 3


class distortionCorrectConfig(parameters_generic.calRequirementConfig):
    suffix = config.Field("Filename suffix", str, "_distortionCorrected", optional=True)
    interpolant = config.ChoiceField("Type of interpolant", str,
                                     allowed={"nearest": "Nearest neighbour",
                                              "linear": "Linear interpolation",
                                              "poly3": "Cubic polynomial interpolation",
                                              "poly5": "Quintic polynomial interpolation",
                                              "spline3": "Cubic spline interpolation",
                                              "spline5": "Quintic spline interpolation"},
                                     default="poly3", optional=False)
    subsample = config.RangeField("Subsampling", int, 1, min=1)
    dq_threshold = config.RangeField("Fraction from DQ-flagged pixel to count as 'bad'",
                                     float, 0.001, min=0.)


class extractSpectraConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_extracted", optional=True)
    method = config.ChoiceField("Extraction method", str,
                                allowed={"aperture": "no weighting",
                                         "optimal": "optimal extraction",
                                         "default": "use 'optimal' for STANDARDs, and 'aperture' otherwise"},
                                default="aperture")
    width = config.RangeField("Width of extraction aperture (pixels)", float, None, min=1, optional=True)
    grow = config.RangeField("Source aperture avoidance region (pixels)", float, 10, min=0, optional=True)
    subtract_sky = config.Field("Subtract sky spectra if the data have not been sky corrected?", bool, True)
    debug = config.Field("Draw extraction apertures on image display? (not used with interactive)", bool, False)


def check_section(value):
    # Check for validity of a section string
    subsections = value.split(',')
    if len(subsections) == 1 and subsections[0] == '':
        # no Sections
        return True
    for i, (x1, x2) in enumerate(s.split(':') for s in subsections):
        try:
            x1 = int(x1)
        except ValueError:
            if i > 0 or x1 != '':
                return False
            else:
                x1 = 0
        try:
            x2 = int(x2)
        except ValueError:
            if i < len(subsections) - 1 or x2 != '':
                return False
        else:
            if x2 <= x1:
                raise ValueError("Section(s) do not have end pixel number "
                                 "greater than start pixel number")
    return True

class findAperturesConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_aperturesFound", optional=True)
    max_apertures = config.RangeField("Maximum number of sources to find",
                                      int, None, min=1, optional=True)
    percentile = config.RangeField("Percentile to determine signal for each spatial pixel",
                                   int, 80, min=1, max=100, optional=True, inclusiveMax=True)
    section = config.Field("Pixel section(s) for measuring the spatial profile",
                           str, "", optional=False, check=check_section)
    min_sky_region = config.RangeField("Minimum number of contiguous pixels "
                                       "between sky lines", int, 50, min=1)
    min_snr = config.RangeField("Signal-to-noise ratio threshold for peak detection",
                                float, 5.0, min=0.1)
    use_snr = config.Field("Use signal-to-noise ratio rather than data in "
                           "collapsed profile?", bool, True)
    threshold = config.RangeField("Threshold for automatic width determination",
                                  float, 0.1, min=0, max=1)
    interactive = config.Field("Use interactive interface", bool, False)
    max_separation = config.RangeField("Maximum separation from target location (arcsec)",
                                       int, None, min=1, inclusiveMax=True, optional=True)

class transferDistortionModelConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_distortionModelTransferred", optional=True)
    source = config.Field("Stream to transfer from", str, None)

class flagCosmicRaysConfig(config.Config):
    suffix = config.Field(
        doc="Filename suffix",
        dtype=str,
        default="_CRMasked",
        optional=True,
    )
    bitmask = config.Field(
        doc="Bits in the input data quality `flags` that are to be used to "
        "exclude bad pixels from cosmic ray detection and cleaning. Default "
        "65535 (all non-zero bits, up to 16 planes).",
        dtype=int,
        optional=True,
        default=65535,
    )
    debug = config.Field(
        doc="Make diagnostic plots?",
        dtype=bool,
        default=False
    )
    # Fit parameters --------------------------------------------------------
    spectral_order = config.Field(
        doc="Order for fitting and subtracting object continuum and sky line "
        "models, prior to running the main cosmic ray detection algorithm. "
        "To control which fits are performed, use the bkgmodel parameter.",
        dtype=int,
        optional=True,
        default=9,
    )
    spatial_order = config.Field(
        doc="Order for fitting and subtracting object continuum and sky line "
        "models, prior to running the main cosmic ray detection algorithm. "
        "To control which fits are performed, use the bkgmodel parameter.",
        dtype=int,
        optional=True,
        default=5,
    )
    bkgmodel = config.ChoiceField(
        doc="Set which background model(s) to use, between 'object', "
        "'skyline','both', or 'none'. Different data may get better results "
        "with different background models.",
        allowed={
            'both': 'Use both object and sky line models.',
            'object': 'Use object model only.',
            'skyline': 'Use sky line model only.',
            'none': "Don't use a background model.",
        },
        dtype=str,
        optional=True,
        default='skyline',
    )
    bkgfit_niter = config.Field(
        doc="Maximum number of iterations for the objects and sky fits.",
        dtype=int,
        optional=True,
        default=3,
    )
    bkgfit_lsigma = config.Field(
        doc="Rejection threshold in standard deviations below the mean, "
        "for the objects and sky fits.",
        dtype=float,
        optional=True,
        default=4.0,
    )
    bkgfit_hsigma = config.Field(
        doc="Rejection threshold in standard deviations above the mean, "
        "for the objects and sky fits.",
        dtype=float,
        optional=True,
        default=4.0,
    )
    # Astroscrappy's detect_cosmics parameters ------------------------------
    sigclip = config.Field(
        doc="Laplacian-to-noise limit for cosmic ray detection. Lower "
        "values will flag more pixels as cosmic rays.",
        dtype=float,
        optional=True,
        default=4.5,
    )
    sigfrac = config.Field(
        doc="Fractional detection limit for neighboring pixels. For cosmic "
        "ray neighbor pixels, a lapacian-to-noise detection limit of"
        "sigfrac * sigclip will be used.",
        dtype=float,
        optional=True,
        default=0.3,
    )
    objlim = config.Field(
        doc="Minimum contrast between Laplacian image and the fine structure "
        "image.  Increase this value if cores of bright stars are flagged as "
        "cosmic rays.",
        dtype=float,
        optional=True,
        default=5.0,
    )
    niter = config.Field(
        doc="Number of iterations of the LA Cosmic algorithm to perform",
        dtype=int,
        optional=True,
        default=4,
    )
    sepmed = config.Field(
        doc="Use the separable median filter instead of the full median "
        "filter. The separable median is not identical to the full median "
        "filter, but they are approximately the same and the separable median "
        "filter is significantly faster and still detects cosmic rays well.",
        dtype=bool,
        optional=True,
        default=True,
    )
    cleantype = config.ChoiceField(
        doc="Set which clean algorithm is used.",
        allowed={
            'median': 'An umasked 5x5 median filter',
            'medmask': 'A masked 5x5 median filter',
            'meanmask': 'A masked 5x5 mean filter',
            'idw': 'A masked 5x5 inverse distance weighted interpolation',
        },
        dtype=str,
        optional=True,
        default="meanmask",
    )
    fsmode = config.ChoiceField(
        doc="Method to build the fine structure image.",
        allowed={
            'median': 'Use the median filter in the standard LA Cosmic '
            'algorithm',
            'convolve': 'Convolve the image with the psf kernel to calculate '
            'the fine structure image.',
        },
        dtype=str,
        optional=True,
        default='median',
    )
    psfmodel = config.ChoiceField(
        doc="Model to use to generate the psf kernel if fsmode == 'convolve' "
        "and psfk is None. The current choices are Gaussian and Moffat "
        "profiles.",
        allowed={
            'gauss': 'Circular Gaussian kernel',
            'moffat': 'Circular Moffat kernel',
            'gaussx': 'Gaussian kernel in the x direction',
            'gaussy': 'Gaussian kernel in the y direction',
        },
        dtype=str,
        optional=True,
        default="gauss",
    )
    psffwhm = config.Field(
        doc="Full Width Half Maximum of the PSF to use for the kernel.",
        dtype=float,
        optional=True,
        default=2.5,
    )
    psfsize = config.Field(
        doc="Size of the kernel to calculate. Returned kernel will have size "
        "psfsize x psfsize. psfsize should be odd.",
        dtype=int,
        optional=True,
        default=7,
    )
    psfbeta = config.Field(
        doc="Moffat beta parameter. Only used if psfmodel=='moffat'.",
        dtype=float,
        optional=True,
        default=4.765,
    )
    verbose = config.Field(
        doc="Print to the screen or not.",
        dtype=bool,
        optional=True,
        default=False,
    )


def flux_units_check(value):
    # Confirm that the specified units can be converted to a flux density
    try:
        unit = u.Unit(value)
    except:
        raise ValueError(f"{value} is not a recognized unit")
    try:
        unit.to(u.W / u.m ** 3, equivalencies=u.spectral_density(1. * u.m))
    except u.UnitConversionError:
        raise ValueError(f"Cannot convert {value} to a flux density")
    return True


class fluxCalibrateConfig(parameters_generic.calRequirementConfig):
    suffix = config.Field("Filename suffix", str, "_fluxCalibrated", optional=True)
    standard = config.ListField("Standard(s) with sensitivity function", (AstroData, str),
                                None, optional=True, single=True)
    units = config.Field("Units for output spectrum", str, "W m-2 nm-1",
                         check=flux_units_check)


class linearizeSpectraConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_linearized", optional=True)
    w1 = config.RangeField("Starting wavelength (nm)", float, None, min=0., optional=True)
    w2 = config.RangeField("Ending wavelength (nm)", float, None, min=0., optional=True)
    dw = config.RangeField("Dispersion (nm/pixel)", float, None, min=0.01, optional=True)
    npix = config.RangeField("Number of pixels in spectrum", int, None, min=2, optional=True)
    conserve = config.Field("Conserve flux?", bool, None, optional=True)
    interpolant = config.ChoiceField("Type of interpolant", str,
                                     allowed={"nearest": "Nearest neighbour",
                                              "linear": "Linear interpolation",
                                              "poly3": "Cubic polynomial interpolation",
                                              "poly5": "Quintic polynomial interpolation",
                                              "spline3": "Cubic spline interpolation",
                                              "spline5": "Quintic spline interpolation"},
                                     default="poly3", optional=False)

    def validate(self):
        config.Config.validate(self)
        if [self.w1, self.w2, self.dw, self.npix].count(None) not in (1, 4):
            raise ValueError("Exactly 0 or 3 of w1, w2, dw, npix must be specified")
        if self.w1 is not None and self.w2 is not None and self.w2 <= self.w1:
            raise ValueError("Ending wavelength must be greater than starting wavelength")


class maskBeyondSlitConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_maskedBeyondSlit",
                          optional=True)

class normalizeFlatConfig(config.core_1Dfitting_config):
    suffix = config.Field("Filename suffix", str, "_normalized", optional=True)
    center = config.RangeField("Central (spatial axis) row/column for 1D extraction (None => use middle)", int, None, min=1, optional=True)
    offset_from_center = config.Field("Offset in pixels from center of slit",
                                      int, None, optional=True)
    nsum = config.RangeField('Number of rows/columns to average (about "center")', int, 10, min=1)
    threshold = config.RangeField("Threshold for flagging unilluminated pixels",
                                  float, 0.01, min=0.0001, max=1.0)
    interactive = config.Field("Interactive fitting?", bool, False)

    def setDefaults(self):
        self.order = 20


class resampleToCommonFrameConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_align", optional=True)
    w1 = config.RangeField("Starting wavelength (nm)", float, None, min=0., optional=True)
    w2 = config.RangeField("Ending wavelength (nm)", float, None, min=0., optional=True)
    dw = config.RangeField("Dispersion (nm/pixel) or fractional wavelength increase per pixel",
                           float, None, min=0.000001, optional=True)
    npix = config.RangeField("Number of pixels in spectrum", int, None, min=2, optional=True)
    conserve = config.Field("Conserve flux?", bool, None, optional=True)
    interpolant = config.ChoiceField("Type of interpolant", str,
                                     allowed={"nearest": "Nearest neighbour",
                                              "linear": "Linear interpolation",
                                              "poly3": "Cubic polynomial interpolation",
                                              "poly5": "Quintic polynomial interpolation",
                                              "spline3": "Cubic spline interpolation",
                                              "spline5": "Quintic spline interpolation"},
                                     default="poly3", optional=False)
    trim_spatial = config.Field("Trim spatial range to fully-covered region?", bool, True)
    trim_spectral = config.Field("Trim wavelength range to fully-covered region?", bool, False)
    force_linear = config.Field("Force linear wavelength solution?", bool, None,
                                optional=True)
    output_wave_scale = config.ChoiceField("Output wavelength scale", str,
                                           allowed={"reference": "Reference input",
                                                    "linear": "Linear",
                                                    "loglinear": "Log-linear",
                                                    },
                                           default="linear", optional=False)
    dq_threshold = config.RangeField("Fraction from DQ-flagged pixel to count as 'bad'",
                                     float, 0.001, min=0.)

    def validate(self):
        config.Config.validate(self)
        if [self.w1, self.w2, self.dw, self.npix].count(None) == 0:
            raise ValueError("Maximum 3 of w1, w2, dw, npix must be specified")
        if self.w1 is not None and self.w2 is not None and self.w2 <= self.w1:
            raise ValueError("Ending wavelength must be greater than starting wavelength")
        if self.npix is not None and self.dw is not None and self.w1 is None and self.w2 is None:
            raise ValueError("Cannot specify npix and dw without either w1 or w2")

        if self.force_linear is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("always", DeprecationWarning)
                warnings.warn("The parameter 'force_linear' has been deprecated. "
                              "Please use 'output_wave_scale' instead.",
                              category=DeprecationWarning)
            self.output_wave_scale = "linear" if self.force_linear else "reference"

        if self.output_wave_scale == "linear" and self.dw is not None and self.dw < 0.01:
            raise ValueError(f"Requested dispersion of {self.dw} < 0.01 nm/pixel")

        if (self.output_wave_scale == "reference" and
                [self.w1, self.w2, self.dw, self.npix].count(None) < 4):
            raise ValueError("Cannot specify grid parameters if "
                             "output_wave_scale='reference'")

        # We rely on an analytical inverse to the wavelength solution that is
        # only valid over the extent of its extension. Extrapolating it could
        # lead to large errors. TODO: gwcs's numerical inverse?
        if self.output_wave_scale == "reference" and not self.trim_spectral:
            log.warning("output_wave_scale='reference' is incompatible with "
                        "trim_spectral=False. Setting trim_spectral=True")
            self.trim_spectral = True


class separateSkyConfig(parameters_preprocess.separateSkyConfig):
    debug_allowable_perpendicular_offset = config.RangeField(
        "Maximum allowable offset perpendicular to the slit (arcsec)",
        float, None, min=0, inclusiveMin=False, optional=True)


class skyCorrectFromSlitConfig(config.core_1Dfitting_config):
    suffix = config.Field("Filename suffix", str, "_skyCorrected", optional=True)
    regions = config.Field("Sample regions. (eg. 100:150,251:264)", str, None, optional=True)
    aperture_growth = config.RangeField("Aperture avoidance distance (pixels)", float, 2, min=0)
    debug_plot = config.Field("Show diagnostic plots?", bool, False)
    interactive = config.Field("Run primitive interactively?", bool, False)
    debug_allow_skip = config.Field("Allow 'Skip' exit from interactive mode?", bool, False)

    def setDefaults(self):
        self.order = 5
        self.niter = 3
        self.grow = 2


class traceAperturesConfig(config.core_1Dfitting_config):
    """
    Configuration for the traceApertures() primitive.
    """
    suffix = config.Field("Filename suffix",
                          str, "_aperturesTraced", optional=True)
    max_missed = config.RangeField("Maximum number of steps to miss before a line is lost",
                                   int, 5, min=0)
    max_shift = config.RangeField("Maximum shift per pixel in line position",
                                  float, 0.05, min=0.001, max=0.1, inclusiveMax=True)
    nsum = config.RangeField("Number of lines to sum",
                             int, 10, min=1)
    step = config.RangeField("Step in rows/columns for tracing",
                             int, 10, min=1)
    interactive = config.Field("Run primitive interactively?",
                               bool, False)
    debug = config.Field("Draw aperture traces on image display?",
                         bool, False)

    def setDefaults(self):
        del self.function
        self.order = 2


class tracePinholeAperturesConfig(config.Config):
    """
    Configuration for the tracePinholeApertures() primitive.

    While the primitive itself should be useable with various modes, it has
    only been tested with cross-dispersed so far (September 2023). Parameters
    thoughtare therefore left unspecified here, and should be defined in parameter
    files more specific to the mode.
    """
    suffix = config.Field("Filename suffix",
                          str, "_pinholeAperturesTraced", optional=True)
    start_pos = config.RangeField("Row or column to start tracing at (default: halfway)",
                                  int, None, min=0, inclusiveMin=True, optional=True)
    max_missed = config.RangeField("Maximum number of steps to miss before a line is lost",
                                   int, 5, min=0)
    max_shift = config.RangeField("Maximum shift per pixel in line position",
                                  float, 0.05, min=0.001, max=0.3,
                                  inclusiveMax=True)
    min_line_length = config.RangeField("Minimum line length as a fraction of array",
                                        float, 0, min=0, max=1, inclusiveMin=True,
                                        inclusiveMax=True)
    min_snr = config.RangeField("Minimum SNR for apertures", float, 10., min=0.)
    nsum = config.RangeField("Number of lines to sum", int, 10, min=1)
    step = config.RangeField("Step in rows/columns for tracing", int, 10, min=1)
    spectral_order = config.RangeField("Order of fit in spectral direction",
                                       int, 3, min=1)
    # These exist in case excluding some of the pinhole traces is desired. This
    # is important for GNIRS but may not be the case for other instruments/
    # modes, so the defaults here are to use all traces found.
    debug_min_trace_pos = config.RangeField("First pinhole trace to use",
                                            dtype=int, default=None, min=1, optional=True)
    debug_max_trace_pos = config.RangeField("Last pinhole trace to use",
                                            dtype=int, default=None, min=1, optional=True)
    debug_plots = config.Field("Create diagnostic plots of traces", bool, False)

    def validate(self):
        if (self.debug_max_trace_pos is not None and
                self.debug_min_trace_pos is not None and
                self.debug_max_trace_pos < self.debug_min_trace_pos):
            raise ValueError("debug_max_trace_pos cannot be less than debug_min_trace_pos")


def wavelength_units_check(value):
    # Confirm that the specified units are suitable for wavelength or frequency
    try:
        unit = u.Unit(value)
    except:
        raise ValueError(f"{value} is not a recognized unit")
    try:
        unit.to(u.m)
    except u.UnitConversionError:
        raise ValueError(f"{value} is not a wavelength unit")
    return True


def wavelength_units_check(value):
    # Confirm that the specified units are suitable for wavelength or frequency
    try:
        unit = u.Unit(value)
    except:
        raise ValueError(f"{value} is not a recognized unit")
    try:
        unit.to(u.m)
    except u.UnitConversionError:
        raise ValueError(f"{value} is not a wavelength unit")
    return True


class write1DSpectraConfig(config.Config):
    #format = config.Field("Format for writing", str, "ascii")
    format = config.ChoiceField("Format for writing", str,
                                allowed=table_writing_formats(),
                                default="ascii", optional=False)
    header = config.Field("Write full FITS header?", bool, False)
    extension = config.Field("Filename extension", str, "dat")
    apertures = config.Field("Apertures to write", (str, int), None,
                             optional=True, check=list_of_ints_check)
    dq = config.Field("Write Data Quality values?", bool, False)
    var = config.Field("Write Variance values?", bool, False)
    overwrite = config.Field("Overwrite existing files?", bool, False)
    wave_units = config.Field("Output wavelength units", str, None,
                              check=wavelength_units_check, optional=True)
    # Cannot check as we don't know what the input units are
    data_units = config.Field("Output data units", str, None, optional=True)

    def validate(self):
        config.Config.validate(self)
        if self.header and not self.format.startswith("ascii"):
            raise ValueError("FITS header can only be written with ASCII formats")
