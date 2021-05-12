# This parameter file contains the parameters related to the primitives located
# in the primitives_spect.py file, in alphabetical order.
from astropy import table, units as u
from astropy.io import registry

from astrodata import AstroData
from geminidr.core import parameters_generic
from gempy.library import config, astrotools as at


def list_of_ints_check(value):
    [int(x) for x in str(value).split(',')]
    return True


def table_writing_formats():
    t = registry.get_formats(table.Table, readwrite="Write")
    return {fmt: "" for fmt, dep in t["Format", "Deprecated"] if dep != "Yes"}


def validate_regions_float(value):
    at.parse_user_regions(value, dtype=float, allow_step=False)
    return True

def validate_regions_int(value):
    at.parse_user_regions(value, dtype=int, allow_step=True)
    return True


class adjustWCSToReferenceConfig(config.Config):
    suffix = config.Field("Filename suffix",
                          str, "_wcsCorrected", optional=True)
    method = config.ChoiceField("Alignment method", str,
                                allowed={"sources_wcs": "Match sources using WCS",
                                         "sources_offsets": "Match sources using telescope offsets",
                                         "offsets": "Use telescope offsets only"},
                                default="sources_wcs")
    fallback = config.ChoiceField("Fallback method", str,
                                  allowed={"sources_offsets": "Match sources using telescope offsets",
                                           "offsets": "Use telescope offsets only"},
                                  default="offsets", optional=True)
    tolerance = config.RangeField("Maximum distance from the header offset, "
                                  "for the correlation method (arcsec)",
                                  float, 1, min=0., optional=True)


class attachWavelengthSolutionConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_wavelengthSolutionAttached", optional=True)
    arc = config.ListField("Arc(s) with distortion map", (AstroData, str), None,
                           optional=True, single=True)


class calculateSensitivityConfig(config.core_1Dfitting_config):
    suffix = config.Field("Filename suffix", str, "_sensitivityCalculated", optional=True)
    filename = config.Field("Name of spectrophotometric data file", str, None, optional=True)
    bandpass = config.RangeField("Bandpass width (nm) if not supplied",
                                 float, 5., min=0.1, max=10.)
    debug_airmass0 = config.Field("Calculate sensitivity curve at zero airmass?",
                                  bool, False)
    regions = config.Field("Sample regions", str, None, optional=True,
                           check=validate_regions_float)
    debug_plot = config.Field("Plot sensitivity curve?", bool, False)
    interactive_spline = config.Field("Interactive spline selection?", bool, False)
    niter = config.RangeField("Maximum number of clipping iterations to perform of spline fit", int, 3, min=1)
    grow = config.RangeField("Radius to reject pixels adjacent to masked pixels of spline fit", int, 0, min=0)
    interactive = config.Field("Display interactive fitter?", bool, False)

    def setDefaults(self):
        del self.grow


class determineDistortionConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_distortionDetermined", optional=True)
    spatial_order = config.RangeField("Fitting order in spatial direction", int, 3, min=1)
    spectral_order = config.RangeField("Fitting order in spectral direction", int, 4, min=1)
    id_only = config.Field("Use only lines identified for wavelength calibration?", bool, False)
    min_snr = config.RangeField("Minimum SNR for peak detection", float, 5., min=3.)
    fwidth = config.RangeField("Feature width in pixels if reidentifying",
                               float, None, min=2., optional=True)
    nsum = config.RangeField("Number of lines to sum", int, 10, min=1)
    step = config.RangeField("Step in rows/columns for tracing", int, 10, min=1)
    max_shift = config.RangeField("Maximum shift per pixel in line position",
                                  float, 0.05, min=0.001, max=0.1)
    max_missed = config.RangeField("Maximum number of steps to miss before a line is lost", int, 5, min=0)
    debug = config.Field("Display line traces on image display?", bool, False)
    interactive = config.Field("Display interactive fitter?", bool, False)


class determineWavelengthSolutionConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_wavelengthSolutionDetermined", optional=True)
    order = config.RangeField("Order of fitting polynomial", int, 2, min=1)
    center = config.RangeField("Central row/column to extract", int, None, min=1, optional=True)
    nsum = config.RangeField("Number of lines to sum", int, 10, min=1)
    min_snr = config.RangeField("Minimum SNR for peak detection", float, 10., min=1.)
    min_sep = config.RangeField("Minimum feature separation (pixels)", float, 2., min=1.)
    weighting = config.ChoiceField("Weighting of identified peaks", str,
                                   allowed={"none": "no weighting",
                                            "natural": "natural weighting",
                                            "relative": "relative to local peaks"},
                                   default="natural")
    fwidth = config.RangeField("Feature width in pixels", float, None, min=2., optional=True)
    min_lines = config.Field("Minimum number of lines to fit each segment", (str, int), '15,20',
                             check=list_of_ints_check)
    central_wavelength = config.RangeField("Estimated central wavelength (nm)", float, None,
                                           min=300., max=25000., optional=True)
    dispersion = config.Field("Estimated dispersion (nm/pixel)", float, None, optional=True)
    linelist = config.Field("Filename of arc line list", str, None, optional=True)
    in_vacuo = config.Field("Use vacuum wavelength scale (rather than air)?", bool, False)
    alternative_centers = config.Field("Try alternative wavelength centers?", bool, False)
    debug = config.Field("Make diagnostic plots?", bool, False)


class distortionCorrectConfig(parameters_generic.calRequirementConfig):
    suffix = config.Field("Filename suffix", str, "_distortionCorrected", optional=True)
    order = config.RangeField("Interpolation order", int, 3, min=0, max=5, inclusiveMax=True)
    subsample = config.RangeField("Subsampling", int, 1, min=1)


class extract1DSpectraConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_extracted", optional=True)
    method = config.ChoiceField("Extraction method", str,
                                allowed={"standard": "no weighting",
                                         "optimal": "optimal extraction"},
                                default="standard")
    width = config.RangeField("Width of extraction aperture (pixels)", float, None, min=1, optional=True)
    grow = config.RangeField("Source aperture avoidance region (pixels)", float, 10, min=0, optional=True)
    interactive = config.Field("Perform extraction interactively", bool, False)
    subtract_sky = config.Field("Subtract sky spectra if the data have not been sky corrected?", bool, True)
    debug = config.Field("Draw extraction apertures on image display? (not used with interactive)", bool, False)


def check_section(value):
    # Check for validity of a section string
    subsections = value.split(',')
    for i, (x1, x2) in enumerate(s.split(':') for s in subsections):
        try:
            int(x1)
        except ValueError:
            if i > 0 or x1 != '':
                return False
            else:
                x1 = 0
        try:
            int(x2)
        except ValueError:
            if i < len(subsections) - 1 or x2 != '':
                return False
        else:
            if x2 <= x1:
                raise ValueError("Section(s) do not have end pixel number "
                                 "greater than start pixel number")
    return True

class findSourceAperturesConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_aperturesFound", optional=True)
    max_apertures = config.RangeField("Maximum number of sources to find",
                                      int, None, min=1, optional=True)
    percentile = config.RangeField("Percentile to determine signal for each spatial pixel",
                                   float, 95, min=1, max=100, optional=True)
    section = config.Field("Pixel section(s) for measuring the spatial profile",
                           str, None, optional=True, check=check_section)
    min_sky_region = config.RangeField("Minimum number of contiguous pixels "
                                       "between sky lines", int, 20, min=1)
    use_snr = config.Field("Use signal-to-noise ratio rather than data to find peaks?",
                           bool, True)
    threshold = config.RangeField("Threshold for automatic width determination",
                                  float, 0.1, min=0, max=1)
    sizing_method = config.ChoiceField("Method for automatic width determination", str,
                                       allowed={"peak": "height relative to peak",
                                                "integral": "integrated flux"},
                                       default="peak")
    interactive = config.Field("Use interactive interface", bool, False)


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
    x_order = config.Field(
        doc="Order for fitting and subtracting object continuum and sky line "
        "models, prior to running the main cosmic ray detection algorithm. "
        "When None, defaults are used, according to the image size (as in "
        "the IRAF task gemcrspec). When 0, no fit is done.",
        dtype=int,
        optional=True,
        default=None,
    )
    y_order = config.Field(
        doc="Order for fitting and subtracting object continuum and sky line "
        "models, prior to running the main cosmic ray detection algorithm. "
        "When None, defaults are used, according to the image size (as in "
        "the IRAF task gemcrspec). When 0, no fit is done.",
        dtype=int,
        optional=True,
        default=None,
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
        raise ValueError("{} is not a recognized unit".format(value))
    try:
        unit.to(u.W / u.m ** 3, equivalencies=u.spectral_density(1. * u.m))
    except u.UnitConversionError:
        raise ValueError("Cannot convert {} to a flux density".format(value))
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
    order = config.RangeField("Order of interpolation", int, 1, min=0, max=5, inclusiveMax=True)

    def validate(self):
        config.Config.validate(self)
        if [self.w1, self.w2, self.dw, self.npix].count(None) not in (1, 4):
            raise ValueError("Exactly 0 or 3 of w1, w2, dw, npix must be specified")
        if self.w1 is not None and self.w2 is not None and self.w2 <= self.w1:
            raise ValueError("Ending wavelength must be greater than starting wavelength")


class normalizeFlatConfig(config.core_1Dfitting_config):
    suffix = config.Field("Filename suffix", str, "_normalized", optional=True)
    center = config.RangeField("Central row/column to extract", int, None, min=1, optional=True)
    nsum = config.RangeField("Number of lines to sum", int, 10, min=1)

    def setDefaults(self):
        self.order = 20


class resampleToCommonFrameConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_align", optional=True)
    w1 = config.RangeField("Starting wavelength (nm)", float, None, min=0., optional=True)
    w2 = config.RangeField("Ending wavelength (nm)", float, None, min=0., optional=True)
    dw = config.RangeField("Dispersion (nm/pixel)", float, None, min=0.01, optional=True)
    npix = config.RangeField("Number of pixels in spectrum", int, None, min=2, optional=True)
    conserve = config.Field("Conserve flux?", bool, None, optional=True)
    order = config.RangeField("Order of interpolation", int, 1, min=0, max=5, inclusiveMax=True)
    trim_data = config.Field("Trim to field of view of reference image?", bool, False)
    force_linear = config.Field("Force linear wavelength solution?", bool, True)

    def validate(self):
        config.Config.validate(self)
        if [self.w1, self.w2, self.dw, self.npix].count(None) == 0:
            raise ValueError("Maximum 3 of w1, w2, dw, npix must be specified")
        if self.w1 is not None and self.w2 is not None and self.w2 <= self.w1:
            raise ValueError("Ending wavelength must be greater than starting wavelength")


class skyCorrectFromSlitConfig(config.core_1Dfitting_config):
    suffix = config.Field("Filename suffix", str, "_skyCorrected", optional=True)
    regions = config.Field("Sample regions", str, None, optional=True)
    aperture_growth = config.RangeField("Aperture avoidance distance (pixels)", float, 2, min=0)
    debug_plot = config.Field("Show diagnostic plots?", bool, False)

    def setDefaults(self):
        self.order = 5
        self.niter = 3
        self.grow = 2


class traceAperturesConfig(config.Config):
    """
    Configuration for the traceApertures() primitive.
    """
    debug = config.Field("Draw aperture traces on image display?",
                         bool, False)
    interactive = config.Field("Run primitive interactively?",
                               bool, False)
    max_missed = config.RangeField("Maximum number of steps to miss before a line is lost",
                                   int, 5, min=0)
    max_shift = config.RangeField("Maximum shift per pixel in line position",
                                  float, 0.05, min=0.001, max=0.1)
    niter = config.RangeField("Maximum number of rejection iterations",
                              int, None, min=0, optional=True)
    nsum = config.RangeField("Number of lines to sum",
                             int, 10, min=1)
    order = config.RangeField("Order of fitting function",
                              int, 2, min=1)
    sigma = config.RangeField("Rejection in sigma of fit",
                              float, 3, min=0, optional=True)
    step = config.RangeField("Step in rows/columns for tracing",
                             int, 10, min=1)
    suffix = config.Field("Filename suffix",
                          str, "_aperturesTraced", optional=True)



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

    def validate(self):
        config.Config.validate(self)
        if self.header and not self.format.startswith("ascii"):
            raise ValueError("FITS header can only be written with ASCII formats")
