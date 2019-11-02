# This parameter file contains the parameters related to the primitives located
# in the primitives_spect.py file, in alphabetical order.
from gempy.library import config
from astrodata import AstroData
from astropy import units as u

def flux_units_check(value):
    # Confirm that the specified units can be converted to a flux density
    try:
        unit = u.Unit(value)
    except:
        raise ValueError("{} is not a recognized unit".format(value))
    try:
        unit.to(u.W / u.m**3, equivalencies=u.spectral_density(1.*u.m))
    except:
        raise ValueError("Cannot convert {} to a flux density".format(value))
    return True

class fluxCalibrateConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_fluxCalibrated", optional=True)
    standard = config.ListField("Standard(s) with sensitivity function", (AstroData, str),
                                None, optional=True, single=True)
    units = config.Field("Units for output spectrum", str, "W m-2 nm-1",
                         check=flux_units_check)

class calculateSensitivityConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_sensitivityCalculated", optional=True)
    order = config.RangeField("Order of spline fit", int, 6, min=1)

class determineDistortionConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_distortionDetermined", optional=True)
    spatial_order = config.RangeField("Fitting order in spatial direction", int, 3, min=1)
    spectral_order = config.RangeField("Fitting order in spectral direction", int, 4, min=1)
    id_only = config.Field("Use only lines identified for wavelength calibration?", bool, True)
    min_snr = config.RangeField("Minimum SNR for peak detection", float, 5., min=3.)
    fwidth = config.RangeField("Feature width in pixels if reidentifying",
                               float, None, min=2., optional=True)
    nsum = config.RangeField("Number of lines to sum", int, 10, min=1)
    step = config.RangeField("Step in rows/columns for tracing", int, 10, min=1)
    max_shift = config.RangeField("Maximum shift per pixel in line position",
                                  float, 0.05, min=0.001, max=0.1)
    max_missed = config.RangeField("Maximum number of steps to miss before a line is lost", int, 5, min=0)

class determineWavelengthSolutionConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_wavelengthSolutionDetermined", optional=True)
    center = config.RangeField("Central row/column to extract", int, None, min=1, optional=True)
    nsum = config.RangeField("Number of lines to sum", int, 10, min=1)
    min_snr = config.RangeField("Minimum SNR for peak detection", float, 10., min=3.)
    weighting = config.ChoiceField("Weighting of identified peaks", str,
                                   allowed={"none": "no weighting",
                                            "natural": "natural weighting",
                                            "relative": "relative to local peaks"},
                                   default="natural")
    fwidth = config.RangeField("Feature width in pixels", float, None, min=2., optional=True)
    order = config.RangeField("Order of fitting polynomial", int, 2, min=1)
    central_wavelength = config.RangeField("Estimated central wavelength (nm)", float, None,
                                           min=300., max=25000., optional=True)
    dispersion = config.Field("Estimated dispersion (nm/pixel)", float, None, optional=True)
    linelist = config.Field("Filename of arc line list", str, None, optional=True)
    plot = config.Field("Make diagnostic plots?", bool, False)

class distortionCorrectConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_distortionCorrected", optional=True)
    arc = config.ListField("Arc(s) with distortion map", (AstroData, str), None,
                           optional=True, single=True)
    order = config.RangeField("Interpolation order", int, 3, min=0, max=5)
    subsample = config.RangeField("Subsampling", int, 1, min=1)

class extract1DSpectraConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_extracted", optional=True)
    method = config.ChoiceField("Extraction method", str,
                                allowed = {"standard": "no weighting",
                                           "weighted": "inverse-variance weighted",
                                           "optimal": "optimal extraction"},
                                default="standard")
    width = config.RangeField("Width of extraction aperture (pixels)", float, 10, min=1, optional=True)
    grow = config.RangeField("Source aperture avoidance region (pixels)", float, 10, min=0, optional=True)

class findSourceAperturesConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_aperturesFound", optional=True)
    max_apertures = config.RangeField("Maximum number of sources to find", int, None, min=1, optional=True)
    threshold = config.RangeField("Threshold relative to peak for automatic width determination", float, 0.01, min=0, max=1)
    min_sky_region = config.RangeField("Minimum number of contiguous pixels between sky lines", int, 20, min=1)


class linearizeSpectraConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_linearized", optional=True)
    w1 = config.RangeField("Starting wavelength (nm)", float, None, min=0., optional=True)
    w2 = config.RangeField("Ending wavelength (nm)", float, None, min=0., optional=True)
    dw = config.RangeField("Dispersion (nm/pixel)", float, None, min=0.01, optional=True)
    npix = config.RangeField("Number of pixels in spectrum", int, None, min=2, optional=True)
    conserve = config.Field("Conserve flux?", bool, False)

    def validate(self):
        config.Config.validate(self)
        if [self.w1, self.w2, self.dw, self.npix].count(None) not in (1, 4):
            raise ValueError("Exactly 0 or 3 of w1, w2, dw, npix must be specified")
        if self.w1 is not None and self.w2 is not None and self.w2 <= self.w1:
            raise ValueError("Ending wavelength must be greater than starting wavelength")

class normalizeFlatConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_normalized", optional=True)
    center = config.RangeField("Central row/column to extract", int, None, min=1, optional=True)
    nsum = config.RangeField("Number of lines to sum", int, 10, min=1)
    spectral_order = config.RangeField("Fitting order in spectral direction", int, 20, min=1)

class traceAperturesConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_aperturesTraced", optional=True)
    trace_order = config.RangeField("Fitting order in spectral direction", int, 2, min=1)
    nsum = config.RangeField("Number of lines to sum", int, 10, min=1)
    step = config.RangeField("Step in rows/columns for tracing", int, 10, min=1)
    max_shift = config.RangeField("Maximum shift per pixel in line position",
                                  float, 0.05, min=0.001, max=0.1)
    max_missed = config.RangeField("Maximum number of steps to miss before a line is lost", int, 5, min=0)

class skyCorrectFromSlitConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_skyCorrected", optional=True)
    order = config.RangeField("Sky spline fitting order", int, 5, min=1, optional=True)
    grow = config.RangeField("Aperture growth distance (pixels)", float, 0, min=0)
