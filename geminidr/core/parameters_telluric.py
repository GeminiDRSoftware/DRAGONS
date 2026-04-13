from gempy.library import config
from . import parameters_generic, parameters_spect
from astrodata import AstroData
from gempy.library.astrotools import Magnitude


def validate_magstr(value):
    try:
        Magnitude(value)
    except (IndexError, ValueError):
        return False
    return True


class divideByTelluricConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_telluricDivided", optional=True)
    telluric = config.ListField("File with telluric model", (AstroData, str), None,
                                optional=True, single=True)
    pixel_shift = config.RangeField("Shift in pixels", float, None, min=-5, max=5,
                                     inclusiveMin=True, inclusiveMax=True, optional=True)


class fitTelluricConfig(config.core_1Dfitting_config):
    suffix = config.Field("Filename suffix", str, "_telluricFitted", optional=True)
    bbtemp = config.RangeField("Stellar blackbody temperature", float, 9650,
                               min=3000, max=50000)
    magnitude = config.Field("Magnitude normalization", str, "K=10",
                             check=validate_magstr)
    abmag = config.Field("Magnitude is AB (rather than Vega)?", bool, False)
    lsf_scaling = config.RangeField("LSF scaling factor", float, None,
                                    min=0.5, max=2.0, optional=True)
    regions = config.Field("Wavelength sample regions (nm)", str, None, optional=True,
                           check=parameters_spect.validate_regions_float)
    order = config.RangeField("Order of fitting function", int, 6, min=1, max=30,
                       inclusiveMax=True)
    interactive = config.Field("Display interactive fitter?", bool, False)
    weighting = config.ChoiceField("Weighting scheme", str,
                                   allowed={"variance": "Inverse variance",
                                            "uniform": "Uniform weights"},
                                   default="variance", optional=False)
    shift_tolerance = config.RangeField(
        "Tolerance (pixels) for ignoring cross-correlation shift",
        float, None, min=0, max=5, inclusiveMax=True, optional=True)
    apply_shift = config.Field("Permanently apply pixel shift?", bool, True)  # debug?
    debug_lsf_sampling = config.RangeField("Number of sample points for each LSF parameter",
                                           int, 5, min=3, optional=False)
    debug_stellar_mask_threshold = config.RangeField(
        "Absorption threshold for stellar mask", float, 0.5, min=0., max=1.0)
    debug_stellar_mask_max_extent = config.RangeField(
        "Maximum extent of a stellar mask region (nm)", float, 40., min=0.)

    def setDefaults(self):
        self.niter = 1


class telluricCorrectConfig(parameters_generic.calRequirementConfig):
    suffix = config.Field("Filename suffix", str, "_telluricCorrected", optional=True)
    telluric = config.ListField("File with telluric model", (AstroData, str), None,
                                optional=True, single=True)
    apply_model = config.Field("Apply the synthetic absorption rather than the data?",
                               bool, True, optional=False)
    interactive = config.Field("Tweak correction interactively?", bool, False)
    pixel_shift = config.RangeField("Shift in pixels", float, None, min=-5, max=5,
                                     inclusiveMin=True, inclusiveMax=True, optional=True)
    delta_airmass = config.RangeField("Airmass difference", float, None, min=-1, max=1,
                                      inclusiveMin=True, inclusiveMax=True, optional=True)
    shift_tolerance = config.RangeField(
        "Tolerance (pixels) for ignoring cross-correlation shift",
        float, None, min=0, max=5, inclusiveMax=True, optional=True)
    apply_shift = config.Field("Permanently apply pixel shift?", bool, True)  # debug?
