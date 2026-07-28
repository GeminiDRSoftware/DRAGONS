# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_spect.py file, in alphabetical order.
from astrodata import AstroData
from gempy.library import config
from geminidr.core import parameters_spect, parameters_preprocess
from geminidr.core import parameters_generic


class QECorrectConfig(parameters_generic.calRequirementConfig):
    suffix = config.Field("Filename suffix", str, "_QECorrected", optional=True)
    use_iraf = config.Field("Use IRAF polynomial fits for Hamamatsu CCDs?",
                            bool, False)

class findAcquisitionSlitsConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_acqSlitsAdded", optional=True)

class flagCosmicRaysConfig(parameters_spect.flagCosmicRaysConfig):
    spectral_order = config.Field(
        doc="Order for fitting and subtracting object continuum and sky line "
        "models, prior to running the main cosmic ray detection algorithm. "
        "When None, defaults are used, according to the image size. "
        "To control which fits are performed, use the bkgmodel parameter.",
        dtype=int,
        optional=True,
        default=None,
    )
    spatial_order = config.Field(
        doc="Order for fitting and subtracting object continuum and sky line "
        "models, prior to running the main cosmic ray detection algorithm. "
        "When None, defaults are used, according to the image size. "
        "To control which fits are performed, use the bkgmodel parameter.",
        dtype=int,
        optional=True,
        default=None,
    )

class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    nbright = config.RangeField("Number of bright lines to eliminate", int, 0, min=0)

    def setDefaults(self):
        self.weighting = "local"
        self.order = 3
        self._fields["central_wavelength"].max = 1200
        del self.absorption
        del self.wv_band
        del self.resolution
        del self.num_lines
