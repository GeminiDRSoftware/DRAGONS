# This parameter file contains the parameters related to the primitives located
# in the primitives_gemini.py file, in alphabetical order.
from gempy.library import config
from ..core import parameters_standardize


class addMDFConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_mdfAdded", optional=True)
    mdf = config.Field("Name of MDF", str, None, optional=True)


class standardizeStructureConfig(addMDFConfig):
    attach_mdf = config.Field("Attach MDF to spectroscopic data?", bool, True)

    def setDefaults(self):
        self.suffix = "_structureStandardized"

class standardizeWCSConfig(parameters_standardize.standardizeWCSConfig):
    bad_wcs = config.ChoiceField("Method for WCS handling", str,
                                 allowed={'exit': "Exit reduction if discrepant WCS found",
                                          'fix': "Attempt to fix discrepant WCS using offsets",
                                          'new': "Create new WCS from target coordinates and offsets",
                                          'ignore': "Do not check or fix the WCS"},
                                 default='exit')
    debug_consistency_limit = config.RangeField("Maximum discrepancy limit for pointing (arcsec)",
                                                float, 30., min=2)
    debug_max_deadtime = config.RangeField("Maximum dead time between exposures to require a new pointing",
                                           float, 60., min=10)


class checkWCSConfig(config.Config):
    pass
