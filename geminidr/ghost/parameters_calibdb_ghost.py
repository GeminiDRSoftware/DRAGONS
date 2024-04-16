# This parameter file contains the parameters related to the primitives located
# in the primitives_calibdb_ghost.py file, in alphabetical order.

from gempy.library import config
from geminidr.core import parameters_calibdb


#class getProcessedArcConfig(parameters_calibdb.getProcessedArcConfig):
#    howmany = config.Field("How many arcs to return", int, 1, optional=True)


class getProcessedSlitConfig(config.Config):
    pass


class getProcessedSlitFlatConfig(config.Config):
    pass


class storeCalibrationConfig(parameters_calibdb.storeCalibrationConfig):
    caltype = config.ChoiceField("Type of calibration", str,
                                 allowed={"processed_arc": "processed ARC",
                                          "processed_bias": "procsessed BIAS",
                                          "processed_bpm": "processed BPM",
                                          "processed_dark": "processed DARK",
                                          "processed_flat": "processed FLAT",
                                          "processed_slitflat": "processed slit-viewer flat",
                                          "processed_slit": "processed slit-viewer",
                                          "processed_standard": "processed standard"},
                                 optional=False)


class storeProcessedSlitConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_slit", optional=True)


class storeProcessedSlitFlatConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_slitflat", optional=True)


#class storeProcessedSlitBiasConfig(config.Config):
#    suffix = config.Field("Filename suffix", str, "_slitbias", optional=True)


#class storeProcessedSlitDarkConfig(config.Config):
#    suffix = config.Field("Filename suffix", str, "_slitdark", optional=True)
