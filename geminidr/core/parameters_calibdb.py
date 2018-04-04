# This parameter file contains the parameters related to the primitives located
# in primitives_calibdb.py, in alphabetical order.
from gempy.library import config

class addCalibration(config.Config):
    caltype = config.ChoiceField("Type of calibration required", str,
                                 allowed = {"processed_arc": "processed ARC",
                                            "processed_bias": "procsessed BIAS",
                                            "processed_dark": "processed DARK",
                                            "processed_flat": "processed FLAT",
                                            "processed_fringe": "processed fringe"}
                                 )
    calfile = config.Field("Filename of calibration", str)

class getCalibration(config.Config):
    caltype = config.ChoiceField("Type of calibration required", str,
                                 allowed = {"processed_arc": "processed ARC",
                                            "processed_bias": "procsessed BIAS",
                                            "processed_dark": "processed DARK",
                                            "processed_flat": "processed FLAT",
                                            "processed_fringe": "processed fringe"})
    refresh = config.Field("Refresh existing calibration associations?", bool, True)
    howmany = config.RangeField("Maximum number of calibrations to return", int, None, min=1, optional=True)

class getProcessedArcConfig(config.Config):
    refresh = config.Field("Refresh existing calibration associations?", bool, True)

class getProcessedBiasConfig(config.Config):
    refresh = config.Field("Refresh existing calibration associations?", bool, True)

class getProcessedDarkConfig(config.Config):
    refresh = config.Field("Refresh existing calibration associations?", bool, True)

class getProcessedFlatConfig(config.Config):
    refresh = config.Field("Refresh existing calibration associations?", bool, True)

class getProcessedFringeConfig(config.Config):
    refresh = config.Field("Refresh existing calibration associations?", bool, True)

class storeProcessedArcConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_arc")

class storeProcessedBiasConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_bias")

class storeProcessedDarkConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_dark")

class storeProcessedFlatConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_flat")

class storeProcessedFringeConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_fringe")

class storeCalibrationConfig(config.Config):
    caltype = config.ChoiceField("Type of calibration", str,
                                  allowed = {"processed_arc": "processed ARC",
                                            "processed_bias": "procsessed BIAS",
                                            "processed_dark": "processed DARK",
                                            "processed_flat": "processed FLAT",
                                            "processed_fringe": "processed fringe"})
