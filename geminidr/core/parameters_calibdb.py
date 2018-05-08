# This parameter file contains the parameters related to the primitives located
# in primitives_calibdb.py, in alphabetical order.
from gempy.library import config

class addCalibrationConfig(config.Config):
    caltype = config.ChoiceField("Type of calibration required", str,
                                 allowed = {"processed_arc": "processed ARC",
                                            "processed_bias": "procsessed BIAS",
                                            "processed_dark": "processed DARK",
                                            "processed_flat": "processed FLAT",
                                            "processed_fringe": "processed fringe"}
                                 )
    calfile = config.Field("Filename of calibration", str)

class getCalibrationConfig(config.Config):
    caltype = config.ChoiceField("Type of calibration required", str,
                                 allowed = {"processed_arc": "processed ARC",
                                            "processed_bias": "procsessed BIAS",
                                            "processed_dark": "processed DARK",
                                            "processed_flat": "processed FLAT",
                                            "processed_fringe": "processed fringe"},
                                 optional=False)
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

class getMDFConfig(config.Config):
    pass

class storeCalibrationConfig(config.Config):
    caltype = config.ChoiceField("Type of calibration", str,
                                 allowed = {"processed_arc": "processed ARC",
                                            "processed_bias": "procsessed BIAS",
                                            "processed_dark": "processed DARK",
                                            "processed_flat": "processed FLAT",
                                            "processed_fringe": "processed fringe"},
                                 optional=False)

class storeProcessedArcConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_arc", optional=True)

class storeProcessedBiasConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_bias", optional=True)

class storeBPMConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_bpm", optional=True)

class storeProcessedDarkConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_dark", optional=True)

class storeProcessedFlatConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_flat", optional=True)

class storeProcessedFringeConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_fringe", optional=True)
