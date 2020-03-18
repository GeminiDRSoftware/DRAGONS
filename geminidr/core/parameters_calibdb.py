"""
This parameter file contains the parameters related to the primitives located
in primitives_calibdb.py, in alphabetical order.
"""

from gempy.library import config


class addCalibrationConfig(config.Config):

    caltype = config.ChoiceField(
        "Type of calibration required",
        str,
        allowed={"processed_arc": "processed ARC",
                 "processed_bias": "procsessed BIAS",
                 "processed_dark": "processed DARK",
                 "processed_flat": "processed FLAT",
                 "processed_fringe": "processed fringe",
                 "processed_standard": "processed standard"
                 }
        )

    calfile = config.Field("Filename of calibration", str)


class getCalibrationConfig(config.Config):

    caltype = config.ChoiceField(
        "Type of calibration required",
        str,
        allowed={"processed_arc": "processed ARC",
                 "processed_bias": "procsessed BIAS",
                 "processed_dark": "processed DARK",
                 "processed_flat": "processed FLAT",
                 "processed_fringe": "processed fringe",
                 "processed_standard": "processed standard"
                 },
        optional=False
    )

    refresh = config.Field(
        "Refresh existing calibration associations?", bool, True)

    howmany = config.RangeField(
        "Maximum number of calibrations to return", int, None, min=1, optional=True)


class getProcessedArcConfig(config.Config):

    refresh = config.Field(
        "Refresh existing calibration associations?", bool, True)


class getProcessedBiasConfig(config.Config):

    refresh = config.Field(
        "Refresh existing calibration associations?", bool, True)


class getProcessedDarkConfig(config.Config):

    refresh = config.Field(
        "Refresh existing calibration associations?", bool, True)


class getProcessedFlatConfig(config.Config):

    refresh = config.Field(
        "Refresh existing calibration associations?", bool, True)


class getProcessedFringeConfig(config.Config):
    refresh = config.Field("Refresh existing calibration associations?", bool, True)


class getMDFConfig(config.Config):
    pass


class storeCalibrationConfig(config.Config):

    caltype = config.ChoiceField(
        "Type of calibration", str,
        allowed={"processed_arc": "processed ARC",
                 "processed_bias": "procsessed BIAS",
                 "processed_dark": "processed DARK",
                 "processed_flat": "processed FLAT",
                 "processed_fringe": "processed fringe",
                 "bpm": "bad pixel mask",
                 "sq": "science quality",
                 "qa": "QA",
                 "ql": "quick look",
                 "processed_standard": "processed standard"
                 },
        optional=False
    )


class storeProcessedArcConfig(config.Config):

    suffix = config.Field("Filename suffix", str, "_arc", optional=True)
    force = config.Field(
        "Force input to be identified as an arc?", bool, False)


class storeProcessedBiasConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_bias", optional=True)
    force = config.Field(
        "Force input to be identified as a bias?", bool, False)

class storeBPMConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_bpm", optional=True)


class storeProcessedDarkConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_dark", optional=True)
    force = config.Field("Force input to be identified as a dark?",
                         bool, False)

class storeProcessedFlatConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_flat", optional=True)
    force = config.Field("Force input to be identified as a flat?",
                         bool, False)

class storeProcessedFringeConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_fringe", optional=True)

class storeScienceConfig(config.Config):
    pass

class storeProcessedStandardConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_standard", optional=True)
