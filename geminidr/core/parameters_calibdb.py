"""
This parameter file contains the parameters related to the primitives located
in primitives_calibdb.py, in alphabetical order.
"""
from gempy.library import config


class setCalibrationConfig(config.Config):
    caltype = config.ChoiceField(
        "Type of calibration assigned",
        str,
        allowed={"processed_arc": "processed ARC",
                 "processed_bias": "procsessed BIAS",
                 "processed_dark": "processed DARK",
                 "processed_flat": "processed FLAT",
                 "processed_fringe": "processed fringe",
                 "processed_standard": "processed standard",
                 "processed_slitillum": "processed slitillum",
                 }
        )
    calfile = config.Field("Filename of calibration", str)


class getProcessedArcConfig(config.Config):
    pass


class getProcessedBiasConfig(config.Config):
    pass


class getProcessedDarkConfig(config.Config):
    pass


class getProcessedFlatConfig(config.Config):
    pass


class getProcessedFringeConfig(config.Config):
    pass


class getProcessedStandardConfig(config.Config):
    pass


class getProcessedSlitIllumConfig(config.Config):
    pass


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
                 "processed_standard": "processed standard",
                 "processed_slitillum": "processed slitillum",
                 },
        optional=False
    )


class storeProcessedArcConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_arc", optional=True)
    force = config.Field("Force input to be identified as an arc?", bool, False)


class storeProcessedBiasConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_bias", optional=True)
    force = config.Field("Force input to be identified as a bias?", bool, False)


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


class storeProcessedScienceConfig(config.Config):
    suffix = config.Field("Filename suffix", str, None, optional=True)


class storeProcessedStandardConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_standard", optional=True)


class storeProcessedSlitIllumConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_slitIllum", optional=True)
