from gempy.library import config

class calRequirementConfig(config.Config):
    do_cal = config.ChoiceField("Calibration requirement", str,
                    allowed={"procmode": "Use the default rules set by the processing"
                                         "mode.",
                             "force": "Require a calibration regardless of the"
                                      "processing mode.",
                             "skip": "Skip this correction, no calibration required."},
                    default="procmode")
