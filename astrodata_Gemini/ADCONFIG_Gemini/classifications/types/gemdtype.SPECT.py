class SPECT(DataClassification):
    name="SPECT"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = """
        Applies to all Gemini spectroscopy datasets
        """
    parent = "GENERIC"
    requirement = OR([  ISCLASS("F2_SPECT"),
                        ISCLASS("GMOS_SPECT"),
                        ISCLASS("GNIRS_SPECT"),
                        ISCLASS("MICHELLE_SPECT"),
                        ISCLASS("NIFS_SPECT"),
                        ISCLASS("NIRI_SPECT"),
                        ISCLASS("TRECS_SPECT"),
                        ISCLASS("BHROS"),
                        ISCLASS("GPI_SPECT"),
                        ISCLASS("GRACES")
                    ])

newtypes.append(SPECT())
