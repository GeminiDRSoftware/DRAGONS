class LS(DataClassification):
    name="LS"
    usage = """
        Applies to all Gemini long slit spectroscopy datasets
        """
    parent = "SPECT"
    requirement = OR([  ISCLASS("F2_LS"),
                        ISCLASS("GMOS_LS"),
                        ISCLASS("NIRI_SPECT"),
                        ISCLASS("GNIRS_LS"),
                        ISCLASS("MICHELLE_SPECT"),
                        ISCLASS("TRECS_SPECT")])

newtypes.append(LS())
