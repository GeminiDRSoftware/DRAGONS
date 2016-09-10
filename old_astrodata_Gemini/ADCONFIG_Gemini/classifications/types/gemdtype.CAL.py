class CAL(DataClassification):
    name="CAL"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = """
        Special parent to group generic types (e.g. IMAGE, SPECT, MOS, IFU)
        """
    parent = "GENERIC"
    requirement = OR([  ISCLASS("F2_CAL"),
                        ISCLASS("GMOS_CAL"),
                        ISCLASS("GNIRS_CAL"),
                        ISCLASS("GSAOI_CAL"),
                        ISCLASS("NICI_CAL"),
                        ISCLASS("NIRI_CAL")])

newtypes.append(CAL())
