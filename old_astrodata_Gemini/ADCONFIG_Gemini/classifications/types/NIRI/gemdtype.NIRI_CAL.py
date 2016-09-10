class NIRI_CAL(DataClassification):
    name="NIRI_CAL"
    usage = """
        Applies to all calibration datasets from NIRI
        """
    parent = "NIRI"
    requirement = ISCLASS("NIRI") & OR([ ISCLASS("NIRI_IMAGE_FLAT"),
                                          ISCLASS("NIRI_PINHOLE"),
                                          ISCLASS("NIRI_DARK")
                                        ])

newtypes.append(NIRI_CAL())
