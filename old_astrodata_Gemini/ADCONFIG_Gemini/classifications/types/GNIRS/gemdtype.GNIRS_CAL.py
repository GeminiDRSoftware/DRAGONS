class GNIRS_CAL(DataClassification):
    name="GNIRS_CAL"
    usage = """
        Applies to all calibration datasets from GNIRS
        """
    parent = "GNIRS"
    requirement = ISCLASS("GNIRS") & OR([ ISCLASS("GNIRS_IMAGE_FLAT"),
                                          ISCLASS("GNIRS_PINHOLE"),
                                          ISCLASS("GNIRS_DARK")
                                        ])

newtypes.append(GNIRS_CAL())