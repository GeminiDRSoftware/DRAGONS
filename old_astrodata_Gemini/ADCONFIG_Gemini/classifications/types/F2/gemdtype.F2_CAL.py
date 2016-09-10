class F2_CAL(DataClassification):
    name="F2_CAL"
    usage = """
        Applies to all calibration datasets from the FLAMINGOS-2 instrument
        """
    parent = "F2"
    requirement = ISCLASS("F2") & OR([  ISCLASS("F2_IMAGE_FLAT"),
                                        ISCLASS("F2_IMAGE_TWILIGHT"),
                                        ISCLASS("F2_DARK"),
                                        ISCLASS("F2_LS_FLAT"),
                                        ISCLASS("F2_LS_TWILIGHT"),
                                        ISCLASS("F2_LS_ARC"),
                                        ISCLASS("F2_MOS_FLAT"),
                                        ISCLASS("F2_MOS_TWILIGHT"),
                                        ISCLASS("F2_MOS_ARC")  ])

newtypes.append(F2_CAL())
