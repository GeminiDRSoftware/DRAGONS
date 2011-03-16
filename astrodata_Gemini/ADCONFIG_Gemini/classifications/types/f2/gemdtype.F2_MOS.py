class F2_MOS(DataClassification):
    name="F2_MOS"
    usage = """
        Applies to all MOS datasets from the FLAMINGOS-2 instrument
        """
    parent = "F2_SPECT"
    requirement = ISCLASS("F2_SPECT") & OR([  PHU(DCKERPOS="mos"),
                                              PHU(MOSPOS="mos.?")  ])

newtypes.append(F2_MOS())
