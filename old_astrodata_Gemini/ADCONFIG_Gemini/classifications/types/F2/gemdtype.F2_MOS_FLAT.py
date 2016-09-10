class F2_MOS_FLAT(DataClassification):
    name="F2_MOS_FLAT"
    usage = """
        Applies to all MOS flat datasets from the FLAMINGOS-2 instrument
        """
    parent = "F2_MOS"
    requirement = AND([  ISCLASS("F2_MOS"),
                         PHU(OBSTYPE="FLAT"),
                         NOT(ISCLASS("F2_MOS_TWILIGHT"))  ])

newtypes.append(F2_MOS_FLAT())
