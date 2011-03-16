class F2_MOS_ARC(DataClassification):
    name="F2_MOS_ARC"
    usage = """
        Applies to all MOS arc datasets from the FLAMINGOS-2 instrument
        """
    parent = "F2_MOS"
    requirement = AND([  ISCLASS("F2_MOS"),
                         PHU(OBSTYPE="ARC")  ])

newtypes.append(F2_MOS_ARC())
