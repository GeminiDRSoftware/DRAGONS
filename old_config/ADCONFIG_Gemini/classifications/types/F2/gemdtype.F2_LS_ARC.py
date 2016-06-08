class F2_LS_ARC(DataClassification):
    name="F2_LS_ARC"
    usage = """
        Applies to all longslit arc datasets from the FLAMINGOS-2 instrument
        """
    parent = "F2_LS"
    requirement = AND([  ISCLASS("F2_LS"),
                         PHU(OBSTYPE="ARC")  ])

newtypes.append(F2_LS_ARC())
