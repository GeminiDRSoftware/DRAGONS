class GMOS_MOS_ARC(DataClassification):
    name="GMOS_MOS_ARC"
    usage = """
        Applies to all MOS arc datasets from the GMOS instruments
        """
    parent = "GMOS_MOS"
    requirement = AND([  ISCLASS("GMOS_MOS"),
                         PHU(OBSTYPE="ARC")  ])

newtypes.append(GMOS_MOS_ARC())
