class GMOS_MOS_FLAT(DataClassification):
    name="GMOS_MOS_FLAT"
    usage = """
        Applies to all MOS flat datasets from the GMOS instruments
        """
    parent = "GMOS_MOS"
    requirement = AND([  ISCLASS("GMOS_MOS"),
                         PHU(OBSTYPE="FLAT"),
                         NOT(ISCLASS("GMOS_MOS_TWILIGHT"))  ])

newtypes.append(GMOS_MOS_FLAT())
