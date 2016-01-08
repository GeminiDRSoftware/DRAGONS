class GMOS_LS_FLAT(DataClassification):
    name="GMOS_LS_FLAT"
    usage = """
        Applies to all longslit flat datasets from the GMOS instruments
        """
    parent = "GMOS_LS"
    requirement = AND([  ISCLASS("GMOS_LS"),
                         PHU(OBSTYPE="FLAT"),
                         NOT(ISCLASS("GMOS_LS_TWILIGHT"))  ])

newtypes.append(GMOS_LS_FLAT())
