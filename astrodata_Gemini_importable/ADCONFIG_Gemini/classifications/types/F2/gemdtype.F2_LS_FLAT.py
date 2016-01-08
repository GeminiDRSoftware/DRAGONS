class F2_LS_FLAT(DataClassification):
    name="F2_LS_FLAT"
    usage = """
        Applies to all longslit flat datasets from the FLAMINGOS-2 instrument
        """
    parent = "F2_LS"
    requirement = AND([  ISCLASS("F2_LS"),
                         PHU(OBSTYPE="FLAT"),
                         NOT(ISCLASS("F2_LS_TWILIGHT"))  ])

newtypes.append(F2_LS_FLAT())
