class GMOS_LS_TWILIGHT(DataClassification):
    name="GMOS_LS_TWILIGHT"
    usage = """
        Applies to all longslit twilight flat datasets from the GMOS
        instruments
        """
    parent = "GMOS_LS"
    requirement = AND([  ISCLASS("GMOS_LS"),
                         PHU(OBSTYPE="FLAT"),
                         PHU(OBJECT="Twilight")  ])

newtypes.append(GMOS_LS_TWILIGHT())
