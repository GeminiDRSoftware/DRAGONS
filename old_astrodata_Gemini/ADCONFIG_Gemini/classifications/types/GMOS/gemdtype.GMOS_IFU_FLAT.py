class GMOS_IFU_FLAT(DataClassification):
    name="GMOS_IFU_FLAT"
    usage = """
        Applies to all IFU flat datasets from the GMOS instruments
        """
    parent = "GMOS_IFU"
    requirement = AND([  ISCLASS("GMOS_IFU"),
                         PHU(OBSTYPE="FLAT"),
                         NOT(ISCLASS("GMOS_IFU_TWILIGHT"))  ])

newtypes.append(GMOS_IFU_FLAT())
