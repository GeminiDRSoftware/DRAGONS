class GMOS_IFU_TWILIGHT(DataClassification):
    name="GMOS_IFU_TWILIGHT"
    usage = """
        Applies to all IFU twilight flat datasets from the GMOS instruments
        """
    parent = "GMOS_IFU"
    requirement = AND([  ISCLASS("GMOS_IFU"),
                         PHU(OBSTYPE="FLAT"),
                         PHU(OBJECT="Twilight")  ])

newtypes.append(GMOS_IFU_TWILIGHT())
