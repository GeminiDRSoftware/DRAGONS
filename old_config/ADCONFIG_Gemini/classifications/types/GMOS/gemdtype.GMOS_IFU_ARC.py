class GMOS_IFU_ARC(DataClassification):
    name="GMOS_IFU_ARC"
    usage = """
        Applies to all IFU arc datasets from the GMOS instruments
        """
    parent = "GMOS_IFU"
    requirement = AND([  ISCLASS("GMOS_IFU"),
                         PHU(OBSTYPE="ARC")  ])

newtypes.append(GMOS_IFU_ARC())
