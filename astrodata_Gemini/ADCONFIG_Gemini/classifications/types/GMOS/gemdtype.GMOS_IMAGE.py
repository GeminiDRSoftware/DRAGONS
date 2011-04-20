class GMOS_IMAGE(DataClassification):
    name="GMOS_IMAGE"
    usage = """
        Applies to all imaging datasets from the GMOS instruments
        """
    parent = "GMOS"
    requirement = AND([  ISCLASS("GMOS"),
                         PHU(MASKTYP="0"),
                         PHU(GRATING="MIRROR")  ])

newtypes.append(GMOS_IMAGE())
