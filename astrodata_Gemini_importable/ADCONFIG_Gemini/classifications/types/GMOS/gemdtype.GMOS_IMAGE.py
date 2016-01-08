class GMOS_IMAGE(DataClassification):
    name="GMOS_IMAGE"
    usage = """
        Applies to all imaging datasets from the GMOS instruments
        """
    parent = "GMOS"
    requirement = AND([  ISCLASS("GMOS"),
                         PHU(GRATING="MIRROR"),
                         NOT(ISCLASS("GMOS_BIAS"))  ])

newtypes.append(GMOS_IMAGE())
