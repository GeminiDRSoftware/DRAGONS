class F2_IMAGE(DataClassification):
    name="F2_IMAGE"
    usage = """
        Applies to all imaging datasets from the FLAMINGOS-2 instrument
        """
    parent = "F2"
    requirement = AND([  ISCLASS("F2"),
                         PHU(GRISM="Open"),
                         NOT(ISCLASS("F2_DARK"))  ])

newtypes.append(F2_IMAGE())
