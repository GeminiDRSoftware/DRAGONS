class F2_IMAGE_FLAT(DataClassification):
    name="F2_IMAGE_FLAT"
    usage = """
        Applies to all imaging flat datasets from the FLAMINGOS-2 instrument
        """
    parent = "F2_IMAGE"
    requirement = AND([  ISCLASS("F2_IMAGE"),
                         PHU(OBSTYPE="FLAT"), 
                         NOT(ISCLASS("F2_IMAGE_TWILIGHT"))  ])

newtypes.append(F2_IMAGE_FLAT())
