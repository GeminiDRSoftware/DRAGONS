class F2_IMAGE_TWILIGHT(DataClassification):
    name="F2_IMAGE_TWILIGHT"
    usage = """
        Applies to all imaging twilight flat datasets from the FLAMINGOS-2
        instrument
        """
    parent = "F2_IMAGE"
    requirement = AND([  ISCLASS("F2_IMAGE"),
                         PHU(OBSTYPE="FLAT"),
                         PHU(OBJECT="Twilight")  ])

newtypes.append(F2_IMAGE_TWILIGHT())
