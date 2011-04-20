class GMOS_IMAGE_TWILIGHT(DataClassification):
    name="GMOS_IMAGE_TWILIGHT"
    usage = """
        Applies to all imaging twilight flat datasets from the GMOS instruments
        """
    parent = "GMOS_IMAGE"
    requirement = AND([  ISCLASS("GMOS_IMAGE"),
                         PHU(OBSTYPE="FLAT"),
                         PHU(OBJECT="Twilight")  ])

newtypes.append(GMOS_IMAGE_TWILIGHT())
